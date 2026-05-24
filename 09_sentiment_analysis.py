"""
FinBERT 向量情感分析脚本

加载 sliding_finbert_full.npz 的 768 维 embedding，
用 FinBERT 的分类头权重（Linear(768,3)）做 3 分类，
分析滑窗 mean pooling 后向量的情感分布。

用法:
  python 09_sentiment_analysis.py
  python 09_sentiment_analysis.py --embedding sliding_finbert_full.npz
  python 09_sentiment_analysis.py --mode detail   # 额外输出窗口级统计
  python 09_sentiment_analysis.py --max-samples 500
"""

import os
import argparse
import warnings
import numpy as np
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NPZ = os.path.join(DATA_DIR, "sliding_finbert_full.npz")
DEFAULT_OUTPUT = os.path.join(DATA_DIR, "sentiment_by_date.csv")
DEFAULT_OUTPUT_DETAIL = os.path.join(DATA_DIR, "sentiment_detail.csv")

ID2LABEL = {0: "Neutral", 1: "Positive", 2: "Negative"}
LABEL2ID = {"Neutral": 0, "Positive": 1, "Negative": 2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", default=DEFAULT_NPZ)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--output-detail", default=DEFAULT_OUTPUT_DETAIL)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--mode", choices=["simple", "detail"], default="simple")
    args = parser.parse_args()

    print("=" * 60)
    print("FinBERT 向量情感分析（用分类头对 mean pooling 向量做 3 分类）")
    print("=" * 60)

    # 1. 加载 embedding
    print(f"\n[1/3] 加载 embedding: {args.embedding}")
    data = np.load(args.embedding, allow_pickle=True)
    emb = data["finbert_emb"]   # (N, 768)
    labels = data["labels"]
    dates = data["dates"]

    if args.max_samples:
        emb = emb[:args.max_samples]
        labels = labels[:args.max_samples]
        dates = dates[:args.max_samples]

    n = len(emb)
    print(f"  -> {n} 条, 维度 {emb.shape[1]}")

    # 2. 加载分类头权重
    print(f"\n[2/3] 加载 FinBERT 分类头权重: yiyanghkust/finbert-tone-chinese")
    from transformers import AutoModelForSequenceClassification
    import torch
    import torch.nn.functional as F

    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone-chinese")
    classifier = model.classifier  # Linear(768, 3)
    print(f"  分类头: {classifier}")

    # 用 PyTorch 做推理
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.to(device)
    classifier.eval()

    emb_tensor = torch.tensor(emb, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = classifier(emb_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)

    print(f"  -> 推理完成, shape={probs.shape}")

    # 3. 统计
    print(f"\n[3/3] 统计情感分布...")

    # 整体统计
    overall = Counter(preds)
    print(f"\n  整体情感分布 ({n} 条):")
    for label_id in [0, 1, 2]:
        name = ID2LABEL[label_id]
        count = overall[label_id]
        pct = count / n * 100
        avg_conf = probs[preds == label_id, label_id].mean() if count > 0 else 0
        print(f"    {name:<10} {count:>6} ({pct:>5.1f}%)  平均置信度: {avg_conf:.4f}")

    # 按 up/down 分组统计
    print(f"\n  按 up/down 分组:")
    for target_label_name, target_label_val in [("down (0)", 0), ("up (1)", 1)]:
        mask = labels == target_label_val
        subset_probs = probs[mask]
        subset_preds = preds[mask]
        sub_n = len(subset_preds)
        if sub_n == 0:
            continue
        sub_dist = Counter(subset_preds)
        print(f"\n    {target_label_name} ({sub_n} 条):")
        for label_id in [0, 1, 2]:
            name = ID2LABEL[label_id]
            count = sub_dist[label_id]
            pct = count / sub_n * 100
            print(f"      {name:<10} {count:>6} ({pct:>5.1f}%)")

    # 按日期统计
    date_stats = defaultdict(lambda: Counter())
    date_label_dist = defaultdict(lambda: {"up": 0, "down": 0})

    for pred, date, label in zip(preds, dates, labels):
        date_stats[str(date)][int(pred)] += 1
        date_label_dist[str(date)]["up" if label else "down"] += 1

    print(f"\n  按日期统计:")
    header = (f"  {'日期':<10} {'样本数':>6} {'Positive':>9} {'Neutral':>8} "
              f"{'Negative':>8} {'Pos%':>6} {'Neg%':>6} {'up':>5} {'down':>7}")
    print(header)
    print(f"  {'-' * len(header)}")

    rows_out = []
    for date in sorted(date_stats):
        c = date_stats[date]
        total = sum(c.values())
        pos_pct = c[LABEL2ID["Positive"]] / total * 100
        neg_pct = c[LABEL2ID["Negative"]] / total * 100
        ld = date_label_dist[date]
        print(f"  {date:<10} {total:>6} {c[1]:>9} {c[0]:>8} {c[2]:>8} "
              f"{pos_pct:>5.1f}% {neg_pct:>5.1f}% "
              f"{ld['up']:>5} {ld['down']:>7}")
        rows_out.append({
            "date": date,
            "total": total,
            "positive": c[LABEL2ID["Positive"]],
            "neutral": c[LABEL2ID["Neutral"]],
            "negative": c[LABEL2ID["Negative"]],
            "pos_pct": round(pos_pct, 1),
            "neg_pct": round(neg_pct, 1),
            "up": ld["up"],
            "down": ld["down"],
        })

    # 保存按日期统计
    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=[
            "date", "total", "positive", "neutral", "negative",
            "pos_pct", "neg_pct", "up", "down",
        ])
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\n  => 已保存: {args.output}")

    # 保存每条记录详细结果
    detail_rows = []
    for i in range(n):
        detail_rows.append({
            "idx": i,
            "date": str(dates[i]),
            "actual_label": "up" if labels[i] else "down",
            "sentiment": ID2LABEL[int(preds[i])],
            "prob_neg": round(float(probs[i][0]), 4),
            "prob_neu": round(float(probs[i][1]), 4),
            "prob_pos": round(float(probs[i][2]), 4),
        })

    with open(args.output_detail, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "idx", "date", "actual_label", "sentiment",
            "prob_neg", "prob_neu", "prob_pos",
        ])
        writer.writeheader()
        writer.writerows(detail_rows)
    print(f"  => 详细结果: {args.output_detail}")

    # detail 模式：额外输出概率分布统计
    if args.mode == "detail":
        print(f"\n  概率分布统计:")
        for i, name in enumerate(["neg", "neu", "pos"]):
            col = probs[:, i]
            print(f"    {name:<6} mean={col.mean():.4f}  std={col.std():.4f}  "
                  f"p5={np.percentile(col,5):.4f}  p50={np.percentile(col,50):.4f}  "
                  f"p95={np.percentile(col,95):.4f}")

        # 相关性：pos_prob vs label
        from scipy.stats import pearsonr
        corr_pos, p_pos = pearsonr(probs[:, 2], labels)
        corr_neg, p_neg = pearsonr(probs[:, 0], labels)
        print(f"\n  情感概率 vs label 相关性:")
        print(f"    pos_prob vs label: r={corr_pos:.4f}  p={p_pos:.4f}")
        print(f"    neg_prob vs label: r={corr_neg:.4f}  p={p_neg:.4f}")

    print(f"\n{'=' * 60}")
    print("完成\n")


if __name__ == "__main__":
    main()
