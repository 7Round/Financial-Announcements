"""
滑窗 FinBERT 极端值/加权特征提取（方案一 + 方案二）

方案一：概率空间 6 维可解释特征
  [max_pos, max_neg, mean_pos, mean_neg, sentiment_std, neutral_ratio]

方案二：embedding 空间注意力加权 768 维
  用 (1 - p_neutral) 作为权重对窗口 CLS 向量做加权平均

用法:
  # 全量（已有 sliding_finbert_full.npz 时直接 pkl）
  python 10_extreme_sliding.py --mode full

  # 小批量快速测试
  python 10_extreme_sliding.py --mode mini

输出:
  extreme_features.npz — {extreme6, weighted768, labels, prices, dates}
"""

import os
import csv
import argparse
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")
csv.field_size_limit(2**31 - 1)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(DATA_DIR, "train_text_full.csv")
EXISTING_NPZ = os.path.join(DATA_DIR, "sliding_finbert_full.npz")


def extract_extreme_features(args):
    """滑窗 + 分类头: 返回 (extreme6, weighted768, labels, prices, dates)"""
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(DATA_DIR, "hf_cache")
    os.environ["HF_HOME"] = os.path.join(DATA_DIR, "hf_cache")

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    import torch.nn.functional as F

    # 1. 加载数据
    print(f"加载数据: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if len(r["ann_text"].strip()) > 50]
    if args.max_samples:
        rows = rows[:args.max_samples]
    texts = [r["ann_text"] for r in rows]
    labels_raw = [1 if r["label"] == "up" else 0 for r in rows]
    prices_raw = [float(r["clpr"]) for r in rows]
    dates_raw = [r["date"] for r in rows]
    print(f"  -> {len(rows)} 条")

    # 2. 加载带分类头的 FinBERT
    print(f"加载 FinBERT (带分类头): yiyanghkust/finbert-tone-chinese")
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone-chinese")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone-chinese")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"  -> device: {device}")

    # 3. 滑窗提取
    print(f"滑窗提取 (window={args.max_length}, stride={args.stride})...")

    all_extreme6 = []   # (N, 6)
    all_weighted768 = []  # (N, 768)
    start_time = time.time()

    for idx, text in enumerate(texts):
        encoded = tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=args.max_length,
            return_overflowing_tokens=True,
            stride=args.stride,
            return_tensors=None,
        )
        input_ids_list = encoded["input_ids"]
        attn_mask_list = encoded["attention_mask"]
        n_windows = len(input_ids_list)

        if n_windows == 0:
            # 单窗口
            enc = tokenizer(
                text, truncation=True, padding=True,
                max_length=args.max_length, return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                out = model(**enc)
            probs = F.softmax(out.logits, dim=1).cpu().numpy()[0]  # [neg, neu, pos]
            cls_vec = out.logits.new_zeros((1, 768)).cpu().numpy()[0]  # dummy
            # Fallback: 用 hidden_state CLS
            del out, enc
            # 重新取 hidden_state
            enc2 = tokenizer(
                text, truncation=True, padding=True,
                max_length=args.max_length, return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                out2 = model(**enc2, output_hidden_states=True)
            cls_vec = out2.hidden_states[-1][:, 0, :].cpu().numpy()[0]
            del out2, enc2

            # 单窗口：6维和768维都直接用
            p_neg, p_neu, p_pos = probs
            extreme6 = np.array([p_pos, p_neg, p_pos, p_neg, 0.0, 1.0 if p_neu > 0.5 else 0.0])
            all_extreme6.append(extreme6)
            all_weighted768.append(cls_vec)
        else:
            # 多窗口：逐批推理
            window_probs = []
            window_cls = []

            for i in range(0, n_windows, args.batch_size):
                batch_ids = input_ids_list[i: i + args.batch_size]
                batch_mask = attn_mask_list[i: i + args.batch_size]

                max_len = max(len(ids) for ids in batch_ids)
                padded_ids = [ids + [0] * (max_len - len(ids)) for ids in batch_ids]
                padded_mask = [m + [0] * (max_len - len(m)) for m in batch_mask]

                inputs = {
                    "input_ids": torch.tensor(padded_ids).to(device),
                    "attention_mask": torch.tensor(padded_mask).to(device),
                }

                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)

                probs_batch = F.softmax(outputs.logits, dim=1).cpu().numpy()  # (B, 3)
                cls_batch = outputs.hidden_states[-1][:, 0, :].cpu().numpy()  # (B, 768)

                window_probs.append(probs_batch)
                window_cls.append(cls_batch)

            window_probs = np.vstack(window_probs)  # (W, 3)  [neg, neu, pos]
            window_cls = np.vstack(window_cls)       # (W, 768)

            p_neg = window_probs[:, 0]
            p_neu = window_probs[:, 1]
            p_pos = window_probs[:, 2]

            # 方案一：6维极端值特征
            max_pos = float(p_pos.max())
            max_neg = float(p_neg.max())
            mean_pos = float(p_pos.mean())
            mean_neg = float(p_neg.mean())
            sentiment_std = float(p_pos.std())  # Positive概率的跨窗口标准差
            neutral_ratio = float((p_neu > 0.5).mean())  # Neutral窗口占比
            extreme6 = np.array([max_pos, max_neg, mean_pos, mean_neg, sentiment_std, neutral_ratio])

            # 方案二：注意力加权 768 维
            weights = 1.0 - p_neu  # 离 neutral 越远权重越高
            if weights.sum() > 0:
                weights = weights / weights.sum()
            weighted768 = (window_cls * weights[:, np.newaxis]).sum(axis=0)

            all_extreme6.append(extreme6)
            all_weighted768.append(weighted768)

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - start_time
            speed = (idx + 1) / elapsed
            remaining = (len(texts) - idx - 1) / speed
            print(f"  -> {idx+1}/{len(texts)}, {speed:.1f}条/秒, 预计剩余{remaining/60:.1f}分钟")

    elapsed = time.time() - start_time
    print(f"完成! {len(all_extreme6)} 条, 用时 {elapsed/60:.1f} 分钟")

    return (
        np.array(all_extreme6, dtype=np.float32),
        np.array(all_weighted768, dtype=np.float32),
        np.array(labels_raw, dtype=np.int8),
        np.array(prices_raw, dtype=np.float32),
        np.array(dates_raw),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mini", "full"], default="mini",
                        help="mini=500条测试, full=全量")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "mini":
        args.max_samples = 500
        out_name = "extreme_features_mini"
        print("模式: mini (500条快速测试)")
    else:
        out_name = "extreme_features_full"
        print("模式: full (全量)")

    print("=" * 60)
    print("滑窗 FinBERT 极端值/加权特征提取")
    print("=" * 60)

    if args.max_samples:
        print(f"限制: {args.max_samples} 条")

    extreme6, weighted768, labels, prices, dates = extract_extreme_features(args)

    # 保存
    output_path = os.path.join(DATA_DIR, f"{out_name}.npz")
    print(f"\n保存: {output_path}")
    np.savez_compressed(
        output_path,
        extreme6=extreme6,
        weighted768=weighted768,
        labels=labels,
        prices=prices,
        dates=dates,
    )
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"  大小: {file_size:.1f} MB")
    print(f"  extreme6:   {extreme6.shape}")
    print(f"  weighted768: {weighted768.shape}")
    print(f"  labels:      {labels.shape}")

    print("\n统计: 方案一 6维特征分布")
    names = ["max_pos", "max_neg", "mean_pos", "mean_neg", "sentiment_std", "neutral_ratio"]
    for i, name in enumerate(names):
        vals = extreme6[:, i]
        print(f"  {name:<15} mean={vals.mean():.4f}  std={vals.std():.4f}  "
              f"min={vals.min():.4f}  max={vals.max():.4f}")

    # 打印极端样本
    print("\nTop-5 最高 max_pos 的样本:")
    top_idx = np.argsort(extreme6[:, 0])[-5:][::-1]
    for i, idx in enumerate(top_idx):
        print(f"  #{i+1} idx={idx}  max_pos={extreme6[idx, 0]:.4f}  "
              f"max_neg={extreme6[idx, 1]:.4f}  label={'up' if labels[idx] else 'down'}")

    print(f"\n下一步: python 10b_compare_results.py --mode {args.mode}\n")


if __name__ == "__main__":
    main()
