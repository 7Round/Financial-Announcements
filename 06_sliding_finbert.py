"""
滑窗 FinBERT 特征提取脚本

对每篇公告文本用滑动窗口提取多段 CLS 向量，mean pooling 后得到全文表示。
解决单次截断 512 tokens 丢失尾部信息的问题。

用法:
  python 06_sliding_finbert.py --input mini_train_text.csv
  python 06_sliding_finbert.py --input mini_train_text.csv --stride 128  # 更密集的滑窗

输出:
  sliding_finbert.npz — 包含 finbert_emb, labels, prices, dates
"""

import os
import csv
import argparse
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(DATA_DIR, "mini_train_text.csv")
DEFAULT_OUTPUT = os.path.join(DATA_DIR, "sliding_finbert.npz")


def sliding_windows(text: str, tokenizer, max_length: int, stride: int):
    """
    对长文本做滑窗编码，返回多个窗口的 input_ids 和 attention_mask。
    使用 tokenizer 的 return_overflowing_tokens 功能。
    """
    encoded = tokenizer(
        text,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_overflowing_tokens=True,
        stride=stride,
        return_tensors=None,
    )
    return encoded["input_ids"], encoded["attention_mask"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name", default="yiyanghkust/finbert-tone-chinese")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256,
                        help="滑窗步长，默认 256（50pct overlap）")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    print("=" * 60)
    print("滑窗 FinBERT 特征提取")
    print("=" * 60)

    # 1. 加载数据
    print(f"\n[1/4] 加载数据: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if len(r["ann_text"].strip()) > 50]
    texts = [r["ann_text"] for r in rows]
    labels = np.array([1 if r["label"] == "up" else 0 for r in rows])
    prices = np.array([float(r["clpr"]) for r in rows], dtype=np.float32)
    dates = np.array([r["date"] for r in rows])
    print(f"  -> {len(rows)} 条 (过滤空文本后)")

    # 2. 加载 FinBERT
    print(f"\n[2/4] 加载 FinBERT: {args.model_name}")
    from transformers import AutoTokenizer, AutoModel
    import torch

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"  -> 加载完成 (device: {device})")

    # 3. 滑窗提取
    print(f"\n[3/4] 滑窗提取 (window={args.max_length}, stride={args.stride})")
    all_embeddings = []
    window_stats = []
    start_time = time.time()

    for idx, text in enumerate(texts):
        # 对每篇文本做滑窗
        input_ids_list, attn_mask_list = sliding_windows(
            text, tokenizer, args.max_length, args.stride
        )
        n_windows = len(input_ids_list)
        window_stats.append(n_windows)

        if n_windows == 0:
            # 退化为单窗口
            encoded = tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                outputs = model(**encoded)
            emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            all_embeddings.append(emb)
            continue

        # 分批处理所有窗口
        window_embs = []
        for i in range(0, n_windows, args.batch_size):
            batch_ids = input_ids_list[i: i + args.batch_size]
            batch_mask = attn_mask_list[i: i + args.batch_size]

            # padding 到 batch 内最大长度
            max_len_in_batch = max(len(ids) for ids in batch_ids)
            padded_ids = []
            padded_mask = []
            for ids, mask in zip(batch_ids, batch_mask):
                pad_len = max_len_in_batch - len(ids)
                padded_ids.append(ids + [0] * pad_len)
                padded_mask.append(mask + [0] * pad_len)

            encoded = {
                "input_ids": torch.tensor(padded_ids).to(device),
                "attention_mask": torch.tensor(padded_mask).to(device),
            }

            with torch.no_grad():
                outputs = model(**encoded)

            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            window_embs.append(cls_emb)

        # mean pooling: 所有窗口取平均
        window_embs = np.vstack(window_embs)
        doc_emb = window_embs.mean(axis=0)
        all_embeddings.append(doc_emb)

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            avg_windows = np.mean(window_stats[-100:])
            print(f"  -> 已处理 {idx+1}/{len(texts)} 条, "
                  f"平均每篇 {avg_windows:.1f} 个窗口, 用时 {elapsed:.1f}s")

    embeddings = np.vstack(all_embeddings)
    elapsed = time.time() - start_time
    print(f"  -> 完成! 向量维度: {embeddings.shape}, 总用时 {elapsed:.1f}s")

    # 窗口统计
    print(f"\n  滑窗统计:")
    print(f"    平均窗口数: {np.mean(window_stats):.1f}")
    print(f"    最少窗口数: {min(window_stats)}")
    print(f"    最多窗口数: {max(window_stats)}")
    print(f"    单窗口样本: {sum(1 for w in window_stats if w <= 1)} 条")

    # 4. 保存
    print(f"\n[4/4] 保存: {args.output}")
    np.savez_compressed(
        args.output,
        finbert_emb=embeddings,
        labels=labels,
        prices=prices,
        dates=dates,
        model_name=args.model_name,
        max_length=args.max_length,
        stride=args.stride,
        n_samples=len(rows),
    )
    print(f"  -> 大小: {os.path.getsize(args.output) / 1024 / 1024:.1f} MB")
    print(f"\n下一步: python 06b_build_features.py --input {args.input}")
    print(f"         python 07_hybrid_train.py\n")


if __name__ == "__main__":
    main()
