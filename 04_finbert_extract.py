"""
FinBERT 文本向量提取脚本

用 FinBERT 将公告文本转为 768 维 CLS 向量，保存为 numpy 矩阵。
采用策略A（冻结权重，仅做特征提取），与 TF-IDF baseline 公平对比。

用法:
  python 04_finbert_extract.py                          # 提取 mini_train_text.csv 的向量
  python 04_finbert_extract.py --input mini_train_text.csv --output finbert_vectors.npz

输出:
  - finbert_vectors.npz — 包含 finbert_emb、labels、prices、dates 的压缩文件
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
DEFAULT_OUTPUT = os.path.join(DATA_DIR, "finbert_vectors.npz")


def load_data(csv_path: str) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name", default="luhua/chinese_pretrain_mrc_macbert_large",
                        help="FinBERT 模型名称，默认为中文金融预训练模型")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    print("=" * 60)
    print("FinBERT 文本向量提取")
    print("=" * 60)

    # 1. 加载数据
    print(f"\n[1/4] 加载数据: {args.input}")
    rows = load_data(args.input)
    # 过滤空文本
    rows = [r for r in rows if len(r["ann_text"].strip()) > 50]
    texts = [r["ann_text"] for r in rows]
    labels = np.array([1 if r["label"] == "up" else 0 for r in rows])
    prices = np.array([float(r["clpr"]) for r in rows], dtype=np.float32)
    dates = np.array([r["date"] for r in rows])
    print(f"  -> {len(rows)} 条 (过滤空文本后)")

    # 2. 加载 FinBERT
    print(f"\n[2/4] 加载 FinBERT 模型: {args.model_name}")
    print("  -> 首次运行会下载模型 (~400MB)，请耐心等待...")
    from transformers import AutoTokenizer, AutoModel
    import torch

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"  -> 模型加载完成 (device: {device})")

    # 3. 提取向量
    print(f"\n[3/4] 提取 CLS 向量 (batch_size={args.batch_size}, max_length={args.max_length})")
    all_embeddings = []
    start_time = time.time()

    for i in range(0, len(texts), args.batch_size):
        batch_texts = texts[i: i + args.batch_size]

        # 截断到 max_length tokens
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)

        # 取 CLS token (第一个 token) 的向量
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)

        if (i // args.batch_size + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  -> 已处理 {min(i+args.batch_size, len(texts))}/{len(texts)} 条 "
                  f"(用时 {elapsed:.1f}s)")

    embeddings = np.vstack(all_embeddings)
    elapsed = time.time() - start_time
    print(f"  -> 完成! 向量维度: {embeddings.shape}, 总用时 {elapsed:.1f}s")

    # 4. 保存
    print(f"\n[4/4] 保存向量: {args.output}")
    np.savez_compressed(
        args.output,
        finbert_emb=embeddings,
        labels=labels,
        prices=prices,
        dates=dates,
        # 保留元信息
        model_name=args.model_name,
        n_samples=len(rows),
    )
    print(f"  -> 保存完成! 文件大小: {os.path.getsize(args.output) / 1024 / 1024:.1f} MB")
    print(f"\n可用下一步训练: python 05_train_finbert.py --vectors {args.output}\n")


if __name__ == "__main__":
    main()
