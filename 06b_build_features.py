"""
TF-IDF + 历史时序特征构建脚本

从训练文本中构建：
1. 全局 TF-IDF 特征（全文，不截断）
2. 历史价格时序统计特征

输出 features_built_full.npz，供 07_hybrid_train.py 使用。

用法:
  python 06b_build_features.py --input train_text_full.csv
"""

import os
import csv
import argparse
import re
import time
import warnings
import numpy as np
from collections import defaultdict

warnings.filterwarnings("ignore")

csv.field_size_limit(2**31 - 1)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(DATA_DIR, "train_text_full.csv")
DEFAULT_OUTPUT = os.path.join(DATA_DIR, "features_built_full.npz")


def build_time_series_features(rows: list[dict]) -> np.ndarray:
    """
    构建历史价格时序特征。
    对每只股票，按日期排序后依次计算 8 个统计特征。
    """
    n = len(rows)
    features = np.full((n, 8), np.nan, dtype=np.float32)

    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["stkcd"]].append(i)

    for stkcd, indices in groups.items():
        if len(indices) < 2:
            continue

        sorted_idx = sorted(indices, key=lambda i: rows[i]["date"])
        prices = np.array([float(rows[i]["clpr"]) for i in sorted_idx])

        for pos, idx in enumerate(sorted_idx):
            if pos == 0:
                continue

            p_t = prices[pos]
            p_p1 = prices[pos - 1]
            f = features[idx]

            f[3] = (p_t - p_p1) / p_p1  # mom_1

            if pos >= 2:
                f[0] = prices[pos - 2: pos + 1].mean()   # ma_3
                f[4] = (p_t - prices[pos - 2]) / prices[pos - 2]  # mom_3

            if pos >= 4:
                window = prices[pos - 4: pos + 1]
                f[1] = window.mean()     # ma_5
                f[2] = window.std()      # std_5
                f[5] = (p_t - prices[pos - 4]) / prices[pos - 4]  # mom_5
                f[6] = window.min()      # min_5
                f[7] = window.max()      # max_5

    print(f"  -> 时序特征维度: {features.shape[1]}")
    for i, name in enumerate(["ma_3", "ma_5", "std_5", "mom_1", "mom_3", "mom_5", "min_5", "max_5"]):
        nan_count = np.isnan(features[:, i]).sum()
        print(f"     {name}: {nan_count} 个 NaN ({nan_count/n*100:.1f}%)")
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--tfidf-max-features", type=int, default=2000)
    args = parser.parse_args()

    print("=" * 60)
    print("特征构建: TF-IDF + 时序特征")
    print("=" * 60)

    start_time = time.time()

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

    # 2. TF-IDF 向量化（全文，不截断）
    print(f"\n[2/4] TF-IDF 向量化 (max_features={args.tfidf_max_features})")
    import jieba
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        max_features=args.tfidf_max_features,
        tokenizer=lambda x: list(jieba.cut(re.sub(r"\s+", "", x))),
        ngram_range=(1, 2),
        min_df=2,
    )
    tfidf_emb = vectorizer.fit_transform(texts).toarray()
    elapsed = time.time() - start_time
    print(f"  -> TF-IDF 维度: {tfidf_emb.shape}, 用时 {elapsed:.1f}s")
    print(f"  -> TF-IDF 稠密矩阵内存: {tfidf_emb.nbytes / 1024 / 1024:.1f} MB")

    # 3. 时序特征
    print(f"\n[3/4] 构建历史时序特征...")
    ts_features = build_time_series_features(rows)

    # 4. 保存
    print(f"\n[4/4] 保存: {args.output}")
    np.savez_compressed(
        args.output,
        tfidf_emb=tfidf_emb,
        ts_features=ts_features,
        labels=labels,
        prices=prices,
        dates=dates,
        n_samples=len(rows),
    )
    file_size = os.path.getsize(args.output) / 1024 / 1024
    print(f"  -> 大小: {file_size:.1f} MB")

    total_elapsed = time.time() - start_time
    print(f"\n总用时: {total_elapsed:.1f}s")
    print(f"下一步: python 07_hybrid_train.py\n")


if __name__ == "__main__":
    main()
