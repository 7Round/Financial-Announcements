"""
TF-IDF + 历史时序特征构建脚本

从训练文本中构建：
1. 全局 TF-IDF 特征（全文，不截断）
2. 历史价格时序统计特征

输出 features_built.npz，供 07_hybrid_train.py 使用。

用法:
  python 06b_build_features.py --input mini_train_text.csv
"""

import os
import csv
import argparse
import re
import warnings
import numpy as np
from collections import defaultdict

warnings.filterwarnings("ignore")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(DATA_DIR, "mini_train_text.csv")
DEFAULT_OUTPUT = os.path.join(DATA_DIR, "features_built.npz")


def build_time_series_features(rows: list[dict]) -> np.ndarray:
    """
    构建历史价格时序特征。

    对每只股票，按日期排序后依次计算：
      price_ma_3  — 过去3日均价
      price_ma_5  — 过去5日均价
      price_std_5 — 过去5日价格标准差（波动率）
      price_mom_1 — 相对于昨日的涨跌幅
      price_mom_3 — 过去3日累计涨跌幅
      price_mom_5 — 过去5日累计涨跌幅
      price_min_5 — 过去5日最低价
      price_max_5 — 过去5日最高价

    不足 N 天时以 NaN 填充（CatBoost 原生支持 NaN）。
    """
    n = len(rows)
    feature_names = [
        "ma_3", "ma_5", "std_5",
        "mom_1", "mom_3", "mom_5",
        "min_5", "max_5",
    ]
    features = np.full((n, len(feature_names)), np.nan, dtype=np.float32)

    # 按 stkcd 分组
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["stkcd"]].append(i)

    for stkcd, indices in groups.items():
        if len(indices) < 2:
            continue

        # 按日期排序
        sorted_idx = sorted(indices, key=lambda i: rows[i]["date"])
        prices = np.array([float(rows[i]["clpr"]) for i in sorted_idx])

        for pos, idx in enumerate(sorted_idx):
            if pos == 0:
                continue  # 第1条无历史

            p_t = prices[pos]      # 当天价格
            p_p1 = prices[pos - 1]  # 昨天价格

            f = features[idx]

            # mom_1: 相对于昨日涨跌幅
            f[3] = (p_t - p_p1) / p_p1

            if pos >= 2:
                # ma_3
                f[0] = prices[pos - 2: pos + 1].mean()
                # mom_3
                f[4] = (p_t - prices[pos - 2]) / prices[pos - 2]

            if pos >= 4:
                window = prices[pos - 4: pos + 1]
                f[1] = window.mean()
                f[2] = window.std()
                f[5] = (p_t - prices[pos - 4]) / prices[pos - 4]
                f[6] = window.min()
                f[7] = window.max()

    print(f"  -> 时序特征维度: {features.shape[1]}")
    for i, name in enumerate(feature_names):
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
    print(f"  -> TF-IDF 维度: {tfidf_emb.shape}")

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
    print(f"  -> 大小: {os.path.getsize(args.output) / 1024:.1f} KB")
    print(f"\n下一步: python 07_hybrid_train.py\n")


if __name__ == "__main__":
    main()
