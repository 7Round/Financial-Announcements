"""
TF-IDF + XGBoost Baseline 训练脚本

用法:
  python 03_train_baseline.py                     # 使用默认 mini_train_text.csv
  python 03_train_baseline.py --input mini_train_text.csv
  python 03_train_baseline.py --time-split         # 按时间分割（默认）
  python 03_train_baseline.py --random-split       # 随机分割（仅对比用）
"""

import os
import csv
import argparse
import re
import jieba
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier


DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(DATA_DIR, "mini_train_text.csv")


def load_data(csv_path: str) -> list[dict]:
    """加载文本数据"""
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def jieba_tokenizer(text: str) -> list[str]:
    """jieba 中文分词器"""
    # 去掉空白
    text = re.sub(r"\s+", "", text)
    return list(jieba.cut(text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--time-split", action="store_true", default=True)
    parser.add_argument("--random-split", dest="time_split", action="store_false")
    parser.add_argument("--tfidf-max-features", type=int, default=1000)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    print("=" * 60)
    print("TF-IDF + XGBoost Baseline")
    print("=" * 60)

    # 1. 加载数据
    print(f"\n[1/5] 加载数据: {args.input}")
    rows = load_data(args.input)
    print(f"  -> 共 {len(rows)} 条")

    # 过滤空文本
    rows = [r for r in rows if len(r["ann_text"].strip()) > 50]
    print(f"  -> 过滤空文本后: {len(rows)} 条")

    # 2. 准备特征和标签
    print("\n[2/5] 准备特征和标签")

    texts = [r["ann_text"] for r in rows]
    prices = np.array([float(r["clpr"]) for r in rows]).reshape(-1, 1)
    labels = np.array([1 if r["label"] == "up" else 0 for r in rows])
    dates = [r["date"] for r in rows]

    print(f"  up 比例: {labels.mean()*100:.1f}%")
    print(f"  down 比例: {(1-labels.mean())*100:.1f}%")

    # 3. 分割（按时间或随机）
    print("\n[3/5] 数据集分割")
    n = len(rows)
    n_test = max(1, int(n * args.test_size))
    n_train = n - n_test

    if args.time_split:
        # 按日期排序后分割
        sorted_indices = np.argsort(dates)
        train_idx = sorted_indices[:n_train]
        test_idx = sorted_indices[n_train:]
        print(f"  分割方式: 按时间分割 (前 {n_train} 条训练, 后 {n_test} 条验证)")
        print(f"  训练集日期范围: {dates[train_idx[0]]} ~ {dates[train_idx[-1]]}")
        print(f"  验证集日期范围: {dates[test_idx[0]]} ~ {dates[test_idx[-1]]}")
    else:
        np.random.seed(42)
        indices = np.random.permutation(n)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        print(f"  分割方式: 随机分割 (训练 {n_train}, 验证 {n_test})")

    X_text_train = [texts[i] for i in train_idx]
    X_text_test = [texts[i] for i in test_idx]
    X_price_train = prices[train_idx]
    X_price_test = prices[test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    # 4. TF-IDF 向量化
    print(f"\n[4/5] TF-IDF 向量化 (max_features={args.tfidf_max_features})")

    vectorizer = TfidfVectorizer(
        max_features=args.tfidf_max_features,
        tokenizer=jieba_tokenizer,
        ngram_range=(1, 2),
        min_df=2,
    )

    X_tfidf_train = vectorizer.fit_transform(X_text_train).toarray()
    X_tfidf_test = vectorizer.transform(X_text_test).toarray()

    print(f"  TF-IDF 维度: {X_tfidf_train.shape[1]}")

    # 拼接价格特征
    X_train = np.hstack([X_tfidf_train, X_price_train])
    X_test = np.hstack([X_tfidf_test, X_price_test])

    print(f"  最终特征维度: {X_train.shape[1]} (TF-IDF {X_tfidf_train.shape[1]} + 价格 1)")

    # 5. 训练 XGBoost
    print("\n[5/5] 训练 XGBoost...")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # 评估
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'=' * 60}")
    print(f"评估结果")
    print(f"{'=' * 60}")
    print(f"  准确率 (Accuracy):  {acc:.4f} ({acc*100:.1f}%)")
    print(f"  AUC:                {auc:.4f}")
    print(f"")
    print(f"  混淆矩阵:")
    print(f"                 预测 down    预测 up")
    print(f"  实际 down      {cm[0][0]:>5}       {cm[0][1]:>5}")
    print(f"  实际 up        {cm[1][0]:>5}       {cm[1][1]:>5}")
    print(f"")
    if cm[0][0] + cm[0][1] > 0:
        down_acc = cm[0][0] / (cm[0][0] + cm[0][1])
        print(f"  down 召回率: {down_acc:.1%}")
    if cm[1][0] + cm[1][1] > 0:
        up_acc = cm[1][1] / (cm[1][0] + cm[1][1])
        print(f"  up 召回率:   {up_acc:.1%}")

    # Top-20 最重要的特征词
    feature_names = vectorizer.get_feature_names_out().tolist() + ["clpr"]
    importance = model.feature_importances_

    # 按重要性排序，取 Top-20（排除 clpr）
    word_importance = [
        (name, imp)
        for name, imp in zip(feature_names, importance)
        if name != "clpr"
    ]
    word_importance.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  Top-20 最重要的特征词:")
    print(f"  {'排名':>4}  {'特征词':<12}  {'重要性':<10}")
    print(f"  {'-'*30}")
    for rank, (word, imp) in enumerate(word_importance[:20], 1):
        print(f"  {rank:>4}  {word:<12}  {imp:.4f}")

    # clpr 的重要性排名
    clpr_rank = sum(1 for imp in importance if imp > importance[-1]) + 1
    print(f"\n  clpr(价格) 重要性排名: 第 {clpr_rank} / {len(importance)}")

    # 随机猜测对比
    print(f"\n  --- 对比基准 ---")
    print(f"  随机猜测准确率:    50.0%")
    test_up_ratio = y_test.mean()
    print(f"  全部预测多数类:    {max(test_up_ratio, 1-test_up_ratio)*100:.1f}%")
    print(f"  模型准确率:        {acc*100:.1f}%")
    if acc > max(test_up_ratio, 1 - test_up_ratio):
        print(f"  >> 模型优于多数类基准! <<")
    else:
        print(f"  模型未超过多数类基准")

    print(f"\n=> 完成! 结果如上\n")


if __name__ == "__main__":
    main()
