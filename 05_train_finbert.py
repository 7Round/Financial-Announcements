"""
FinBERT + XGBoost 训练脚本

读取 FinBERT 提取的 768 维 CLS 向量，拼接价格特征，训练 XGBoost。
输出结果与 TF-IDF baseline 对比。

用法:
  python 05_train_finbert.py                                              # 使用默认 finbert_vectors.npz
  python 05_train_finbert.py --vectors finbert_vectors.npz
  python 05_train_finbert.py --compare-baseline                           # 同时打印 baseline 结果对比
"""

import os
import csv
import argparse
import numpy as np
import warnings

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VECTORS = os.path.join(DATA_DIR, "finbert_vectors.npz")


def load_vectors(npz_path: str):
    """加载 FinBERT 向量文件"""
    data = np.load(npz_path, allow_pickle=True)
    return {
        "embeddings": data["finbert_emb"],
        "labels": data["labels"],
        "prices": data["prices"],
        "dates": data["dates"],
    }


def train_xgboost(X_train, X_test, y_train, y_test):
    """训练 XGBoost 并返回评估结果"""
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
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "model": model,
        "accuracy": acc,
        "auc": auc,
        "confusion_matrix": cm,
    }


def print_results(name: str, result: dict, y_test: np.ndarray):
    """打印评估结果"""
    cm = result["confusion_matrix"]
    down_acc = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
    up_acc = cm[1][1] / (cm[1][0] + cm[1][1]) if (cm[1][0] + cm[1][1]) > 0 else 0
    majority = max(y_test.mean(), 1 - y_test.mean())

    print(f"\n  {name}:")
    print(f"    准确率 (Accuracy):  {result['accuracy']*100:.1f}%")
    print(f"    AUC:                {result['auc']:.4f}")
    print(f"    混淆矩阵:           TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
    print(f"    down 召回率:        {down_acc*100:.1f}%")
    print(f"    up 召回率:          {up_acc*100:.1f}%")
    print(f"    多数类基准:         {majority*100:.1f}%")
    print(f"    优于基准:           {'是' if result['accuracy'] > majority else '否'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", default=DEFAULT_VECTORS)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--compare-baseline", action="store_true",
                        help="同时读取 mini_train_text.csv 跑 TF-IDF baseline 做对比")
    args = parser.parse_args()

    print("=" * 60)
    print("FinBERT + XGBoost 训练")
    print("=" * 60)

    # 1. 加载 FinBERT 向量
    print(f"\n[1/3] 加载 FinBERT 向量: {args.vectors}")
    data = load_vectors(args.vectors)
    embeddings = data["embeddings"]
    labels = data["labels"]
    prices = data["prices"].reshape(-1, 1)
    dates = data["dates"]

    n = len(labels)
    n_test = max(1, int(n * args.test_size))
    n_train = n - n_test
    print(f"  -> {n} 条样本, FinBERT 向量维度: {embeddings.shape[1]}")

    # 2. 按时间分割
    print(f"\n[2/3] 按时间分割 (训练 {n_train}, 验证 {n_test})")
    sorted_idx = np.argsort(dates)
    train_idx = sorted_idx[:n_train]
    test_idx = sorted_idx[n_train:]

    y_train = labels[train_idx]
    y_test = labels[test_idx]

    # === 方案A: FinBERT 向量 + XGBoost ===
    print("\n[3/3] 训练")

    # 只用 FinBERT 向量（不含价格）
    X_fb_train = embeddings[train_idx]
    X_fb_test = embeddings[test_idx]
    result_fb = train_xgboost(X_fb_train, X_fb_test, y_train, y_test)

    # FinBERT 向量 + 价格
    X_fbp_train = np.hstack([embeddings[train_idx], prices[train_idx]])
    X_fbp_test = np.hstack([embeddings[test_idx], prices[test_idx]])
    result_fbp = train_xgboost(X_fbp_train, X_fbp_test, y_train, y_test)

    # 只用价格
    X_p_train = prices[train_idx]
    X_p_test = prices[test_idx]
    result_p = train_xgboost(X_p_train, X_p_test, y_train, y_test)

    # 3. 打印结果
    print(f"\n{'=' * 60}")
    print(f"评估结果对比")
    print(f"{'=' * 60}")

    print_results("FinBERT 向量 (768d)  + XGBoost", result_fb, y_test)
    print_results("FinBERT 向量 (768d)  + clpr + XGBoost", result_fbp, y_test)
    print_results("仅 clpr 价格特征 + XGBoost", result_p, y_test)

    # 特征重要性（仅 FinBERT 方案）
    imp = result_fb["model"].feature_importances_
    top_idx = np.argsort(imp)[-10:][::-1]
    print(f"\n  FinBERT Top-10 重要特征维度:")
    print(f"  {'排名':>4}  {'维度':<8}  {'重要性':<10}")
    print(f"  {'-'*24}")
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank:>4}  {idx:<8}  {imp[idx]:.4f}")

    # === 可选: TF-IDF baseline 对比 ===
    if args.compare_baseline:
        print(f"\n{'=' * 60}")
        print(f"TF-IDF Baseline 对比")
        print(f"{'=' * 60}")
        csv_path = os.path.join(DATA_DIR, "mini_train_text.csv")
        if os.path.exists(csv_path):
            import re, jieba
            from sklearn.feature_extraction.text import TfidfVectorizer

            with open(csv_path, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            rows = [r for r in rows if len(r["ann_text"].strip()) > 50]

            texts = np.array([r["ann_text"] for r in rows])
            labels2 = np.array([1 if r["label"] == "up" else 0 for r in rows])
            prices2 = np.array([float(r["clpr"]) for r in rows]).reshape(-1, 1)
            dates2 = np.array([r["date"] for r in rows])

            si = np.argsort(dates2)
            Xt = [texts[i] for i in si[:n_train]]
            Xe = [texts[i] for i in si[n_train:]]
            pt = prices2[si[:n_train]]
            pe = prices2[si[n_train:]]
            yt2 = labels2[si[:n_train]]
            ye2 = labels2[si[n_train:]]

            vec = TfidfVectorizer(
                max_features=1000,
                tokenizer=lambda x: list(jieba.cut(re.sub(r"\s+", "", x))),
                ngram_range=(1, 2),
                min_df=2,
            )
            Xtf_train = vec.fit_transform(Xt).toarray()
            Xtf_test = vec.transform(Xe).toarray()

            result_tfidf = train_xgboost(
                np.hstack([Xtf_train, pt]),
                np.hstack([Xtf_test, pe]),
                yt2, ye2,
            )
            print_results("TF-IDF (1000d) + clpr + XGBoost", result_tfidf, ye2)
        else:
            print(f"  (mini_train_text.csv 不存在，跳过对比)")

    print(f"\n=> 完成\n")


if __name__ == "__main__":
    main()
