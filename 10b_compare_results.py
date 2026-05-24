"""
对比训练：极端值特征 (6d) / 加权768 vs 原滑窗 mean pooling (768d) baseline

对比方案:
  1. 原 baseline:  滑窗 FinBERT mean pooling (768d) + CatBoost
  2. 方案一:        extreme6 (6d) + CatBoost
  3. 方案二:        weighted768 (768d) + CatBoost
  4. extreme6 + TF-IDF (2000d)
  5. weighted768 + TF-IDF (2000d)

用法:
  python 10b_compare_results.py --mode mini
  python 10b_compare_results.py --mode full
"""

import os
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def load_npz(path, label=""):
    d = np.load(path, allow_pickle=True)
    print(f"  [{label}] {path}")
    for k, v in d.items():
        shape_info = v.shape if hasattr(v, "shape") else v
        print(f"    {k}: {shape_info}")
    return d


def train_and_eval(X_train, X_test, y_train, y_test, name):
    from catboost import CatBoostClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

    model = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.1,
        loss_function="Logloss", eval_metric="AUC",
        od_type="Iter", od_wait=50,
        verbose=False, random_seed=42,
        allow_writing_files=False, auto_class_weights="Balanced",
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    d_recall = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
    u_recall = cm[1][1] / (cm[1][0] + cm[1][1]) if (cm[1][0] + cm[1][1]) > 0 else 0

    print(f"\n  {name}")
    print(f"  {'=' * 35}")
    print(f"    AUC:          {auc:.4f}")
    print(f"    准确率:        {acc*100:.1f}%")
    print(f"    混淆矩阵:      TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
    print(f"    down召回率:    {d_recall*100:.1f}%")
    print(f"    up召回率:      {u_recall*100:.1f}%")
    return auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mini", "full"], default="mini")
    args = parser.parse_args()

    is_mini = args.mode == "mini"

    print("=" * 60)
    print(f"对比训练: 极端值特征 {'mini(500条)' if is_mini else '全量'}")
    print("=" * 60)

    # 加载数据
    print(f"\n[1/2] 加载特征...")

    # 原 baseline FinBERT: mini 模式从全量切片
    if is_mini:
        baseline_npz = os.path.join(DATA_DIR, "sliding_finbert_full.npz")
    else:
        baseline_npz = os.path.join(DATA_DIR, "sliding_finbert_full.npz")
    d_baseline = load_npz(baseline_npz, "baseline FinBERT")

    # Extreme features
    extreme_npz = os.path.join(DATA_DIR, f"extreme_features_{args.mode}.npz")
    d_extreme = load_npz(extreme_npz, "Extreme features")

    # TF-IDF: mini 模式从全量切片
    if is_mini:
        tfidf_npz = os.path.join(DATA_DIR, "features_built_full.npz")
    else:
        tfidf_npz = os.path.join(DATA_DIR, "features_built_full.npz")
    d_tfidf = load_npz(tfidf_npz, "TF-IDF+时序")

    # 按日期分割（与 07 一致）
    labels = d_baseline["labels"]
    dates = d_baseline["dates"]
    n = len(labels)

    if is_mini:
        # mini 模式: 切片前 500 条（与 extreme 特征对齐，各脚本读 train_text_full.csv 前 500 行）
        extreme_n = len(d_extreme["labels"])
        labels = labels[:extreme_n]
        dates = dates[:extreme_n]
        n = extreme_n

    test_size = max(1, int(n * 0.2))
    sorted_idx = np.argsort(dates)
    n_train = n - test_size
    y_train = labels[sorted_idx[:n_train]]
    y_test = labels[sorted_idx[n_train:]]

    baseline768 = d_baseline["finbert_emb"]
    extreme6 = d_extreme["extreme6"]
    weighted768 = d_extreme["weighted768"]
    tfidf = d_tfidf["tfidf_emb"]

    if is_mini:
        baseline768 = baseline768[:extreme_n]
        tfidf = tfidf[:extreme_n]

    bl_train = baseline768[sorted_idx[:n_train]]
    bl_test = baseline768[sorted_idx[n_train:]]
    e6_train = extreme6[sorted_idx[:n_train]]
    e6_test = extreme6[sorted_idx[n_train:]]
    w768_train = weighted768[sorted_idx[:n_train]]
    w768_test = weighted768[sorted_idx[n_train:]]
    tf_train = tfidf[sorted_idx[:n_train]]
    tf_test = tfidf[sorted_idx[n_train:]]

    print(f"\n  样本: {n} 条, 训练 {n_train}, 验证 {test_size}")
    print(f"  验证集 up 比例: {y_test.mean()*100:.1f}%")

    # 训练对比
    print(f"\n[2/2] 训练对比")
    print(f"\n{'=' * 60}")

    results = {}

    # 方案0: 原 baseline
    auc = train_and_eval(bl_train, bl_test, y_train, y_test,
                          "0. 原滑窗 mean pooling (768d) baseline")
    results["baseline_768d"] = auc

    # 方案1: extreme6
    auc = train_and_eval(e6_train, e6_test, y_train, y_test,
                          "1. extreme6 (6d) + CatBoost")
    results["extreme6"] = auc

    # 方案2: weighted768
    auc = train_and_eval(w768_train, w768_test, y_train, y_test,
                          "2. weighted768 (768d) + CatBoost")
    results["weighted768"] = auc

    # 方案3: extreme6 + TF-IDF
    auc = train_and_eval(
        np.hstack([e6_train, tf_train]),
        np.hstack([e6_test, tf_test]),
        y_train, y_test,
        "3. extreme6 + TF-IDF + CatBoost",
    )
    results["extreme6+tfidf"] = auc

    # 方案4: weighted768 + TF-IDF
    auc = train_and_eval(
        np.hstack([w768_train, tf_train]),
        np.hstack([w768_test, tf_test]),
        y_train, y_test,
        "4. weighted768 + TF-IDF + CatBoost",
    )
    results["weighted768+tfidf"] = auc

    # 汇总
    print(f"\n{'=' * 60}")
    print("AUC 汇总")
    print(f"{'=' * 60}")
    for name, auc in results.items():
        print(f"  {name:<25}  AUC = {auc:.4f}")

    # vs baseline
    baseline_auc = results.get("baseline_768d", 0)
    print(f"\n  相对 baseline ({baseline_auc:.4f}):")
    for name, auc in results.items():
        delta = (auc - baseline_auc) * 100
        sign = "+" if delta > 0 else ""
        print(f"    {name:<25}  {sign}{delta:.2f}pp")

    print(f"\n{'=' * 60}")
    print("完成\n")


if __name__ == "__main__":
    main()
