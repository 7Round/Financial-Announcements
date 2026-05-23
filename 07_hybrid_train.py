"""
混合模型训练脚本：滑窗 FinBERT + TF-IDF + 时序特征 + CatBoost

对比 5 种方案，含 class weight 和最佳阈值搜索。

用法:
  python 07_hybrid_train.py
  python 07_hybrid_train.py --finbert-vec sliding_finbert.npz --features features_built.npz
"""

import os
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FINBERT_VEC = os.path.join(DATA_DIR, "sliding_finbert.npz")
FEATURES = os.path.join(DATA_DIR, "features_built.npz")


def load_npz(path: str, label: str):
    """加载 npz 并返回数据字典"""
    data = np.load(path, allow_pickle=True)
    print(f"  -> 加载 {label}: {dict((k, v.shape if hasattr(v, 'shape') else v) for k, v in data.items())}")
    return data


def find_best_threshold(y_true, y_proba):
    """在验证集上搜索最佳阈值（最大化 F1-score）"""
    from sklearn.metrics import f1_score
    best_th = 0.5
    best_f1 = 0
    for th in np.arange(0.05, 0.95, 0.05):
        y_pred = (y_proba >= th).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    return best_th, best_f1


def train_and_eval(X_train, X_test, y_train, y_test, name: str):
    """训练 CatBoost（含 class weight + 最佳阈值）并返回评估结果"""
    from catboost import CatBoostClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score

    # 计算 class weight：给少数类更高权重
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale = neg / pos if pos > 0 else 1.0

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.1,
        loss_function="Logloss",
        eval_metric="AUC",
        od_type="Iter",
        od_wait=50,
        verbose=False,
        random_seed=42,
        allow_writing_files=False,
        auto_class_weights="Balanced",  # 自动平衡类别权重
    )

    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)

    y_proba = model.predict_proba(X_test)[:, 1]

    # 默认阈值 0.5
    y_pred_default = (y_proba >= 0.5).astype(int)

    # 搜索最佳阈值
    best_th, best_f1 = find_best_threshold(y_test, y_proba)
    y_pred_best = (y_proba >= best_th).astype(int)

    # 默认阈值评估
    acc0 = accuracy_score(y_test, y_pred_default)
    auc = roc_auc_score(y_test, y_proba)
    cm0 = confusion_matrix(y_test, y_pred_default)
    d0 = cm0[0][0] / (cm0[0][0] + cm0[0][1]) if (cm0[0][0] + cm0[0][1]) > 0 else 0
    u0 = cm0[1][1] / (cm0[1][0] + cm0[1][1]) if (cm0[1][0] + cm0[1][1]) > 0 else 0

    # 最佳阈值评估
    acc1 = accuracy_score(y_test, y_pred_best)
    cm1 = confusion_matrix(y_test, y_pred_best)
    d1 = cm1[0][0] / (cm1[0][0] + cm1[0][1]) if (cm1[0][0] + cm1[0][1]) > 0 else 0
    u1 = cm1[1][1] / (cm1[1][0] + cm1[1][1]) if (cm1[1][0] + cm1[1][1]) > 0 else 0

    majority = max(y_test.mean(), 1 - y_test.mean())

    print(f"\n  {name}")
    print(f"  {'=' * 35}")
    print(f"    AUC:           {auc:.4f}")
    print(f"    class weight:  Balanced (neg/pos = {scale:.1f})")
    print(f"")
    print(f"    默认阈值 (0.5):")
    print(f"      准确率: {acc0*100:.1f}%  混淆: TN={cm0[0][0]} FP={cm0[0][1]} FN={cm0[1][0]} TP={cm0[1][1]}")
    print(f"      down召回率: {d0*100:.1f}%  up召回率: {u0*100:.1f}%")
    print(f"      优于基准:   {'是' if acc0 > majority else '否'}")
    print(f"")
    print(f"    最佳阈值 ({best_th:.2f}):")
    print(f"      准确率: {acc1*100:.1f}%  混淆: TN={cm1[0][0]} FP={cm1[0][1]} FN={cm1[1][0]} TP={cm1[1][1]}")
    print(f"      down召回率: {d1*100:.1f}%  up召回率: {u1*100:.1f}%")
    print(f"      F1-score: {best_f1:.4f}")

    return model, best_th


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finbert-vec", default=FINBERT_VEC)
    parser.add_argument("--features", default=FEATURES)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    print("=" * 60)
    print("混合模型训练: FinBERT + TF-IDF + 时序 + CatBoost")
    print("=" * 60)

    # 1. 加载数据
    print(f"\n[1/2] 加载特征...")
    fb = load_npz(args.finbert_vec, "滑窗FinBERT")
    ft = load_npz(args.features, "TF-IDF+时序")

    labels = fb["labels"]
    dates = fb["dates"]
    prices = fb["prices"]

    n = len(labels)
    test_size = max(1, int(n * args.test_size))
    sorted_idx = np.argsort(dates)
    n_train = n - test_size

    y_train = labels[sorted_idx[:n_train]]
    y_test = labels[sorted_idx[n_train:]]
    p_train = prices[sorted_idx[:n_train]].reshape(-1, 1)
    p_test = prices[sorted_idx[n_train:]].reshape(-1, 1)

    finbert_emb = fb["finbert_emb"]
    tfidf_emb = ft["tfidf_emb"]
    ts_features = ft["ts_features"]

    fb_train = finbert_emb[sorted_idx[:n_train]]
    fb_test = finbert_emb[sorted_idx[n_train:]]
    tf_train = tfidf_emb[sorted_idx[:n_train]]
    tf_test = tfidf_emb[sorted_idx[n_train:]]
    ts_train = ts_features[sorted_idx[:n_train]]
    ts_test = ts_features[sorted_idx[n_train:]]

    print(f"\n  样本: {n} 条, 训练 {n_train}, 验证 {test_size}")
    print(f"  验证集 up 比例: {y_test.mean()*100:.1f}%")

    # 2. 训练对比
    print(f"\n[2/2] 训练对比")
    print(f"\n{'=' * 60}")

    # 方案1: 滑窗 FinBERT
    train_and_eval(fb_train, fb_test, y_train, y_test,
                   "1. 滑窗 FinBERT (768d) + CatBoost")

    # 方案2: 滑窗 FinBERT + TF-IDF
    train_and_eval(
        np.hstack([fb_train, tf_train]),
        np.hstack([fb_test, tf_test]),
        y_train, y_test,
        "2. 滑窗 FinBERT + TF-IDF + CatBoost",
    )

    # 方案3: 完整混合
    train_and_eval(
        np.hstack([fb_train, tf_train, ts_train]),
        np.hstack([fb_test, tf_test, ts_test]),
        y_train, y_test,
        "3. 完整混合 (FinBERT + TF-IDF + 时序) + CatBoost",
    )

    # 方案4: TF-IDF baseline
    train_and_eval(tf_train, tf_test, y_train, y_test,
                   "4. TF-IDF (2000d) + CatBoost (baseline)")

    # 方案5: 仅 clpr
    train_and_eval(p_train, p_test, y_train, y_test,
                   "5. 仅 clpr + CatBoost (最低基准)")

    print(f"\n{'=' * 60}")
    print("完成\n")


if __name__ == "__main__":
    main()
