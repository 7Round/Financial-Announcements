"""
情感词典 + TF-IDF 小规模测试脚本

用姜富伟中文金融情感词典提取情感特征，与 TF-IDF 拼接后训练 CatBoost。
对比 3 个方案：
  1. 仅 TF-IDF (baseline)
  2. TF-IDF + 情感词典特征
  3. 仅情感词典特征

用法:
  python 08_sentiment_dict_test.py
  python 08_sentiment_dict_test.py --input mini_train_text.csv
"""

import os
import csv
import re
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore")
csv.field_size_limit(2**31 - 1)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(DATA_DIR, "mini_train_text.csv")
DEFAULT_POS = os.path.join(DATA_DIR, "sentiment_positive.txt")
DEFAULT_NEG = os.path.join(DATA_DIR, "sentiment_negative.txt")

# ========== 1. 加载情感词典 ==========

def load_sentiment_dict(pos_path: str, neg_path: str):
    """加载情感词典，返回 (positive_set, negative_set)"""
    def load_words(path):
        with open(path, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())

    pos_words = load_words(pos_path)
    neg_words = load_words(neg_path)
    print(f"  正面词: {len(pos_words)}")
    print(f"  负面词: {len(neg_words)}")

    # 检查重叠
    overlap = pos_words & neg_words
    if overlap:
        print(f"  重叠词: {len(overlap)} (已从负面词中移除)")
        neg_words -= overlap

    return pos_words, neg_words


def extract_sentiment_features(text: str, pos_words: set, neg_words: set) -> np.ndarray:
    """
    从文本中提取情感特征，返回 5 维向量:
      [pos_count, neg_count, pos_ratio, neg_ratio, sent_score]
    """
    # 简单分词：按非中文字符切分
    words = re.findall(r"[\u4e00-\u9fff]+", text)
    # 再切成 2-4 字的词（情感词典主要是多字词）
    tokens = []
    for w in words:
        # 整词匹配
        tokens.append(w)
        # 2-gram
        for i in range(len(w) - 1):
            tokens.append(w[i:i+2])

    total = len(tokens)
    pos_count = sum(1 for t in tokens if t in pos_words)
    neg_count = sum(1 for t in tokens if t in neg_words)

    pos_ratio = pos_count / total if total > 0 else 0
    neg_ratio = neg_count / total if total > 0 else 0
    sent_score = (pos_count - neg_count) / total if total > 0 else 0

    return np.array([pos_count, neg_count, pos_ratio, neg_ratio, sent_score], dtype=np.float32)


# ========== 2. 加载数据 ==========

def load_data(csv_path: str):
    """加载数据，返回 (rows, texts, labels, prices, dates)"""
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"  总行数: {len(rows)}")

    # 过滤空文本
    rows = [r for r in rows if len(r["ann_text"].strip()) > 50]
    print(f"  过滤空文本后: {len(rows)}")

    texts = [r["ann_text"] for r in rows]
    labels = np.array([1 if r["label"] == "up" else 0 for r in rows])
    prices = np.array([float(r["clpr"]) for r in rows], dtype=np.float32)
    dates = np.array([r["date"] for r in rows])

    print(f"  up 比例: {labels.mean()*100:.1f}%")
    return rows, texts, labels, prices, dates


# ========== 3. 训练评估 ==========

def evaluate(X_train, X_test, y_train, y_test, name: str):
    """训练 CatBoost 并评估"""
    from catboost import CatBoostClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score

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
        auto_class_weights="Balanced",
    )

    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    down_recall = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
    up_recall = cm[1][1] / (cm[1][0] + cm[1][1]) if (cm[1][0] + cm[1][1]) > 0 else 0
    majority = max(y_test.mean(), 1 - y_test.mean())

    print(f"\n  {name}")
    print(f"  {'=' * 35}")
    print(f"  AUC:         {auc:.4f}")
    print(f"  准确率:      {acc*100:.1f}%")
    print(f"  混淆矩阵:    TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
    print(f"  down 召回率: {down_recall*100:.1f}%")
    print(f"  up 召回率:   {up_recall*100:.1f}%")
    print(f"  优于多数类:  {'是' if acc > majority else '否'} (多数类={majority*100:.1f}%)")

    # 最佳阈值搜索
    best_th = 0.5
    best_f1 = 0
    for th in np.arange(0.05, 0.95, 0.05):
        yp = (y_proba >= th).astype(int)
        f1 = f1_score(y_test, yp)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    y_pred_best = (y_proba >= best_th).astype(int)
    cm_best = confusion_matrix(y_test, y_pred_best)
    print(f"  最佳阈值 ({best_th:.2f}): TN={cm_best[0][0]} FP={cm_best[0][1]} "
          f"FN={cm_best[1][0]} TP={cm_best[1][1]} F1={best_f1:.4f}")

    return auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--pos-dict", default=DEFAULT_POS)
    parser.add_argument("--neg-dict", default=DEFAULT_NEG)
    parser.add_argument("--tfidf-max-features", type=int, default=1000)
    args = parser.parse_args()

    print("=" * 60)
    print("情感词典 + TF-IDF 小规模测试")
    print("=" * 60)

    # 加载情感词典
    print(f"\n[1/4] 加载情感词典...")
    pos_words, neg_words = load_sentiment_dict(args.pos_dict, args.neg_dict)

    # 加载数据
    print(f"\n[2/4] 加载数据: {args.input}")
    rows, texts, labels, prices, dates = load_data(args.input)

    # 提取情感特征和 TF-IDF
    print(f"\n[3/4] 提取特征...")

    # 3a. 情感特征
    print(f"  提取情感词典特征...")
    sent_features = np.array([extract_sentiment_features(t, pos_words, neg_words) for t in texts])
    print(f"    情感特征统计:")
    for i, name in enumerate(["pos_count", "neg_count", "pos_ratio", "neg_ratio", "sent_score"]):
        vals = sent_features[:, i]
        print(f"      {name}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
              f"min={vals.min():.4f}, max={vals.max():.4f}")

    # 3b. TF-IDF
    print(f"  TF-IDF 向量化 (max_features={args.tfidf_max_features})...")
    import jieba
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        max_features=args.tfidf_max_features,
        tokenizer=lambda x: list(jieba.cut(re.sub(r"\s+", "", x))),
        ngram_range=(1, 2),
        min_df=2,
    )
    tfidf_emb = vectorizer.fit_transform(texts).toarray()
    print(f"    TF-IDF 维度: {tfidf_emb.shape}")

    # 时间分割
    print(f"\n[4/4] 训练对比...")
    n = len(texts)
    test_size = max(1, int(n * 0.2))
    sorted_idx = np.argsort(dates)
    train_idx = sorted_idx[:n - test_size]
    test_idx = sorted_idx[n - test_size:]

    y_train = labels[train_idx]
    y_test = labels[test_idx]
    print(f"  时间分割: 训练 {len(train_idx)}, 验证 {len(test_idx)}")
    print(f"  验证集日期范围: {dates[min(test_idx)]} ~ {dates[max(test_idx)]}")

    # 方案1: 仅 TF-IDF
    evaluate(tfidf_emb[train_idx], tfidf_emb[test_idx], y_train, y_test,
             "1. 仅 TF-IDF (baseline)")

    # 方案2: TF-IDF + 情感特征
    X_train = np.hstack([tfidf_emb[train_idx], sent_features[train_idx]])
    X_test = np.hstack([tfidf_emb[test_idx], sent_features[test_idx]])
    evaluate(X_train, X_test, y_train, y_test,
             "2. TF-IDF + 情感词典 (5维)")

    # 方案3: 仅情感特征
    evaluate(sent_features[train_idx], sent_features[test_idx], y_train, y_test,
             "3. 仅情感词典 (5维)")

    # 方案4: 情感特征 + clpr
    X_train4 = np.hstack([sent_features[train_idx], prices[train_idx].reshape(-1, 1)])
    X_test4 = np.hstack([sent_features[test_idx], prices[test_idx].reshape(-1, 1)])
    evaluate(X_train4, X_test4, y_train, y_test,
             "4. 情感词典 + clpr")

    print(f"\n{'=' * 60}")
    print("测试完成\n")


if __name__ == "__main__":
    main()
