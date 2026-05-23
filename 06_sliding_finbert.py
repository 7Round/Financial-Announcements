"""
滑窗 FinBERT 特征提取脚本（支持断点续传 + 内存优化）

对每篇公告文本用滑动窗口提取多段 CLS 向量，mean pooling 后得到全文表示。

用法:
  python 06_sliding_finbert.py --input train_text_full.csv
  python 06_sliding_finbert.py --input train_text_full.csv --stride 128

输出:
  sliding_finbert_full.npz — 包含 finbert_emb, labels, prices, dates
  checkpoint/ — 中间 checkpoint 目录（可删除，用于断点续传）
"""

import os
import csv
import argparse
import time
import json
import shutil
import warnings
import numpy as np

warnings.filterwarnings("ignore")
csv.field_size_limit(2**31 - 1)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(DATA_DIR, "train_text_full.csv")
DEFAULT_OUTPUT = os.path.join(DATA_DIR, "sliding_finbert_full.npz")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoint_sliding")
CHECKPOINT_INTERVAL = 200  # 每处理多少条保存一次 checkpoint


def sliding_windows(text: str, tokenizer, max_length: int, stride: int):
    """对长文本做滑窗编码，返回多个窗口的 input_ids 和 attention_mask。"""
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


def save_checkpoint(embeddings, labels, prices, dates, processed_idx, checkpoint_dir):
    """保存 checkpoint，覆盖写入统一文件名（只保留最新状态）"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, "checkpoint.npz")
    meta_path = os.path.join(checkpoint_dir, "checkpoint_meta.json")

    embeddings = np.vstack(embeddings) if embeddings else np.empty((0, 768), dtype=np.float32)
    np.savez_compressed(
        ckpt_path,
        finbert_emb=embeddings,
        labels=np.array(labels, dtype=np.int8),
        prices=np.array(prices, dtype=np.float32),
        dates=np.array(dates),
    )
    with open(meta_path, "w") as f:
        json.dump({"processed_idx": processed_idx}, f)
    print(f"  [Checkpoint] 已保存 {processed_idx} 条结果 → {ckpt_path}")


def load_checkpoint(checkpoint_dir, rows):
    """加载 checkpoint，返回 (embeddings列表, labels列表, prices列表, dates列表, 已处理条数)。"""
    ckpt_path = os.path.join(checkpoint_dir, "checkpoint.npz")
    meta_path = os.path.join(checkpoint_dir, "checkpoint_meta.json")

    if not os.path.exists(ckpt_path) or not os.path.exists(meta_path):
        return [], [], [], [], 0

    with open(meta_path) as f:
        meta = json.load(f)
    processed_idx = meta.get("processed_idx", 0)

    if processed_idx >= len(rows):
        print(f"  [续传] 所有 {processed_idx} 条已处理完毕，无需继续")
        return [], [], [], [], processed_idx

    # 只加载 rows[:processed_idx] 对应部分
    d = np.load(ckpt_path, allow_pickle=True)
    n_loaded = len(d["labels"])
    if n_loaded != processed_idx:
        print(f"  [警告] checkpoint 样本数 ({n_loaded}) 与记录数 ({processed_idx}) 不匹配，将重新从 0 开始")
        return [], [], [], [], 0

    print(f"  [续传] 发现已有 {processed_idx} 条处理结果，将从第 {processed_idx} 条继续")
    return (
        list(d["finbert_emb"][:n_loaded]),
        list(d["labels"][:n_loaded]),
        list(d["prices"][:n_loaded]),
        list(d["dates"][:n_loaded]),
        processed_idx,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name", default="yiyanghkust/finbert-tone-chinese")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL,
                        help="每 N 条保存一次 checkpoint")
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    parser.add_argument("--no-resume", action="store_true",
                        help="忽略已有 checkpoint，从头开始")
    args = parser.parse_args()

    print("=" * 60)
    print("滑窗 FinBERT 特征提取（支持断点续传）")
    print("=" * 60)

    # 设置 transformers 缓存到 E 盘，避免占 C 盘
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(DATA_DIR, "hf_cache")
    os.environ["HF_HOME"] = os.path.join(DATA_DIR, "hf_cache")
    os.environ["TORCH_HOME"] = os.path.join(DATA_DIR, "torch_cache")

    # 1. 加载数据
    print(f"\n[1/4] 加载数据: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if len(r["ann_text"].strip()) > 50]
    texts = [r["ann_text"] for r in rows]
    labels_raw = [1 if r["label"] == "up" else 0 for r in rows]
    prices_raw = [float(r["clpr"]) for r in rows]
    dates_raw = [r["date"] for r in rows]
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

    # 3. 检查 checkpoint（断点续传）
    print(f"\n[3/4] 滑窗提取 (window={args.max_length}, stride={args.stride})")
    if args.no_resume:
        all_embeddings, all_labels, all_prices, all_dates, processed_idx = [], [], [], [], 0
        print("  --no-resume: 忽略 checkpoint，从头开始")
    else:
        all_embeddings, all_labels, all_prices, all_dates, processed_idx = load_checkpoint(
            args.checkpoint_dir, rows
        )

    start_time = time.time()
    window_stats = []

    for idx, text in enumerate(texts):
        if idx < processed_idx:
            continue  # 已处理，跳过

        # 滑窗
        input_ids_list, attn_mask_list = sliding_windows(
            text, tokenizer, args.max_length, args.stride
        )
        n_windows = len(input_ids_list)
        window_stats.append(n_windows)

        if n_windows == 0:
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
            # 及时释放
            del encoded, outputs
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            window_embs = []
            for i in range(0, n_windows, args.batch_size):
                batch_ids = input_ids_list[i: i + args.batch_size]
                batch_mask = attn_mask_list[i: i + args.batch_size]

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

                # 及时释放 batch 变量
                del encoded, outputs, padded_ids, padded_mask, batch_ids, batch_mask

            # mean pooling
            window_embs = np.vstack(window_embs)
            doc_emb = window_embs.mean(axis=0)
            all_embeddings.append(doc_emb)

            # 释放窗口级变量
            del window_embs, input_ids_list, attn_mask_list
            if device.type == "cuda":
                torch.cuda.empty_cache()

        all_labels.append(labels_raw[idx])
        all_prices.append(prices_raw[idx])
        all_dates.append(dates_raw[idx])

        # 进度打印
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            avg_windows = np.mean(window_stats[-100:]) if window_stats else 0
            speed = (idx + 1 - processed_idx) / elapsed
            remaining = (len(texts) - idx - 1) / speed if speed > 0 else 0
            print(f"  -> 已处理 {idx+1}/{len(texts)} 条, "
                  f"平均每篇 {avg_windows:.1f} 窗口, "
                  f"速度 {speed:.2f} 条/秒, "
                  f"预计剩余 {remaining/60:.1f} 分钟")

        # Checkpoint 保存
        if (idx + 1) % args.checkpoint_interval == 0:
            save_checkpoint(all_embeddings, all_labels, all_prices, all_dates,
                           idx + 1, args.checkpoint_dir)

    # 最终保存
    embeddings = np.vstack(all_embeddings)
    elapsed = time.time() - start_time
    print(f"\n  -> 完成! 向量维度: {embeddings.shape}, 总用时 {elapsed/60:.1f} 分钟")

    # 窗口统计
    if window_stats:
        print(f"\n  滑窗统计:")
        print(f"    平均窗口数: {np.mean(window_stats):.1f}")
        print(f"    最少窗口数: {min(window_stats)}")
        print(f"    最多窗口数: {max(window_stats)}")
        print(f"    单窗口样本: {sum(1 for w in window_stats if w <= 1)} 条")

    # 保存最终结果
    print(f"\n[4/4] 保存: {args.output}")
    np.savez_compressed(
        args.output,
        finbert_emb=embeddings,
        labels=np.array(all_labels, dtype=np.int8),
        prices=np.array(all_prices, dtype=np.float32),
        dates=np.array(all_dates),
        model_name=args.model_name,
        max_length=args.max_length,
        stride=args.stride,
        n_samples=len(rows),
    )
    file_size = os.path.getsize(args.output) / 1024 / 1024
    print(f"  -> 大小: {file_size:.1f} MB")

    # 清理 checkpoint（已完成）
    if os.path.exists(args.checkpoint_dir):
        shutil.rmtree(args.checkpoint_dir)
        print(f"  -> 已清理 checkpoint 目录")

    print(f"\n下一步: python 06b_build_features.py --input {args.input}")
    print(f"         python 07_hybrid_train.py\n")


if __name__ == "__main__":
    main()
