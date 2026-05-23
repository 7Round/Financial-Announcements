"""
PDF文本提取脚本 — 小规模验证

从 train_aligned.csv 中筛选有公告的样本，提取文本并清洗。
先跑小批量（默认 500 条）验证效果。

用法:
  python 02_pdf_extraction.py                     # 默认取 500 条
  python 02_pdf_extraction.py --max-samples 100   # 取 100 条
  python 02_pdf_extraction.py --all               # 取全部有公告的样本
"""

import os
import csv
import argparse
import time
import fitz  # PyMuPDF


DATA_DIR = os.path.dirname(os.path.abspath(__file__))
ALIGNED_CSV = os.path.join(DATA_DIR, "train_aligned.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "mini_train_text.csv")  # 默认 500 条
OUTPUT_CSV_ALL = os.path.join(DATA_DIR, "train_text_full.csv")  # --all 模式


def extract_pdf_text(pdf_path: str) -> str:
    """用 PyMuPDF 提取 PDF 文本，返回纯文本"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text
    except Exception as e:
        return f"[提取失败: {e}]"


def clean_text(text: str) -> str:
    """基础清洗"""
    if not text:
        return ""

    # 去除页眉页脚中的页码行（如 "第 1 页 / 共 5 页"）
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # 跳过纯数字页码行
        if stripped.isdigit():
            continue
        # 跳过 "第 X 页" 行
        if stripped.startswith("第") and "页" in stripped and len(stripped) < 20:
            continue
        cleaned.append(line)

    text = "\n".join(cleaned)

    # 去除多余空行（保留最多一个连续换行）
    import re
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 去除行首行尾空白
    text = "\n".join(line.strip() for line in text.split("\n"))

    # 去除表格残留的竖线分割符
    text = re.sub(r"\|+", " ", text)

    # 去除连续短横线（表格边框）
    text = re.sub(r"-{3,}", "", text)

    # 去除多余空白
    text = re.sub(r" +", " ", text)

    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=500,
                        help="提取前 N 条样本（默认 500）")
    parser.add_argument("--all", action="store_true",
                        help="提取全部有公告的样本")
    args = parser.parse_args()

    print("=" * 60)
    print("PDF 文本提取 — 小规模验证")
    print("=" * 60)

    # 读取对齐后的训练集
    print(f"\n[1/3] 读取训练集: {ALIGNED_CSV}")
    with open(ALIGNED_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
    print(f"  -> 共 {len(all_rows)} 条")

    # 筛选有公告匹配的样本
    matched_rows = [r for r in all_rows if r["ann_count"] != "0"]
    print(f"  -> 有公告匹配: {len(matched_rows)} 条")

    # 按日期排序（旧到新）
    matched_rows.sort(key=lambda r: r["date"])

    # 确定提取数量
    if args.all:
        target_rows = matched_rows
        tag = "全部"
        output_path = OUTPUT_CSV_ALL
    else:
        target_rows = matched_rows[:args.max_samples]
        output_path = OUTPUT_CSV
        tag = f"前 {len(target_rows)}"

    print(f"  -> 本次提取: {tag} 条\n")

    # 提取文本
    print(f"[2/3] 开始提取 PDF 文本...")
    results = []
    success = 0
    fail = 0
    start_time = time.time()

    for i, row in enumerate(target_rows):
        pdf_path = row["ann_files"]
        text = extract_pdf_text(pdf_path)
        cleaned = clean_text(text)

        results.append({
            "uuid": row["uuid"],
            "stkcd": row["stkcd"],
            "date": row["date"],
            "clpr": row["clpr"],
            "label": row["label"],
            "ann_text": cleaned,
        })

        if cleaned and len(cleaned) > 50:
            success += 1
        else:
            fail += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  -> 已处理 {i+1}/{len(target_rows)} 条 "
                  f"(成功 {success}, 失败 {fail}, 用时 {elapsed:.1f}s)")

    elapsed = time.time() - start_time
    print(f"  -> 完成! 共 {len(target_rows)} 条 "
          f"(成功 {success}, 失败 {fail}, 总用时 {elapsed:.1f}s)")

    # 保存
    print(f"\n[3/3] 保存结果: {output_path}")
    fieldnames = ["uuid", "stkcd", "date", "clpr", "label", "ann_text"]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # 统计文本长度
    lengths = [len(r["ann_text"]) for r in results]
    print(f"\n文本长度统计:")
    print(f"  总条数:    {len(results)}")
    print(f"  平均长度:  {sum(lengths)/max(len(lengths),1):.0f} 字符")
    print(f"  最短:      {min(lengths)} 字符")
    print(f"  最长:      {max(lengths)} 字符")
    print(f"  空文本数:  {sum(1 for l in lengths if l == 0)}")

    print(f"\n=> 完成! 输出文件: {output_path}")
    print(f"   可用下一步训练: python 06_sliding_finbert.py --input {output_path}\n")


if __name__ == "__main__":
    main()
