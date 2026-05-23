"""
数据准备脚本：将训练集/测试集CSV与公告PDF对齐

输出:
  - train_aligned.csv     — 训练集 + 对应公告信息（文件路径+日期）
  - test_aligned.csv      — 测试集 + 对应公告信息
  - unmatched_samples.txt — 未匹配到公告的样本（供后续分析）

对齐策略:
  默认取该股票在 date 当天或之前最近一份公告。
  可通过 mode 参数切换为取当天前 N 天的全部公告合并。
"""

import os
import re
import csv
import sys
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional


# ========== 配置 ==========

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(DATA_DIR, "dataset")
TRAIN_CSV = os.path.join(DATA_DIR, "训练集.csv")
TEST_CSV = os.path.join(DATA_DIR, "测试集.csv")
SUBMIT_CSV = os.path.join(DATA_DIR, "提交示例.csv")

OUT_TRAIN = os.path.join(DATA_DIR, "train_aligned.csv")
OUT_TEST = os.path.join(DATA_DIR, "test_aligned.csv")
OUT_UNMATCHED = os.path.join(DATA_DIR, "unmatched_samples.txt")


# ========== 1. 解析文件名中的股票代码和日期 ==========

PDF_PATTERN = re.compile(
    r"^(\d{6}[^\W_]+?)"   # 股票代码（6位数字+中文名，非贪婪）
    r"_.*?"                # 公告标题（任意内容，非贪婪）
    r"_(\d{4}-\d{2}-\d{2})"  # 日期 YYYY-MM-DD
    r"\.pdf$"
)


def parse_pdf_filename(filename: str) -> Optional[tuple[str, str]]:
    """从PDF文件名提取 (股票代码, 日期)，失败返回 None"""
    m = PDF_PATTERN.search(filename)
    if m:
        # 提取纯数字股票代码（去掉后面的中文名称）
        raw_code = m.group(1)
        code_match = re.match(r"(\d{6})", raw_code)
        if code_match:
            return code_match.group(1), m.group(2)
    return None


def scan_pdfs(pdf_dir: str) -> dict[str, list[tuple[str, str, str]]]:
    """
    扫描所有PDF，返回:
      { 股票代码: [(文件名, 日期, 完整路径), ...] }
    并按日期升序排列
    """
    index: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    total = 0
    matched = 0

    print(f"正在扫描 PDF 目录: {pdf_dir}")
    for fname in os.listdir(pdf_dir):
        if not fname.lower().endswith(".pdf"):
            continue
        total += 1
        result = parse_pdf_filename(fname)
        if result:
            code, date_str = result
            index[code].append((fname, date_str, os.path.join(pdf_dir, fname)))
            matched += 1

    # 按日期排序
    for code in index:
        index[code].sort(key=lambda x: x[1])

    print(f"  -> 共扫描 {total} 个PDF，成功解析 {matched} 个（{len(index)} 只股票）")
    return index


# ========== 2. 公告对齐策略 ==========

def _parse_date(d: str) -> datetime:
    """兼容 YYYYMMDD 和 YYYY-MM-DD 两种格式"""
    d = d.replace("-", "")
    return datetime.strptime(d, "%Y%m%d")


def _fmt_date_for_display(d: str) -> str:
    """统一显示为 YYYY-MM-DD"""
    return _parse_date(d).strftime("%Y-%m-%d")


def find_nearest_announcement(
    pdf_index: dict[str, list[tuple[str, str, str]]],
    stkcd: str,
    target_date: str,
    mode: str = "nearest",
    days_before: int = 7,
) -> Optional[list[dict]]:
    """
    查找股票 stkcd 在 target_date 前后的公告。

    mode:
      - "nearest": 取 target_date 当天或之前最近一份公告
      - "window":  取 [target_date - days_before, target_date] 范围内所有公告

    返回:
      [{"file_path": ..., "ann_date": ..., "file_name": ...}, ...] 或 None
    """
    records = pdf_index.get(stkcd)
    if not records:
        return None

    target_dt = _parse_date(target_date)
    target_str = target_dt.strftime("%Y-%m-%d")

    if mode == "nearest":
        for fname, date_str, fpath in reversed(records):
            if date_str <= target_str:
                return [{"file_name": fname, "ann_date": date_str, "file_path": fpath}]
        return None  # 没有找到早于 target_date 的公告

    elif mode == "window":
        matched = []
        for fname, date_str, fpath in records:
            if date_str <= target_str:
                ann_dt = _parse_date(date_str)
                if (target_dt - ann_dt).days <= days_before:
                    matched.append({"file_name": fname, "ann_date": date_str, "file_path": fpath})
        return matched if matched else None

    else:
        raise ValueError(f"未知 mode: {mode}")


# ========== 3. 对齐CSV与公告 ==========

def normalize_stkcd(code: str) -> str:
    """将股票代码统一为6位数字格式"""
    digits = re.sub(r"\D", "", code)  # 去掉非数字字符
    return digits.zfill(6)


def align_dataset(
    csv_path: str,
    pdf_index: dict[str, list[tuple[str, str, str]]],
    mode: str = "nearest",
    days_before: int = 7,
    has_label: bool = True,
) -> tuple[list[dict], list[str]]:
    """
    将CSV中的每一行与公告对齐。

    返回:
      rows:      对齐成功的结果列表
      unmatched: 未匹配到的样本描述列表
    """
    rows = []
    unmatched = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        for i, row in enumerate(reader):
            stkcd = normalize_stkcd(row["stkcd"])
            date = row["date"]

            result = find_nearest_announcement(
                pdf_index, stkcd, date,
                mode=mode, days_before=days_before,
            )

            if result:
                # 合并公告信息：有多个公告时用 | 分隔
                row["ann_files"] = "|".join(r["file_path"] for r in result)
                row["ann_dates"] = "|".join(r["ann_date"] for r in result)
                row["ann_count"] = str(len(result))
                rows.append(row)
            else:
                row["ann_files"] = ""
                row["ann_dates"] = ""
                row["ann_count"] = "0"
                rows.append(row)
                unmatched.append(f"行{i+2}: stkcd={stkcd}, date={date} — 未匹配到公告")

    print(f"  -> 共 {len(rows)} 条，匹配成功 {len(rows) - len(unmatched)} 条，失败 {len(unmatched)} 条")
    return rows, unmatched


# ========== 4. 主流程 ==========

def main():
    print("=" * 60)
    print("数据准备阶段")
    print("=" * 60)

    # Step 1: 扫描PDF
    print("\n[1/3] 扫描公告PDF...")
    pdf_index = scan_pdfs(PDF_DIR)

    # Step 2: 对齐训练集
    print(f"\n[2/3] 对齐训练集 ({TRAIN_CSV})...")
    train_rows, train_unmatched = align_dataset(TRAIN_CSV, pdf_index, has_label=True)

    # Step 3: 对齐测试集
    print(f"\n[3/3] 对齐测试集 ({TEST_CSV})...")
    test_rows, test_unmatched = align_dataset(TEST_CSV, pdf_index, has_label=False)

    # 输出训练集
    out_fields = ["uuid", "stkcd", "date", "clpr", "clpr_next", "label",
                  "ann_count", "ann_dates", "ann_files"]
    with open(OUT_TRAIN, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(train_rows)
    print(f"\n=> 训练集已保存: {OUT_TRAIN} ({len(train_rows)} 行)")

    # 输出测试集
    out_fields_test = ["uuid", "stkcd", "date",
                       "ann_count", "ann_dates", "ann_files"]
    with open(OUT_TEST, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields_test)
        writer.writeheader()
        writer.writerows(test_rows)
    print(f"=> 测试集已保存: {OUT_TEST} ({len(test_rows)} 行)")

    # 输出未匹配记录
    all_unmatched = train_unmatched + test_unmatched
    if all_unmatched:
        with open(OUT_UNMATCHED, "w", encoding="utf-8") as f:
            f.write("\n".join(all_unmatched))
            f.write(f"\n\n共 {len(all_unmatched)} 条未匹配\n")
        print(f"=> 未匹配样本记录: {OUT_UNMATCHED} ({len(all_unmatched)} 条)")
    else:
        print(f"=> 所有样本均已匹配到公告")

    # 统计摘要
    print("\n" + "=" * 60)
    print("数据准备完成 — 摘要")
    print("=" * 60)
    print(f"  训练集样本: {len(train_rows)}")
    print(f"  测试集样本: {len(test_rows)}")
    print(f"  未匹配样本: {len(all_unmatched)}")
    matched_train = len(train_rows) - len(train_unmatched)
    matched_test = len(test_rows) - len(test_unmatched)
    print(f"  匹配成功率: 训练集 {matched_train}/{len(train_rows)}"
          f" ({matched_train/max(len(train_rows),1)*100:.1f}%), "
          f"测试集 {matched_test}/{len(test_rows)}"
          f" ({matched_test/max(len(test_rows),1)*100:.1f}%)")
    print(f"  PDF数据库: {len(pdf_index)} 只股票, "
          f"{sum(len(v) for v in pdf_index.values())} 份公告")
    print()


if __name__ == "__main__":
    main()
