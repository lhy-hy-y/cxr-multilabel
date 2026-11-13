"""提交前自检：格式、10 类齐全、概率范围、Study_id 覆盖。

用法
----
python validate_submission.py --csv submission.csv --ids runs/data_check/test/study_ids.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import NUM_CLASSES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="submission.csv")
    ap.add_argument("--ids", default=None, help="测试集应有的 Study_id 列表（每行一个）")
    ap.add_argument("--class-ids", default=None, help="官方类别 id，如 1,2,...,10；省略则要求恰好 num-classes 个不同 id")
    ap.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    args = ap.parse_args()

    errors: list[str] = []
    warns: list[str] = []

    df = pd.read_csv(args.csv, dtype={"Study_id": str})
    print(f"文件: {args.csv}（{len(df)} 行, {len(df.columns)} 列）")
    print(f"列名: {list(df.columns)}")

    # 官方"提交示例"表格为 5 列：Subject_id, Study_id, Predict_class, Probability, Gate
    expected5 = ["Subject_id", "Study_id", "Predict_class", "Probability", "Gate"]
    expected4 = ["Study_id", "Predict_class", "Probability", "Gate"]
    cols = list(df.columns)
    if cols == expected5:
        print("列格式: 5 列，与官方提交示例一致 ✓")
    elif cols == expected4:
        print("列格式: 4 列（缺少 Subject_id，官方示例为 5 列）")
        warns.append("官方提交示例首列为 Subject_id；建议补上该列")
    else:
        errors.append(f"列名/顺序与要求不一致，应为 {expected5}（或 {expected4}），实际 {cols}")

    # Subject_id 只做一致性检查：同一 Study 应对应唯一 Subject
    if "Subject_id" in df.columns:
        n_subj = df.groupby("Study_id")["Subject_id"].nunique()
        bad = int((n_subj > 1).sum())
        if bad:
            errors.append(f"{bad} 个 Study_id 对应了多个 Subject_id，数据有误")
        else:
            print(f"Subject_id: {df['Subject_id'].nunique()} 个，与 Study_id 一一对应 ✓")

    for col in ("Study_id", "Predict_class", "Probability"):
        if col in df.columns and df[col].isna().any():
            errors.append(f"{col} 存在空值 {int(df[col].isna().sum())} 个")

    # 概率范围
    prob = pd.to_numeric(df["Probability"], errors="coerce")
    bad = (~prob.between(0.0, 1.0)).sum() if "Probability" in df.columns else len(df)
    if bad:
        errors.append(f"Probability 有 {int(bad)} 行不在 [0,1]")

    # 类别与行数
    cls = pd.to_numeric(df["Predict_class"], errors="coerce")
    if cls.notna().any():
        cls = cls.astype("Int64")
        if args.class_ids:
            allowed = {int(x) for x in args.class_ids.split(",")}
            bad_ids = sorted(set(cls.dropna().unique()) - allowed)
            if bad_ids:
                errors.append(f"Predict_class 出现官方类别之外的 id: {bad_ids}（应为 {sorted(allowed)}）")
        else:
            got_ids = sorted(set(int(x) for x in cls.dropna().unique()))
            if len(got_ids) != args.num_classes:
                errors.append(f"Predict_class 只有 {len(got_ids)} 个不同取值 {got_ids}，应为 {args.num_classes} 个")
            else:
                print(f"类别 id 取值: {got_ids}")
        if df.duplicated(subset=["Study_id", "Predict_class"]).any():
            errors.append(f"存在重复的 (Study_id, Predict_class) {int(df.duplicated(subset=['Study_id','Predict_class']).sum())} 行")

    per_study = df.groupby("Study_id")["Predict_class"].nunique()
    wrong = per_study[per_study != args.num_classes]
    if len(wrong):
        preview = wrong.head(5).to_dict()
        errors.append(f"{len(wrong)} 个 Study_id 的类别数不是 {args.num_classes}，例如 {preview}")
    else:
        print(f"每个 Study_id 均有 {args.num_classes} 个类别 ✓")

    # 覆盖
    if args.ids:
        ids_path = Path(args.ids)
        if ids_path.exists():
            expected_ids = {line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()}
            got_ids = set(df["Study_id"].astype(str))
            missing = expected_ids - got_ids
            extra = got_ids - expected_ids
            print(f"Study_id 覆盖: {len(got_ids)}/{len(expected_ids)}")
            if missing:
                errors.append(f"缺少 {len(missing)} 个 Study_id，例如 {list(missing)[:5]}")
            if extra:
                warns.append(f"多出 {len(extra)} 个不在列表里的 Study_id，例如 {list(extra)[:5]}")
        else:
            warns.append(f"--ids 文件不存在: {ids_path}")

    # Gate 列
    if "Gate" in df.columns:
        g = pd.to_numeric(df["Gate"], errors="coerce")
        filled = int(g.notna().sum())
        if filled and filled != len(df):
            warns.append(f"Gate 列只有 {filled}/{len(df)} 行有值，其余为空；建议整列统一（默认全空）")
        if filled:
            print(f"Gate 自定义阈值: {sorted(g.dropna().unique())[:5]}")

    print("\n" + "=" * 60)
    if errors:
        print(f"[FAIL] {len(errors)} 个错误：")
        for e in errors:
            print("  -", e)
    else:
        print("[PASS] 未发现问题，可以提交。")
    if warns:
        print(f"\n{len(warns)} 个提醒：")
        for w in warns:
            print("  -", w)
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
