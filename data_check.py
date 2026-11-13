"""数据体检：扫描目录 -> 解析标注 -> 配对图片 -> 写 manifest + class_ids + 一页报告。

用法
----
python data_check.py --images data/train --labels data/train.csv --out-dir runs/data_check/train
python data_check.py --images data/test --out-dir runs/data_check/test     # 测试集无标注

输出（训练/推理都吃这一套）：
    runs/data_check/xxx/manifest.csv     image_path + Study_id + y0..y9
    runs/data_check/xxx/class_ids.json   官方类别 id 顺序（本赛事是 [1..10]）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import build_manifest, read_label_table, scan_class_ids, scan_images


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="图片根目录（会递归扫描）")
    ap.add_argument("--labels", default=None, nargs="+", help="一个或多个标注 csv；测试集没有标注时省略")
    ap.add_argument("--class-ids", default=None, help="手工指定官方类别 id，如 1,2,...,10")
    ap.add_argument("--out-dir", default="runs/data_check")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] 扫描图片: {args.images}")
    images = scan_images(args.images)
    print(f"      找到 {len(images)} 张图，目录数 {images['parent'].nunique() if len(images) else 0}")
    if images.empty:
        raise SystemExit("没有找到任何图片，检查 --images 路径")

    # ---- 测试集：没有标注，只产出 image_path + Study_id ----
    if not args.labels:
        manifest = images[["image_path", "parent"]].rename(columns={"parent": "Study_id"})
        manifest.to_csv(out / "manifest.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame({"Study_id": manifest["Study_id"]}).to_csv(
            out / "study_ids.txt", index=False, header=False, encoding="utf-8")
        print(f"\n无标注模式：{len(manifest)} 张图，Study_id 取父目录名，manifest 已写 {out/'manifest.csv'}")
        for name, cnt in manifest["Study_id"].value_counts().head(3).items():
            print(f"      示例 Study_id {name}: {cnt} 张")
        return

    print("[2/3] 解析标注")
    class_ids = ([int(x) for x in args.class_ids.split(",")] if args.class_ids
                 else scan_class_ids(args.labels))
    n = len(class_ids)
    print(f"      官方类别 id: {class_ids}（共 {n} 类；内部按下标 0..{n-1} 处理）")

    frames = []
    for lp in args.labels:
        df, _ = read_label_table(lp, class_ids=class_ids, num_classes=n)
        pos = df[[f"y{c}" for c in range(n)]].sum(axis=1)
        print(f"      {Path(lp).name}: {len(df)} 个 Study_id, "
              f"平均 {pos.mean():.2f} 个标签/检查, 最多 {int(pos.max())} 个")
        frames.append(df)
    studies = pd.concat(frames, ignore_index=True).drop_duplicates("Study_id", keep="first").reset_index(drop=True)

    print("[3/3] Study_id <-> 图片 配对")
    manifest, stats = build_manifest(studies, images, n)
    print(f"      配上的 Study_id: {stats['matched']}/{stats['studies']}")
    print(f"      多图 Study_id: {stats['multi_image_studies']}")
    print(f"      产出训练行数（图级）: {len(manifest)}")
    if stats["unmatched_ids"]:
        print(f"      !! {len(stats['unmatched_ids'])} 个 Study_id 找不到图，前 5 个：{stats['unmatched_ids'][:5]}")

    ycols = [f"y{c}" for c in range(n)]
    per_study = manifest.drop_duplicates("Study_id")
    counts = per_study[ycols].sum().astype(int)
    total = max(len(per_study), 1)
    dist = pd.DataFrame({
        "官方class_id": class_ids,
        "内部下标": list(range(n)),
        "positive_studies": counts.values,
        "ratio": np.round(counts.values / total, 4),
    })
    print("\n类别分布（按 Study_id 去重）:")
    print(dist.to_string(index=False))
    zero = [class_ids[i] for i, c in enumerate(counts.values) if c == 0]
    if zero:
        print(f"!! 类别 {zero} 在训练集里 0 个正样本，检查标注解析")

    co = per_study[ycols].T.astype(int).to_numpy() @ per_study[ycols].astype(int).to_numpy()
    pd.DataFrame(co, index=class_ids, columns=class_ids).to_csv(
        out / "label_cooccurrence.csv", encoding="utf-8-sig")
    manifest.to_csv(out / "manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"Study_id": per_study["Study_id"]}).to_csv(
        out / "study_ids.txt", index=False, header=False, encoding="utf-8")
    dist.to_csv(out / "label_distribution.csv", index=False, encoding="utf-8-sig")
    (out / "class_ids.json").write_text(json.dumps(class_ids), encoding="utf-8")

    report = {
        "images_found": int(len(images)),
        "studies_from_csv": int(len(studies)),
        "studies_matched": int(stats["matched"]),
        "rows_in_manifest": int(len(manifest)),
        "class_ids": class_ids,
        "unmatched_study_ids": stats["unmatched_ids"][:50],
        "class_positive_counts": {str(cid): int(v) for cid, v in zip(class_ids, counts.values)},
    }
    (out / "data_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n完成。manifest: {out/'manifest.csv'}\nclass_ids: {out/'class_ids.json'}\n报告: {out/'data_check.json'}")
    if stats["matched"] < stats["studies"]:
        print("注意：有 Study_id 未配上图，训练会丢掉这些样本。")


if __name__ == "__main__":
    main()
