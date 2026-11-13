"""Study-level evaluation for one or more checkpoints.

The competition submission is study-level, while train.py / predict.py --eval
report image-level macro AUC. This script evaluates both levels, reports
per-class AUC / AUPRC, and compares deterministic study aggregations.

Usage
-----
python tools/eval_study.py \
    --ckpt model/r4_convnext_320_best.pt model/r6_convnext_320_s123_best.pt \
    --manifest runs/data_check/valid/manifest.csv \
    --out-json runs/eval_study.json \
    --out-csv  runs/eval_study_per_class.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import safe_macro_auc  # noqa: E402
from predict import load_ensemble, predict_multi  # noqa: E402


def aggregate(study_ids: list[str], probs: np.ndarray, how: str) -> tuple[list[str], np.ndarray]:
    buckets: dict[str, list[int]] = {}
    for i, sid in enumerate(study_ids):
        buckets.setdefault(str(sid), []).append(i)
    sids, vecs = [], []
    for sid, idx in buckets.items():
        block = probs[idx]
        if how == "mean":
            v = block.mean(0)
        elif how == "max":
            v = block.max(0)
        elif how == "meanmax":
            v = 0.5 * block.mean(0) + 0.5 * block.max(0)
        elif how == "noisyor":
            v = 1.0 - np.prod(1.0 - block, axis=0)
        else:
            raise ValueError(how)
        sids.append(sid)
        vecs.append(v)
    return sids, np.stack(vecs)


def per_class(y_true: np.ndarray, y_prob: np.ndarray) -> list[dict]:
    rows = []
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        pos = int(col.sum())
        if pos == 0 or pos == len(col):
            auc = ap = None
        else:
            auc = float(roc_auc_score(col, y_prob[:, c]))
            ap = float(average_precision_score(col, y_prob[:, c]))
        rows.append({"class_index": c, "positives": pos, "auc": auc, "auprc": ap})
    return rows


def _clean(value):
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, nargs="+")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--weights", default=None, help="ensemble weights, e.g. 0.5,0.5")
    ap.add_argument("--out-json", default="runs/eval_study.json")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest, dtype={"Study_id": str, "Subject_id": str})
    if "image_path" not in df.columns:
        raise SystemExit("manifest needs an image_path column; build it with data_check.py")

    models, device = load_ensemble(args.ckpt)
    ckpt = models[0][1]
    class_ids = list(ckpt.get("class_ids") or range(int(ckpt.get("num_classes", 10))))
    n = len(class_ids)
    ycols = [f"y{c}" for c in range(n)]
    if not all(c in df.columns for c in ycols):
        raise SystemExit(f"manifest needs label columns {ycols}")

    weights = [float(x) for x in args.weights.split(",")] if args.weights else None
    paths = df["image_path"].astype(str).tolist()
    probs = predict_multi(models, paths, device, batch_size=args.batch_size,
                          tta=args.tta, weights=weights)

    y_img = df[ycols].to_numpy(dtype=np.int32)
    img_auc, img_valid = safe_macro_auc(y_img, probs)
    img_rows = per_class(y_img, probs)

    study_ids = df["Study_id"].astype(str).tolist()
    lab = df.drop_duplicates("Study_id").set_index("Study_id")[ycols]
    agg_report = {}
    for how in ("mean", "max", "meanmax", "noisyor"):
        sids, agg = aggregate(study_ids, probs, how)
        y_study = lab.loc[sids].to_numpy(dtype=np.int32)
        auc, valid = safe_macro_auc(y_study, agg)
        rows = per_class(y_study, agg)
        agg_report[how] = {
            "macro_auc": _clean(float(auc)),
            "valid_classes": len(valid),
            "per_class": rows,
        }

    main_rows = agg_report["mean"]["per_class"]
    table = []
    for c in range(n):
        table.append({
            "class_index": c,
            "class_id": class_ids[c],
            "positive_images": img_rows[c]["positives"],
            "positive_studies": main_rows[c]["positives"],
            "image_auc": img_rows[c]["auc"],
            "image_auprc": img_rows[c]["auprc"],
            "study_auc": main_rows[c]["auc"],
            "study_auprc": main_rows[c]["auprc"],
        })

    report = {
        "checkpoints": [str(p) for p in args.ckpt],
        "device": str(device),
        "tta": bool(args.tta),
        "n_images": int(len(df)),
        "n_studies": int(len(lab)),
        "class_ids": class_ids,
        "image_level": {"macro_auc": _clean(float(img_auc)),
                        "valid_classes": len(img_valid),
                        "per_class": img_rows},
        "study_level": agg_report,
        "primary_aggregation": "mean",
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_clean),
                        encoding="utf-8")

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(table).to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"images={len(df)} studies={len(lab)} tta={args.tta}")
    print(f"image-level macro AUC = {img_auc:.4f}")
    print(f"study-level macro AUC = {agg_report['mean']['macro_auc']:.4f} (mean aggregation)")
    print("aggregation comparison:")
    for how, item in agg_report.items():
        print(f"  {how:8s} {item['macro_auc']:.4f}")
    print("\nper-class (mean aggregation):")
    print(f"{'id':>3} {'img+':>5} {'study+':>6} {'img AUC':>8} {'study AUC':>9} {'study AUPRC':>11}")
    for row in table:
        def fmt(v):
            return "   n/a" if v is None else f"{v:8.4f}"
        print(f"{row['class_id']:>3} {row['positive_images']:>5} {row['positive_studies']:>6} "
              f"{fmt(row['image_auc'])} {fmt(row['study_auc']):>9} {fmt(row['study_auprc']):>11}")
    print(f"\nwrote {out_json}" + (f" and {args.out_csv}" if args.out_csv else ""))


if __name__ == "__main__":
    main()
