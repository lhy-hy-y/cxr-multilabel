"""在验证集上按类别搜最优 F1 阈值，产出 gates.json（可选，仅影响副指标 F1 与 Gate 列）。

主指标 Macro-AUC 与阈值无关，所以这一步只在两种情况下有意义：
1. 你需要 Gate 列给出更合理的二值判定；
2. 主指标并列时靠 Macro-F1 二次排序。

用法
----
python tune_gate.py --ckpt runs/train/best.pt --manifest runs/data_check/val/manifest.csv \
                    --out runs/gates.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from common import NUM_CLASSES, eval_transform
from predict import load_checkpoint, predict_batch


def best_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    grid = np.unique(np.round(np.linspace(0.05, 0.95, 91), 4))
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, float(best_f1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/train/best.pt")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--out", default="runs/gates.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt, device = load_checkpoint(args.ckpt, device)
    transform = eval_transform(int(ckpt["img_size"]))
    class_ids = ckpt.get("class_ids") or list(range(int(ckpt.get("num_classes", NUM_CLASSES))))
    n = len(class_ids)
    print(f"类别 id={class_ids}")

    df = pd.read_csv(args.manifest)
    probs = predict_batch(model, df["image_path"].astype(str).tolist(), transform, device,
                          batch_size=8, tta=args.tta)
    y = df[[f"y{c}" for c in range(n)]].to_numpy()

    gates, report = {}, []
    for c in range(n):
        cid = class_ids[c]
        col = y[:, c]
        if col.min() == col.max():
            gates[str(cid)] = 0.5
            report.append({"class_id": cid, "threshold": 0.5, "f1_val": None, "note": "验证集单类别"})
            continue
        t, f1 = best_threshold(col, probs[:, c])
        gates[str(cid)] = t
        report.append({"class_id": cid, "threshold": t, "f1_val": round(f1, 4)})

    f1_default = f1_score(y, (probs >= 0.5).astype(int), average="macro", zero_division=1)
    f1_tuned = f1_score(y, (probs >= np.array([gates[str(cid)] for cid in class_ids])).astype(int),
                        average="macro", zero_division=1)
    print(pd.DataFrame(report).to_string(index=False))
    print(f"\nMacro-F1 @0.5 = {f1_default:.4f} -> 调阈值后 = {f1_tuned:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(gates, indent=2), encoding="utf-8")
    print(f"已写 {args.out}；生成提交时加 --gates-json {args.out} 即可写入 Gate 列")


if __name__ == "__main__":
    main()
