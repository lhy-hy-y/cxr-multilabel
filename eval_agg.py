"""比较同一次检查多张片的不同聚合方式（在验证集上按 Study 算宏平均 AUC）。

背景：submission 是 Study 级的，但模型输出是图级的。多图检查怎么合并成一条，
直接影响排行榜指标。这里把 mean / max / 两者平均 / noisy-or 都算一遍。

用法
----
python eval_agg.py --ckpt runs\\r1_d121_320\\best.pt --manifest runs\\real\\valid\\manifest.csv --tta
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import eval_transform, safe_macro_auc
from predict import load_checkpoint, predict_batch


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--save-probs", default=None, help="把图级概率存成 .npy 复用")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt, device = load_checkpoint(args.ckpt, device)
    class_ids = ckpt.get("class_ids") or list(range(1, 11))
    n = len(class_ids)
    transform = eval_transform(int(ckpt["img_size"]))

    df = pd.read_csv(args.manifest)
    probs = predict_batch(model, df["image_path"].astype(str).tolist(), transform, device,
                          batch_size=8, tta=args.tta)
    if args.save_probs:
        np.save(args.save_probs, probs)
        df[["Study_id"]].to_csv(str(args.save_probs) + ".studies.csv", index=False)
    print(f"图级推理完成: {probs.shape}（tta={args.tta}）")

    y_img = df[[f"y{c}" for c in range(n)]].to_numpy()
    auc_img, _ = safe_macro_auc(y_img, probs)

    lab = df.drop_duplicates("Study_id").set_index("Study_id")[[f"y{c}" for c in range(n)]]
    print(f"\n图级宏平均 AUC = {auc_img:.4f}")
    print("-" * 46)
    best = ("", -1.0)
    for how in ("mean", "max", "meanmax", "noisyor"):
        sids, agg = aggregate(df["Study_id"].astype(str).tolist(), probs, how)
        y = lab.loc[sids].to_numpy()
        auc, valid = safe_macro_auc(y, agg)
        print(f"{how:9s} Study 级宏平均 AUC = {auc:.4f}  (有效类别 {len(valid)}/{n})")
        if auc > best[1]:
            best = (how, auc)
    print("-" * 46)
    print(f"最佳聚合: {best[0]}  {best[1]:.4f}")


if __name__ == "__main__":
    main()
