"""计时 + 一致性核验（官方要求的 cost.py）。

官方调用方式（规则第 7 条）：
    python cost.py submission.csv 图片目录 pictures_100 pictures_5

规则要求：
* 必须导入并调用与 predict.py 相同的推理函数 —— 这里直接 `from predict import load_ensemble, predict_one_multi`；
* 5 张预热图不计时；100 张样本连续 3 轮；单张耗时先取 3 轮均值，再对 100 张取平均；
* 与 submission.csv 逐张比对：二值判定（Gate/Pred_Label）与概率排序必须完全一致，
  不一致张数为 0 才算通过；概率绝对误差 ≤1e-3 视为浮点正常波动。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 官方调用方式是在提交包根目录执行 `python cost.py ...`，此时 common.py / predict.py
# 位于 code/ 子目录，必须显式加入 sys.path，否则会 ModuleNotFoundError。
_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE / "code"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd
import torch

from common import NUM_CLASSES, eval_transform
from predict import load_ensemble, predict_one_multi


def collect_images(directory: str | Path, limit: int | None = None) -> list[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = sorted([str(p) for p in Path(directory).rglob("*") if p.is_file() and p.suffix.lower() in exts])
    if not paths:
        raise SystemExit(f"{directory} 下没有图片")
    return paths[:limit] if limit else paths


def load_reference(csv_path: str | Path, class_ids: list[int]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """返回 (每个 Study_id 的 N 维概率, 每个内部下标的阈值向量)。

    submission.csv 里写的是**官方类别 id**（本赛事 1..10），这里映射回模型内部下标；
    Gate 列按行填，所以允许每个类别用不同阈值，没填的按默认 0.5。
    """
    df = pd.read_csv(csv_path)
    for col in ("Study_id", "Predict_class", "Probability"):
        if col not in df.columns:
            raise SystemExit(f"{csv_path} 缺少列 {col}")

    index_of = {int(cid): i for i, cid in enumerate(class_ids)}
    n = len(class_ids)

    gate_vec = np.full(n, 0.5, dtype=np.float64)
    if "Gate" in df.columns:
        for _, row in df.iterrows():
            g = pd.to_numeric(pd.Series([row["Gate"]]), errors="coerce").iloc[0]
            cid = int(row["Predict_class"])
            if pd.notna(g) and cid in index_of:
                gate_vec[index_of[cid]] = float(g)

    ref = {}
    for sid, sub in df.groupby("Study_id", sort=False):
        vec = np.full(n, np.nan, dtype=np.float64)
        for _, row in sub.iterrows():
            cid = int(row["Predict_class"])
            if cid not in index_of:
                raise SystemExit(f"submission 里出现未知类别 id {cid}，模型类别为 {class_ids}")
            vec[index_of[cid]] = float(row["Probability"])
        if np.isnan(vec).any():
            raise SystemExit(f"Study_id {sid} 不满 {n} 个类别，先跑 validate_submission.py")
        ref[str(sid)] = vec
    return ref, gate_vec


def peak_from_path(path: str, ref: dict[str, np.ndarray]) -> str | None:
    """图片 -> Study_id。

    本赛事的文件名是 pic_XXXXXX，Study_id 在**父目录名**里（study_YYYYY），
    所以先试父目录名，再试文件名，最后才按前缀模糊匹配。
    """
    p = Path(path)
    for cand in (p.parent.name, p.stem):
        if cand in ref:
            return cand
    for cand in (p.stem, p.parent.name):
        for sid in ref:
            if cand.startswith(sid + "_") or cand.startswith(sid + "-") or cand == sid:
                return sid
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    # 兼容官方位置参数：submission.csv 图片目录 [pictures_100] [pictures_5]
    ap.add_argument("csv", nargs="?", default="submission.csv")
    ap.add_argument("images", nargs="?", default=None, help="图片根目录（100 张与 5 张都在里面时可省略后两个参数）")
    ap.add_argument("pictures_100", nargs="?", default=None)
    ap.add_argument("pictures_5", nargs="?", default=None)
    ap.add_argument("--ckpt", nargs="+", default=None,
                    help="一个或多个 ckpt；多个则按概率平均集成。缺省自动使用包内 model/*.pt")
    ap.add_argument("--manifest", default=None, help="测试集 manifest，用于按 Study 聚合比对")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--weights", default=None, help="集成权重，须与生成 submission 时一致")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", default="runs/cost_report.csv")
    args = ap.parse_args()

    # 权重缺省值：优先用提交包内的 model/*.pt（即生成 submission 时实际加载的权重）
    if not args.ckpt:
        candidates = sorted(_HERE.glob("model/*.pt")) or sorted((_HERE / "code").glob("model/*.pt"))
        if candidates:
            args.ckpt = [str(p) for p in candidates]
            print(f"未指定 --ckpt，自动使用: {[p.name for p in candidates]}")
        else:
            args.ckpt = ["runs/train/best.pt"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, device = load_ensemble(args.ckpt, device)
    ckpt0 = models[0][1]
    class_ids = ckpt0.get("class_ids") or list(range(int(ckpt0.get("num_classes", NUM_CLASSES))))
    desc = " + ".join(f"{c['backbone']}@{c['img_size']}" for _, c in models)
    print(f"models=[{desc}] ({len(models)} 个) device={device} tta={args.tta}")
    print(f"类别 id={class_ids}")

    W = [float(x) for x in args.weights.split(",")] if args.weights else None
    ref, gate = load_reference(args.csv, class_ids)
    print(f"参考文件: {args.csv}（{len(ref)} 个 Study_id，阈值: "
          f"{'全部 0.5' if np.allclose(gate, 0.5) else np.round(gate, 3).tolist()}）")

    warm_dir = args.pictures_5 or args.images
    test_dir = args.pictures_100 or args.images
    if warm_dir and Path(warm_dir).is_dir():
        warm = collect_images(warm_dir, limit=5)
        for p in warm:
            predict_one_multi(models, p, device, tta=args.tta, weights=W)
        if device.type == "cuda":
            torch.cuda.synchronize()
        print(f"预热完成: {len(warm)} 张（不计时）")
    else:
        print("[warn] 没有找到预热图目录，跳过预热（首次推理会偏慢）")

    if not test_dir or not Path(test_dir).is_dir():
        raise SystemExit("需要提供 100 张测试图目录")
    paths = collect_images(test_dir, limit=args.limit)
    print(f"计时样本: {len(paths)} 张")

    per_image_probs: list[np.ndarray] = []
    per_image_ms: list[float] = []
    for i, p in enumerate(paths):
        times = []
        probs = None
        for _ in range(3):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            probs = predict_one_multi(models, p, device, tta=args.tta, weights=W)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
        per_image_probs.append(probs)
        per_image_ms.append(float(np.mean(times)))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(paths)} 已测，当前均值 {np.mean(per_image_ms):.1f} ms", flush=True)

    avg_ms = float(np.mean(per_image_ms))
    p95_ms = float(np.percentile(per_image_ms, 95))

    # ---------------- 一致性核验 ----------------
    # submission 是按 Study_id 聚合的（同一次检查多张片取均值），
    # 所以比对时也要把该 Study 的所有片都算出来再平均，否则多图检查必然"不一致"。
    study_to_paths: dict[str, list[str]] = {}
    if args.manifest and Path(args.manifest).exists():
        mdf = pd.read_csv(args.manifest)
        for sid, sub in mdf.groupby("Study_id", sort=False):
            study_to_paths[str(sid)] = sub["image_path"].astype(str).tolist()
        print(f"manifest: {args.manifest}（每个 Study 的图片数已加载，用于按 Study 聚合比对）")
    else:
        # 官方命令不带 manifest：就用图片目录里同一 Study 的全部图片做聚合，
        # 与生成 submission 时的"同 Study 多图取均值"保持一致；单图 Study 不受影响。
        for p in paths:
            sid0 = peak_from_path(p, ref)
            if sid0 is not None:
                study_to_paths.setdefault(sid0, []).append(p)
        print(f"未提供 manifest：按目录内同 Study 图片聚合比对（{len(study_to_paths)} 个 Study）")

    prob_cache: dict[str, np.ndarray] = {}

    def study_probs(sid: str, primary: str) -> np.ndarray:
        plist = study_to_paths.get(sid, [primary])
        vecs = []
        for pth in plist:
            if pth not in prob_cache:
                prob_cache[pth] = predict_one_multi(models, pth, device, tta=args.tta, weights=W)
            vecs.append(prob_cache[pth])
        return np.mean(np.stack(vecs, 0), axis=0)

    for p, probs in zip(paths, per_image_probs):
        prob_cache.setdefault(p, probs)

    mismatched = 0
    max_abs_delta = 0.0
    detail = []
    for p in paths:
        sid = peak_from_path(p, ref)
        if sid is None:
            detail.append({"image": Path(p).name, "study_id": None, "status": "未在 submission 中找到"})
            mismatched += 1
            continue
        r = ref[sid]
        agg = study_probs(sid, p)
        delta = float(np.max(np.abs(agg - r)))
        max_abs_delta = max(max_abs_delta, delta)
        bin_new, bin_ref = (agg >= gate).astype(int), (r >= gate).astype(int)
        rank_new, rank_ref = np.argsort(-agg, kind="stable"), np.argsort(-r, kind="stable")
        same = np.array_equal(bin_new, bin_ref) and np.array_equal(rank_new, rank_ref)
        # 规则：二值判定/排序任一改变，或单张概率绝对误差 > 1e-3，均计为不一致
        if (not same) or delta > 1e-3:
            mismatched += 1
        detail.append({"image": Path(p).name, "study_id": sid,
                       "n_images_in_study": len(study_to_paths.get(sid, [p])),
                       "max_abs_delta": round(delta, 8),
                       "binary_same": bool(np.array_equal(bin_new, bin_ref)),
                       "rank_same": bool(np.array_equal(rank_new, rank_ref)),
                       "status": "ok" if (same and delta <= 1e-3) else "MISMATCH"})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(detail).to_csv(args.out, index=False, encoding="utf-8-sig")

    precision = "fp16" if device.type == "cuda" else "fp32"
    verdict = "未超过上限" if avg_ms <= 100.0 else "超过 100ms 上限"
    print("\n[CONCLUSION] 结论：{} 张样本、batch=1 串行、连续 3 轮；单张端到端钟"
          "时间先按每张 3 轮取均值，再对 {} 张取平均，最终平均耗时 {:.0f} ms，{}；"
          "预测结果与 submission.csv 比对 {} 张，不一致 {} 张；推理精度 {}。".format(
              len(paths), len(paths), avg_ms, verdict, len(paths), mismatched, precision))
    print(f"（p95 = {p95_ms:.0f} ms，单张最大概率偏差 = {max_abs_delta:.2e}；明细见 {args.out}）")

    if mismatched:
        print("\n[FAIL] 存在不一致样本，按规则相关成绩不予认定。常见原因：")
        print("  1. 生成 submission.csv 时的 --tta / gate / 模型权重与本次不同；")
        print("  2. 预处理有随机成分（Resize 未固定、随机增强没关）；")
        print("  3. Study_id 与图片名对不上，比对到了错误的行。")
        raise SystemExit(1)
    if avg_ms > 100.0:
        print("\n[FAIL] 平均耗时超过 100ms 上限，按规则需优化推理实现后申请复测。")
        raise SystemExit(1)
    print("\n[PASS] 一致性与耗时均通过（阈值：不一致 0 张，平均 ≤100ms）。")


if __name__ == "__main__":
    main()
