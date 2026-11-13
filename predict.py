"""推理：全项目唯一的推理实现。predict.py / cost.py / 提交脚本都调这里的函数。

为什么强调"唯一"：规则第 7 条 + 补充说明第九条要求 cost.py 必须导入并调用与
predict.py 相同的推理函数，且 100 张样本的二值判定与概率排序与 submission.csv
完全一致（不一致张数必须为 0，概率误差 ≤1e-3）。只要只有一个实现，这条就自动满足。

用法
----
# 生成提交文件（测试集 manifest 只需 image_path 和 Study_id 两列）
python predict.py --ckpt runs/train/best.pt --manifest test_manifest.csv --out submission.csv --tta

# 在验证集上评估（manifest 带标签时）
python predict.py --ckpt runs/train/best.pt --manifest val_manifest.csv --eval
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from common import (
    NUM_CLASSES,
    build_model,
    eval_transform,
    load_rgb,
    safe_macro_auc,
)


# --------------------------------------------------------------------------- #
# 模型加载
# --------------------------------------------------------------------------- #
def load_checkpoint(ckpt_path: str | Path, device: torch.device | None = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["backbone"], ckpt.get("num_classes", NUM_CLASSES), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    # 确定性：关闭 cudnn 自动调优（否则同一张图两次推理可能选到不同 kernel，概率尾数会飘）
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return model, ckpt, device


# --------------------------------------------------------------------------- #
# 单张 / 批量推理（cost.py 直接用 predict_one）
# --------------------------------------------------------------------------- #
def preprocess(path: str | Path, transform) -> torch.Tensor:
    """端到端的一环：读文件 -> PIL -> RGB -> transform。"""
    return transform(load_rgb(path))


@torch.inference_mode()
def forward_probs(model, batch: torch.Tensor, device: torch.device) -> np.ndarray:
    batch = batch.to(device, non_blocking=True)
    use_amp = device.type == "cuda"
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        logits = model(batch)
    return torch.sigmoid(logits.float()).cpu().numpy()


@torch.inference_mode()
def predict_one(model, path: str | Path, transform, device: torch.device, tta: bool = False) -> np.ndarray:
    """单张图 -> 10 维概率。cost.py 计时就是卡在这个函数外面。"""
    x = preprocess(path, transform)
    probs = forward_probs(model, x.unsqueeze(0), device)[0]
    if tta:
        # 水平翻转是确定性算子，不引入随机性；不做跨样本统计，符合规则第 6 条
        probs = (probs + forward_probs(model, torch.flip(x, dims=[2]).unsqueeze(0), device)[0]) / 2.0
    return probs.astype(np.float32)


def predict_batch(model, paths: list[str], transform, device: torch.device,
                  batch_size: int = 8, tta: bool = False) -> np.ndarray:
    """批量版：与 predict_one 完全相同的预处理与后处理，只是攒成 batch 加速。"""
    out = []
    for i in range(0, len(paths), batch_size):
        chunk = paths[i:i + batch_size]
        batch = torch.stack([preprocess(p, transform) for p in chunk])
        probs = forward_probs(model, batch, device)
        if tta:
            probs = (probs + forward_probs(model, torch.flip(batch, dims=[3]), device)) / 2.0
        out.append(probs)
    return np.concatenate(out, axis=0) if out else np.zeros((0, NUM_CLASSES), dtype=np.float32)


# --------------------------------------------------------------------------- #
# 结果聚合 -> submission.csv
# --------------------------------------------------------------------------- #
def aggregate_by_study(study_ids: list[str], probs: np.ndarray) -> dict[str, np.ndarray]:
    """同一个 Study_id 多张图 -> 概率取平均（确定性聚合，规则允许）。"""
    buckets: dict[str, list[np.ndarray]] = {}
    for sid, p in zip(study_ids, probs):
        buckets.setdefault(str(sid), []).append(p)
    return {sid: np.mean(np.stack(ps, 0), axis=0) for sid, ps in buckets.items()}


def gate_for_class(gate, cid: int) -> float:
    """gate 可以是 None / 单一浮点 / {'1': 0.42, ...}（按**官方**类别 id 索引）。"""
    if gate is None:
        return 0.5
    if isinstance(gate, dict):
        return float(gate.get(str(cid), gate.get(str(int(cid)), 0.5)))
    return float(gate)


def to_submission(study_probs: dict[str, np.ndarray], gate=None,
                  class_ids: list[int] | None = None,
                  study_subject: dict[str, str] | None = None) -> pd.DataFrame:
    """长表：每个 Study_id 恰好 N 行。

    class_ids 是官方类别 id（本赛事 1..10）。内部下标 c 必须映射回官方 id，
    否则平台读到的类别全错位 —— 这是最隐蔽也最致命的一个坑。

    study_subject 给出 Study_id -> Subject_id 映射时，输出第一列为 Subject_id，
    与官方 submission 示例的列顺序完全一致。
    """
    class_ids = list(class_ids) if class_ids is not None else list(range(NUM_CLASSES))
    n = len(class_ids)
    rows = []
    for sid in sorted(study_probs):
        p = study_probs[sid]
        if p.shape[0] != n:
            raise ValueError(f"{sid}: 概率维度 {p.shape} 与类别数 {n} 不符")
        for c in range(n):
            cid = class_ids[c]
            g = gate_for_class(gate, cid)
            row = {
                "Study_id": sid,
                "Predict_class": int(cid),
                "Probability": round(float(np.clip(p[c], 0.0, 1.0)), 6),
                # 默认阈值 0.5 时该列留空（规则原文：否则该列不填）
                "Gate": "" if abs(g - 0.5) < 1e-9 else round(g, 4),
            }
            if study_subject is not None:
                row = {"Subject_id": study_subject.get(sid, ""), **row}
            rows.append(row)
    columns = ["Study_id", "Predict_class", "Probability", "Gate"]
    if study_subject is not None:
        columns = ["Subject_id"] + columns
    return pd.DataFrame(rows, columns=columns)


def subject_from_path(path: str) -> str:
    """从 test/subject_XXXXX/study_YYYYY/pic.jpg 的结构里取 Subject_id。"""
    parts = Path(path).parts
    for part in reversed(parts[:-1]):
        if part.startswith("subject_"):
            return part
    return ""


# --------------------------------------------------------------------------- #
# 数据集包装（只依赖 image_path，不需要标签）
# --------------------------------------------------------------------------- #
class PathDataset(Dataset):
    def __init__(self, paths: list[str], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        return self.transform(load_rgb(self.paths[i])), self.paths[i]


def load_ensemble(ckpt_paths: list[str], device: torch.device | None = None):
    """加载多个 ckpt 组成集成；返回 [(model, ckpt), ...] 与 device。"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for p in ckpt_paths:
        m, c, _ = load_checkpoint(p, device)
        models.append((m, c))
    ids = [tuple(c.get("class_ids") or range(int(c.get("num_classes", NUM_CLASSES)))) for _, c in models]
    if len(set(ids)) > 1:
        raise SystemExit(f"集成里的模型类别不一致: {ids}")
    return models, device


def _norm_weights(weights, n: int) -> np.ndarray:
    if weights is None:
        return np.ones(n, dtype=np.float64) / n
    w = np.asarray([float(x) for x in weights], dtype=np.float64)
    if len(w) != n:
        raise SystemExit(f"--weights 数量 {len(w)} 与模型数 {n} 不一致")
    if w.sum() <= 0:
        raise SystemExit("--weights 之和必须为正")
    return w / w.sum()


def predict_multi(models, paths: list[str], device: torch.device,
                  batch_size: int = 1, tta: bool = False,
                  weights=None) -> np.ndarray:
    """多模型概率加权平均；每个模型按自己的 img_size 预处理。"""
    w = _norm_weights(weights, len(models))
    per_model = []
    for model, ckpt in models:
        transform = eval_transform(int(ckpt["img_size"]))
        per_model.append(predict_batch(model, paths, transform, device, batch_size=batch_size, tta=tta))
    return np.tensordot(w, np.stack(per_model, 0), axes=(0, 0))


def predict_one_multi(models, path: str, device: torch.device, tta: bool = False,
                      weights=None) -> np.ndarray:
    """单张图走完整集成（cost.py 计时用它，保证与提交完全同路径）。"""
    w = _norm_weights(weights, len(models))
    probs = []
    for model, ckpt in models:
        transform = eval_transform(int(ckpt["img_size"]))
        probs.append(predict_one(model, path, transform, device, tta=tta))
    return np.tensordot(w, np.stack(probs, 0), axes=(0, 0)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, nargs="+", help="一个或多个 ckpt；多个则做概率平均集成")
    ap.add_argument("--manifest", required=True, help="需要 image_path 列；Study_id 缺省用文件名")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tta", action="store_true", help="水平翻转 TTA（确定性）")
    ap.add_argument("--gate", type=float, default=0.5)
    ap.add_argument("--gates-json", default=None, help="tune_gate.py 产出的按类别阈值")
    ap.add_argument("--eval", action="store_true", help="manifest 带 y0..y9 时算宏平均 AUC")
    ap.add_argument("--weights", default=None, help="集成权重，如 0.3,0.7（默认等权）")
    args = ap.parse_args()

    models, device = load_ensemble(args.ckpt)
    ckpt = models[0][1]
    class_ids = ckpt.get("class_ids") or list(range(int(ckpt.get("num_classes", NUM_CLASSES))))
    n = len(class_ids)
    desc = " + ".join(f"{c['backbone']}@{c['img_size']}" for _, c in models)
    print(f"ckpt: {len(models)} 个模型 [{desc}] | 类别 id={class_ids} | device={device}")

    df = pd.read_csv(args.manifest, dtype={"Study_id": str, "Subject_id": str})
    if "image_path" not in df.columns:
        raise SystemExit("manifest 缺少 image_path 列，先用 data_check.py 生成")
    study_ids = df["Study_id"].astype(str).tolist() if "Study_id" in df.columns \
        else [Path(p).stem for p in df["image_path"]]
    if "Subject_id" in df.columns:
        study_subject = dict(zip(df["Study_id"].astype(str), df["Subject_id"].astype(str)))
    else:
        study_subject = {}
        for sid, p in zip(study_ids, df["image_path"].astype(str)):
            study_subject.setdefault(sid, subject_from_path(p))

    t0 = time.time()
    weights = [float(x) for x in args.weights.split(",")] if args.weights else None
    probs = predict_multi(models, df["image_path"].astype(str).tolist(), device,
                          batch_size=args.batch_size, tta=args.tta, weights=weights)
    print(f"推理 {len(probs)} 张图用时 {time.time()-t0:.1f}s（含读图，未计模型加载）")

    if args.eval:
        ycols = [f"y{c}" for c in range(n)]
        if all(c in df.columns for c in ycols):
            auc, valid = safe_macro_auc(df[ycols].to_numpy(), probs)
            print(f"宏平均 AUC（{len(valid)}/{n} 个有效类别）= {auc:.4f}")
        else:
            print("manifest 里没有 y0..yN 列，跳过 AUC 评估")

    study_probs = aggregate_by_study(study_ids, probs)
    gate = args.gate
    if args.gates_json:
        import json
        gate = json.loads(Path(args.gates_json).read_text(encoding="utf-8"))
        print(f"按类别阈值: {gate}")
    sub = to_submission(study_probs, gate=gate, class_ids=class_ids,
                        study_subject=study_subject)
    sub.to_csv(args.out, index=False, encoding="utf-8")
    print(f"已写 {args.out}: {len(study_probs)} 个 Study_id, {len(sub)} 行, {len(sub.columns)} 列")
    print(sub.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
