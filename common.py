"""共享工具：标签解析、图片匹配、数据集、模型工厂。

设计要点
--------
1. 官方只给了"图片文件夹说明"的截图，目录结构没有文字描述，所以这里不写死路径：
   先扫描磁盘上所有图片和标注 csv，再按 Study_id 自动配对，产出 manifest.csv。
2. 标注 csv 的格式也可能是"长表"(每行一个 Predict_class) 或"宽表"(每个类别一列 0/1)，
   read_label_table 两种都能吃。
3. 训练 / 推理 / 计时全部共用同一个 manifest 与同一套预处理，避免三处实现漂移。
"""

from __future__ import annotations

import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

NUM_CLASSES = 10
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# timm 的权重来自 HuggingFace，国内直连经常被重置；默认走镜像（可用环境变量覆盖）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 元数据列，不是标签
META_COLS = {
    "study_id", "studyid", "subject_id", "subjectid", "predict_class", "predictclass",
    "viewposition", "view_position", "view position", "view", "patientorientation",
    "patient_orientation", "patient orientation", "rows", "columns", "support devices",
    "support_devices", "image", "image_path", "path", "filename", "file", "img", "image_index",
    "patient age", "patient_age", "age", "patient gender", "patient_gender", "gender",
}


# --------------------------------------------------------------------------- #
# 基础
# --------------------------------------------------------------------------- #
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_label_cell(value) -> list[int]:
    """把一格里的类别 id 抠出来。支持 '3'、'3;5'、'[0, 3]'、'0|3|5'。"""
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [int(v) for v in value]
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null", "[]", "-"}:
        return []
    return [int(v) for v in re.findall(r"\d+", s)]


def _find_col(columns: list[str], *candidates: str) -> str | None:
    lower = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


# --------------------------------------------------------------------------- #
# 标注表 -> 宽表 (Study_id, y0..y9)
# --------------------------------------------------------------------------- #
def scan_class_ids(csv_paths: list[str | Path]) -> list[int]:
    """扫出标注里出现的全部官方类别 id（本赛事是 1..10，不是 0..9）。"""
    ids: set[int] = set()
    for p in csv_paths:
        df = pd.read_csv(p, dtype=str, encoding="utf-8-sig")
        col = _find_col(list(df.columns), "predict_class", "predictclass", "class_id", "label", "labels")
        if col is None:
            continue
        for v in df[col]:
            ids.update(parse_label_cell(v))
    if not ids:
        raise ValueError(f"在 {csv_paths} 里没找到任何类别 id")
    return sorted(ids)


def read_label_table(path: str | Path, class_ids: list[int] | None = None,
                     num_classes: int | None = None) -> tuple[pd.DataFrame, list[int]]:
    """标注 csv -> 宽表 (Study_id, y0..yN-1)，并返回官方类别 id 顺序。

    官方 id 到内部下标做一次映射：例如 class_ids=[1..10] 时，官方 10 -> 内部 y9。
    提交时必须再映射回去，否则整份 submission.csv 作废。
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    cols = list(df.columns)

    study_col = _find_col(cols, "study_id", "studyid", "study", "检查id", "检查编号", "id")
    if study_col is None:
        raise ValueError(f"{path}: 找不到 Study_id 列，实际列为 {cols}")

    out = pd.DataFrame({"Study_id": df[study_col].astype(str).str.strip()})
    subj_col = _find_col(cols, "subject_id", "subjectid", "patient_id", "patientid")
    if subj_col:
        out["Subject_id"] = df[subj_col].astype(str).str.strip()

    pc_col = _find_col(cols, "predict_class", "predictclass", "class_id", "label", "labels")
    if pc_col is None:
        raise ValueError(f"{path}: 找不到 Predict_class 列，实际列为 {cols}")

    # 官方 id 顺序：给定的优先，否则从本文件扫
    if class_ids is None:
        class_ids = sorted({i for v in df[pc_col] for i in parse_label_cell(v)})
    class_ids = list(class_ids)
    index_of = {cid: i for i, cid in enumerate(class_ids)}
    n = num_classes or len(class_ids)

    grouped: dict[str, set[int]] = {}
    for sid, cell in zip(out["Study_id"], df[pc_col]):
        grouped.setdefault(sid, set()).update(parse_label_cell(cell))

    unknown: set[int] = set()
    official = set(class_ids)
    rows_labels: list[set[int]] = []
    for sid in out["Study_id"]:
        got = grouped.get(sid, set())
        unknown.update(got - official)
        rows_labels.append({index_of[o] for o in got if o in index_of})
    for c in range(n):
        out[f"y{c}"] = [1 if c in labs else 0 for labs in rows_labels]

    if unknown:
        print(f"[warn] {Path(path).name}: 出现 class_ids 之外的 id {sorted(unknown)}，已忽略")
    return _finalize_labels(out, n), class_ids


def _finalize_labels(out: pd.DataFrame, num_classes: int) -> pd.DataFrame:
    ycols = [f"y{c}" for c in range(num_classes)]
    out = out.groupby("Study_id", as_index=False, sort=False).agg(
        {**{c: "max" for c in ycols}, **({"Subject_id": "first"} if "Subject_id" in out.columns else {})}
    )
    for c in ycols:
        out[c] = out[c].astype("int8")
    return out


# --------------------------------------------------------------------------- #
# 图片扫描与 Study_id 配对
# --------------------------------------------------------------------------- #
def scan_images(root: str | Path) -> pd.DataFrame:
    rows = []
    for p in Path(root).rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            rows.append({"image_path": str(p), "stem": p.stem, "parent": p.parent.name})
    return pd.DataFrame(rows)


def build_manifest(
    studies: pd.DataFrame,
    images: pd.DataFrame,
    num_classes: int = NUM_CLASSES,
) -> tuple[pd.DataFrame, dict]:
    """把每个 Study_id 和它的图（可能多张）配上。

    本赛事实际结构：train/subject_XXXXX/study_YYYYY/pic_ZZZZZZ.jpg
    —— 图片名不是 Study_id，靠**父目录名**配上，所以按 父目录 > 文件名 > 前缀 的顺序找。
    """
    from bisect import bisect_left

    by_stem: dict[str, list[str]] = {}
    by_parent: dict[str, list[str]] = {}
    for row in images.itertuples(index=False):
        by_stem.setdefault(str(row.stem), []).append(row.image_path)
        by_parent.setdefault(str(row.parent), []).append(row.image_path)

    sorted_stems = sorted(by_stem)
    sorted_parents = sorted(by_parent)

    def prefix_hits(keys: list[str], index: dict[str, list[str]], sid: str) -> list[str]:
        out = []
        for sep in ("_", "-"):
            prefix = sid + sep
            i = bisect_left(keys, prefix)
            while i < len(keys) and keys[i].startswith(prefix):
                out.extend(index[keys[i]])
                i += 1
        return out

    rows = []
    stats = {"studies": 0, "matched": 0, "unmatched_ids": [], "multi_image_studies": 0}

    for row in studies.itertuples(index=False):
        sid = str(row.Study_id)
        stats["studies"] += 1
        paths = (by_parent.get(sid) or by_stem.get(sid)
                 or prefix_hits(sorted_parents, by_parent, sid)
                 or prefix_hits(sorted_stems, by_stem, sid))
        if not paths:
            stats["unmatched_ids"].append(sid)
            continue

        stats["matched"] += 1
        if len(paths) > 1:
            stats["multi_image_studies"] += 1
        labels = {f"y{c}": int(getattr(row, f"y{c}")) for c in range(num_classes)}
        for p in sorted(paths):
            rows.append({"Study_id": sid, "image_path": p, **labels})

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError(
            "一张图都没配上。请检查 --images 指向的目录是否包含图片，"
            "以及图片名是否以 Study_id 命名（或放在以 Study_id 命名的子目录里）。"
        )
    if "Subject_id" in studies.columns:
        mapping = dict(zip(studies["Study_id"].astype(str), studies["Subject_id"].astype(str)))
        manifest["Subject_id"] = manifest["Study_id"].map(mapping)
    return manifest, stats


# --------------------------------------------------------------------------- #
# 数据集 / 预处理
# --------------------------------------------------------------------------- #
def load_rgb(path: str | Path, draft_size: int = 512) -> Image.Image:
    """统一读成 RGB。X 光片常见 16bit / 灰度 / 带 alpha，这里一并处理。

    实测：本赛事 6MP JPEG 直接用 PIL 解码只有 15 img/s/核，训练时数据管道会成为瓶颈。
    JPEG 可以用 draft() 让解码器直接按 1/2、1/4、1/8 降采样输出，实测提速 2 倍以上，
    对 320px 训练没有可见质量损失。
    """
    img = Image.open(path)
    if img.format == "JPEG" and draft_size:
        try:
            img.draft("RGB", (draft_size, draft_size))
        except Exception:
            pass
    if img.mode in ("I;16", "I;16B", "I"):
        arr = np.asarray(img, dtype=np.float32)
        hi = float(arr.max()) or 1.0
        img = Image.fromarray((arr / hi * 255.0).clip(0, 255).astype(np.uint8))
    return img.convert("RGB")


def train_transform(img_size: int) -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.80, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.RandomRotation(degrees=7)], p=0.3),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def eval_transform(img_size: int) -> T.Compose:
    resize = int(round(img_size * 1.14))
    return T.Compose([
        T.Resize(resize),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


class ChestDataset(torch.utils.data.Dataset):
    def __init__(self, manifest: pd.DataFrame, transform, num_classes: int = NUM_CLASSES):
        self.paths = manifest["image_path"].tolist()
        self.labels = manifest[[f"y{c}" for c in range(num_classes)]].to_numpy(dtype=np.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = self.transform(load_rgb(self.paths[i]))
        return img, torch.from_numpy(self.labels[i]), self.paths[i]


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
# 优先用 torchvision 的权重：它从 download.pytorch.org 拉，国内一般可达；
# timm 走 HuggingFace，国内经常被墙（可用 HF_ENDPOINT=https://hf-mirror.com 绕过）。
_TV_BACKBONES = {
    "densenet121": "DenseNet121_Weights",
    "resnet50": "ResNet50_Weights",
    "mobilenet_v3_large": "MobileNet_V3_Large_Weights",
    "efficientnet_b0": "EfficientNet_B0_Weights",
    "convnext_tiny": "ConvNeXt_Tiny_Weights",
    "convnext_small": "ConvNeXt_Small_Weights",
    "efficientnet_b1": "EfficientNet_B1_Weights",
}


def _replace_head(model: torch.nn.Module, num_classes: int) -> torch.nn.Module:
    """把分类头换成 num_classes 输出；densenet/fc/classifier 三种命名都覆盖。"""
    for attr in ("classifier", "fc", "head"):
        if not hasattr(model, attr):
            continue
        head = getattr(model, attr)
        if isinstance(head, torch.nn.Linear):
            setattr(model, attr, torch.nn.Linear(head.in_features, num_classes))
            return model
        if isinstance(head, torch.nn.Sequential):
            for i in range(len(head) - 1, -1, -1):
                if isinstance(head[i], torch.nn.Linear):
                    head[i] = torch.nn.Linear(head[i].in_features, num_classes)
                    return model
    raise ValueError("这个骨干网没找到可替换的分类头，请换一个或改用 timm 名字")


def build_model(backbone: str, num_classes: int = NUM_CLASSES, pretrained: bool = True,
                weights_path: str | None = None):
    """backbone 支持 torchvision 短名（densenet121 等）或 timm 全名（xxx.yyy）。"""
    import torchvision

    if backbone in _TV_BACKBONES:
        weights = getattr(torchvision.models, _TV_BACKBONES[backbone]) if pretrained else None
        model = torchvision.models.get_model(backbone, weights=weights)
        model = _replace_head(model, num_classes)
    else:
        import timm

        model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)

    if weights_path:
        # 离线场景：直接从本地 .pth/.bin 加载（HF 或 pytorch 都下不动时用）
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state.get("model", state))
        state = {k.replace("module.", ""): v for k, v in state.items()
                 if not k.endswith(("num_batches_tracked",))}
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[weights] 从 {weights_path} 加载：缺 {len(missing)} 个键，多 {len(unexpected)} 个键")
    return model


def safe_macro_auc(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, list[int]]:
    """宏平均 AUC；跳过验证集里只有单一类别的列（sklearn 会报错）。"""
    from sklearn.metrics import roc_auc_score

    aucs, valid = [], []
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        if col.min() == col.max():
            continue
        aucs.append(roc_auc_score(col, y_prob[:, c]))
        valid.append(c)
    return (float(np.mean(aucs)) if aucs else float("nan")), valid
