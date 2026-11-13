"""数据与预处理工具入口（附录 A 要求的 utils.py）。

实现集中在 common.py，这里做稳定导出：标签解析、manifest 配对、预处理、
数据集与宏平均 AUC。
"""

from common import (
    NUM_CLASSES,
    ChestDataset,
    build_manifest,
    eval_transform,
    load_rgb,
    parse_label_cell,
    read_label_table,
    safe_macro_auc,
    scan_images,
    seed_everything,
    train_transform,
)

__all__ = [
    "NUM_CLASSES", "ChestDataset", "build_manifest", "eval_transform", "load_rgb",
    "parse_label_cell", "read_label_table", "safe_macro_auc", "scan_images",
    "seed_everything", "train_transform",
]
