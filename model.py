"""模型定义入口（附录 A 要求的 model.py）。

实现集中在 common.py，这里做稳定导出，保证评审可以按标准结构找到模型定义。
"""

from common import NUM_CLASSES, _replace_head, build_model

BACKBONES = [
    "densenet121", "resnet50", "mobilenet_v3_large",
    "efficientnet_b0", "efficientnet_b1",
    "convnext_tiny", "convnext_small",
]

__all__ = ["NUM_CLASSES", "BACKBONES", "build_model", "_replace_head"]
