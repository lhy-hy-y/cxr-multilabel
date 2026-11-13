# 胸部 X 光多标签分类技术报告

**赛项**：NCCCU AI Challenge 2026 · 胸部 X 光多标签分类  
**提交主体**：林宏熠  
**报告版本**：v1.0（2026-11-14）  
**代码与结果**：见本仓库 `README.md`、`results/`、`docs/MODEL_CARD.md`

---

## 摘要

本工作面向胸部 X 光片的 10 类多标签分类任务，提交格式为 Study 级长表（每个 Study 恰好 10 行）。方法采用两个不同随机种子的 **ConvNeXt-Tiny @320px** 模型，以 torchvision 官方 ImageNet-1K 权重初始化，使用 BCEWithLogitsLoss + 类别正样本权重、AdamW、余弦退火、AMP FP16 与 EMA 训练；推理阶段两个模型等权平均，同一 Study 的多张图概率取算术平均，阈值固定 0.5，不做 TTA。

本地验证集（859 个 Study / 1,422 张图）结果：

| 指标 | 数值 |
|---|---:|
| 单模型 r4（seed 42）图级宏平均 AUC | 0.8004 |
| 单模型 r6（seed 123）图级宏平均 AUC | 0.8013 |
| 双模型集成图级宏平均 AUC | **0.8070** |
| 双模型集成 Study 级宏平均 AUC | **0.7994** |
| 推理权重总大小 | 212.4 MB |
| 本机单张耗时（RTX 4060 Laptop, FP16, batch=1） | 76 ms（p95 88 ms） |
| 与 submission.csv 一致性核验 | 100/100 张一致，最大概率偏差 4.9e-7 |

结论：该方案是一个可复现、合规、成本可控的比赛基线；但验证集规模小、稀有类别样本少、未做外部验证与概率校准，**不能直接用于临床诊断**。

---

## 1. 任务与数据

### 1.1 任务定义

- 输入：胸部 X 光图像，图像按 `subject_XXXXX/study_YYYYY/pic_ZZZZZZ.jpg` 组织。
- 输出：每个 Study 在 10 个官方类别（`Predict_class = 1..10`）上的概率。
- 提交：长表，每个 Study 恰好 10 行，列为 `Subject_id, Study_id, Predict_class, Probability, Gate`；`Gate` 留空表示使用默认阈值 0.5。
- 本地评估指标：宏平均 AUC（macro AUC）。`safe_macro_auc` 会跳过验证集中只有单一类别的列。

> 数据文件未提供类别 id 到医学名称的映射，因此本报告统一使用官方类别 id 1–10，不对具体疾病名称作推断。

### 1.2 数据规模

| 划分 | Study 数 | Subject 数 | 图像数 | 多图 Study 数 |
|---|---:|---:|---:|---:|
| train | 18,513 | 11,055 | 30,344 | — |
| valid | 859 | 228 | 1,422 | 468 |
| test | 1,414 | 136 | 2,218 | — |

三个划分的 Subject 互不重叠（train/valid、train/test、valid/test 的交集均为 0），不存在患者级泄漏。

### 1.3 类别分布（Study 级正样本数）

| 类别 id | train 正样本 | train 占比 | valid 正样本 | valid 占比 |
|---:|---:|---:|---:|---:|
| 1 | 622 | 3.4% | 24 | 2.8% |
| 2 | 900 | 4.9% | 37 | 4.3% |
| 3 | 942 | 5.1% | 46 | 5.4% |
| 4 | 1,436 | 7.8% | 68 | 7.9% |
| 5 | 2,404 | 13.0% | 115 | 13.4% |
| 6 | 3,885 | 21.0% | 177 | 20.6% |
| 7 | 3,968 | 21.4% | 193 | 22.5% |
| 8 | 4,508 | 24.3% | 214 | 24.9% |
| 9 | 4,654 | 25.1% | 260 | 30.3% |
| 10 | 6,683 | 36.1% | 306 | 35.6% |

训练集 18,513 个 Study 中，11,597 个只有 1 个标签（62.6%），6,916 个有 2 个及以上标签（37.4%）；标签数最多为 7。类别 1–4 明显长尾，验证集中类别 1 只有 24 个正样本，类别 2 只有 37 个。

### 1.4 视图分布（图像级）

| 视图 | train | valid |
|---|---:|---:|
| AP | 12,152 | 584 |
| PA | 7,604 | 364 |
| LATERAL | 6,554 | 284 |
| LL | 2,831 | 151 |
| PA LLD | 1 | 0 |

训练集中另有 1,202 张图像缺失或使用其他 `ViewPosition` 取值（valid 为 39 张），未计入上表。训练集中约 31% 的图像是侧位片（LATERAL + LL），但当前模型没有视图条件化，正位与侧位混在一起训练。

---

## 2. 方法

### 2.1 总体流程

```
data_check.py  ->  manifest.csv + class_ids.json
      |
      v
train.py       ->  best.pt（按 valid macro AUC 选最优）
      |
      v
predict.py     ->  submission.csv（唯一推理实现）
      |
      +--> validate_submission.py（格式自检）
      +--> cost.py（耗时 + 与 submission 一致性核验）
```

设计原则：**训练、推理、计时共用同一套 manifest、预处理和后处理**，避免三处实现漂移；`cost.py` 直接导入 `predict.py` 的推理函数，保证一致性核验具有实际意义。

### 2.2 标签解析与 manifest

- 支持长表（每行一个 `Predict_class`）与宽表（每个类别一列 0/1）。
- 官方类别 id 与内部下标做显式映射：官方 1..10 → 内部 0..9；提交时再映射回官方 id。
- 同一 Study 多行标签取 `max` 合并为多标签向量。
- 图片通过父目录名、文件名、前缀三级匹配到 Study_id；本赛事图片名不是 Study_id，父目录名才是。
- 输出 `manifest.csv`（`image_path`, `Study_id`, `Subject_id`, `y0..y9`）与 `class_ids.json`。

### 2.3 预处理与增强

**读取**：`load_rgb` 统一转 RGB，兼容 16-bit / 灰度 / 带 alpha；JPEG 使用 `draft()` 让解码器按 1/2、1/4 降采样，降低数据管道开销。

**训练增强**：

- RandomResizedCrop 到 320×320，scale=(0.80, 1.0)，ratio=(0.9, 1.1)
- RandomHorizontalFlip(p=0.5)
- RandomRotation(±7°)，p=0.3
- ColorJitter(brightness=0.15, contrast=0.15)
- ImageNet 均值/标准差归一化

**验证/推理**：Resize 到 365（320×1.14）后 CenterCrop 320，不做随机增强。

### 2.4 模型

- 骨干：`torchvision.models.convnext_tiny`，权重 `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`。
- 分类头替换为 10 维线性输出；模型输出 logits，推理时统一 `sigmoid`。
- 两个独立训练的同架构模型：r4（seed 42）与 r6（seed 123），均使用 EMA 权重。
- 推理权重各 106.2 MB，合计 212.4 MB。

### 2.5 训练配置

以下为 `train.py` 默认配置；checkpoint 元数据记录了实际最佳 epoch。

| 项目 | 配置 |
|---|---|
| 损失 | BCEWithLogitsLoss，`pos_weight = clip((N-P)/P, 1, 20)`（按 Study 去重统计） |
| 优化器 | AdamW，lr=3e-4，weight_decay=1e-4；norm/bias 不做 weight decay |
| 调度 | 0.5 epoch warmup + cosine decay |
| 精度 | AMP FP16 |
| 梯度裁剪 | max_norm=5.0 |
| EMA | decay=0.999，带 warmup |
| batch size | 16 |
| epochs | 8（默认） |
| 模型选择 | 每个 epoch 同时评估 raw 与 EMA，取验证集宏平均 AUC 更高者保存 |
| 随机种子 | 42 / 123 |

> 提交包未保留完整训练日志；上表来自 `train.py` 默认值与 checkpoint 元数据，未补造未记录的超参数。

### 2.6 推理与提交

- 两个模型各自按自己的 `img_size` 预处理，输出概率等权平均。
- 同一 Study 多张图：概率算术平均（确定性、不拟合参数）。
- 阈值固定 0.5，`Gate` 列留空；不做 TTA。
- 推理关闭 cudnn benchmark、开启 deterministic，降低同图多次推理的尾数漂移。
- 提交前用 `validate_submission.py` 检查列名、类别齐全、概率范围、Study 覆盖、重复行。

### 2.7 合规设计

- 只使用官方 train 做有监督微调；官方 valid 不参与训练。
- 不使用外部医学数据、医学预训练权重、自监督预训练或知识蒸馏。
- 不使用测试集标签、测试集统计量或阈值拟合。
- 推理确定性：固定预处理、固定阈值、固定聚合方式；无 TTA。
- 提交包记录预训练权重 SHA-256、推理权重 SHA-256、许可证全文。
- 推理权重 212.4 MB，全包解压约 321.5 MB，均低于 500 MB 限制。

---

## 3. 实验结果

### 3.1 单模型结果

| 模型 | 骨干 | 输入 | 种子 | 最佳 epoch | 验证集图级宏平均 AUC |
|---|---|---|---:|---:|---:|
| r4 | ConvNeXt-Tiny | 320 | 42 | 6 | 0.8004 |
| r6 | ConvNeXt-Tiny | 320 | 123 | 7 | 0.8013 |

以上数值来自 checkpoint 元数据，与 `train.py` 的图级评估口径一致。

### 3.2 集成与聚合方式

在官方 valid 上复算（859 Study / 1,422 图，无 TTA）：

| 配置 | 图级宏平均 AUC | Study 级宏平均 AUC |
|---|---:|---:|
| 双模型等权集成 | **0.8070** | **0.7994** |

Study 级聚合方式对比：

| 聚合方式 | Study 级宏平均 AUC |
|---|---:|
| mean（提交采用） | **0.7994** |
| max | 0.7939 |
| meanmax | 0.7985 |
| noisyor | 0.7695 |

均值聚合在本验证集上最好，且与提交实现一致。

### 3.3 逐类结果（双模型集成，mean 聚合）

| 类别 id | valid 正样本（Study） | Study AUC | Study AUPRC |
|---:|---:|---:|---:|
| 1 | 24 | 0.6786 | 0.0726 |
| 2 | 37 | 0.8368 | 0.2879 |
| 3 | 46 | 0.7698 | 0.2030 |
| 4 | 68 | 0.7609 | 0.2378 |
| 5 | 115 | 0.8805 | 0.4377 |
| 6 | 177 | 0.7724 | 0.4363 |
| 7 | 193 | 0.7899 | 0.5012 |
| 8 | 214 | 0.7175 | 0.4393 |
| 9 | 260 | 0.9062 | 0.7846 |
| 10 | 306 | 0.8814 | 0.8367 |
| **宏平均** | — | **0.7994** | — |

完整逐类结果（含图级 AUC/AUPRC）见 `results/eval_study_per_class.csv`，原始 JSON 见 `results/eval_study.json`。

### 3.4 提交文件统计

| 项目 | 数值 |
|---|---:|
| 行数 | 14,140 |
| Study 数 | 1,414 |
| Subject 数 | 136 |
| 类别数 | 10（id 1..10） |
| `Gate` 非空行数 | 0 |
| 概率范围 | 0.000905 – 0.989797 |
| 列顺序 | `Subject_id, Study_id, Predict_class, Probability, Gate` |

### 3.5 成本与一致性

| 项目 | 结果 |
|---|---|
| 推理权重 | 2 × 106.2 MB = 212.4 MB |
| 全包解压 | 约 321.5 MB（≤ 500 MB） |
| 精度 | FP16 |
| batch | 1（官方计时口径） |
| 单张耗时 | 76 ms（p95 88 ms，RTX 4060 Laptop） |
| TTA | 无 |
| 与 submission 一致性 | 100/100 张一致，最大概率偏差 4.9e-7 |
| 官方 T4 复测 | 待赛事环境复测 |

### 3.6 可复现性

```bash
# 1) 数据体检与 manifest
python data_check.py --images data/train --labels data/train.csv --out-dir runs/data_check/train
python data_check.py --images data/valid --labels data/valid.csv --out-dir runs/data_check/valid
python data_check.py --images data/test  --out-dir runs/data_check/test

# 2) 训练两个种子
python train.py --train-manifest runs/data_check/train/manifest.csv \
  --val-manifest runs/data_check/valid/manifest.csv \
  --backbone convnext_tiny --img-size 320 --epochs 8 --batch-size 16 \
  --seed 42 --out-dir runs/r4_convnext_320
python train.py --train-manifest runs/data_check/train/manifest.csv \
  --val-manifest runs/data_check/valid/manifest.csv \
  --backbone convnext_tiny --img-size 320 --epochs 8 --batch-size 16 \
  --seed 123 --out-dir runs/r6_convnext_320

# 3) 生成提交
python predict.py --ckpt runs/r4_convnext_320/best.pt runs/r6_convnext_320/best.pt \
  --manifest runs/data_check/test/manifest.csv --out submission.csv --weights 0.5,0.5

# 4) 格式自检
python validate_submission.py --csv submission.csv --ids runs/data_check/test/study_ids.txt

# 5) 耗时与一致性
python cost.py submission.csv data/test <pictures_100> <pictures_5> \
  --ckpt runs/r4_convnext_320/best.pt runs/r6_convnext_320/best.pt \
  --manifest runs/data_check/test/manifest.csv

# 6) Study 级逐类评估
python tools/eval_study.py --ckpt runs/r4_convnext_320/best.pt runs/r6_convnext_320/best.pt \
  --manifest runs/data_check/valid/manifest.csv \
  --out-json runs/eval_study.json --out-csv runs/eval_study_per_class.csv
```

---

## 4. 讨论

### 4.1 主要发现

1. **集成有效但增益有限**：两个同架构不同种子的模型把图级宏平均 AUC 从 0.8004/0.8013 提升到 0.8070；Study 级为 0.7994。
2. **长尾是主要瓶颈**：类别 1 只有 24 个验证正样本，Study AUC 0.6786；类别 8 也只有 0.7175。类别 9/10 的 AUC 在 0.88–0.91，说明数据量充足的类别明显更好。
3. **验证集很小**：859 个 Study，稀有类别只有几十个正样本，逐类 AUC 的置信区间很宽；当前结果不足以支撑临床性能声明。
4. **视图混训**：约 31% 训练图是侧位片，但模型没有视图条件化；这可能是类别 1/8 表现偏弱的原因之一。
5. **聚合方式有影响**：Study 级 mean 聚合（0.7994）优于 max（0.7939）与 noisyor（0.7695），说明“多图取均值”是当前数据下合理且确定性的选择。

### 4.2 局限

- 单中心、单数据源；没有外部医院/设备验证。
- 官方 valid 规模小，稀有类别正样本少，指标方差大。
- 标签为图像/Study 级弱标签，存在标签噪声与共现偏差。
- 未做概率校准；固定阈值 0.5 不等于临床可用阈值。
- 只报告 AUC/AUPRC，没有敏感度、特异度、PPV、NPV、决策曲线或医生读片研究。
- 输入是比赛提供的 JPEG，不是原始 DICOM；未验证窗宽窗位、设备差异、体位差异。
- 没有病灶定位/热图，医生难以判断模型依据。
- 比赛规则禁止医学预训练与外部数据；产品化时可解除该限制，但需重新验证与授权。

### 4.3 与临床可用的差距

AUC 0.80 左右说明模型有信号，但离临床辅助诊断仍有系统性差距：需要多中心外部验证、前瞻性研究、逐类 operating point、校准、亚组公平性、OOD/非胸片拒识、DICOM 接入、可解释性、质量体系与注册。当前成果更适合定位为**科研/质控/预筛工具的技术底座**，而不是诊断产品。

---

## 5. 未来工作

1. **视图感知**：训练 AP/PA/LATERAL 视图分类器，分视图建模或加入视图 embedding。
2. **Study 级多图融合**：用注意力池化/MIL 替代均值，让模型学习同一 Study 内不同视图的权重。
3. **长尾优化**：类别共现建模、focal/ASL loss、部分标签学习、稀有类过采样。
4. **概率校准与拒识**：temperature scaling、ECE/Brier、OOD/非胸片拒识。
5. **外部验证**：多中心、多设备、按科室/体位/年龄分层，报告逐类敏感度/特异度。
6. **可解释性**：Grad-CAM、肺部分割、病灶定位，输出医生可复核的热图。
7. **工程化**：DICOMweb/C-STORE 接入、FastAPI 服务、ONNX/TensorRT、监控与审计。
8. **合规路径**：如转临床产品，按 NMPA 二类/三类医疗器械准备临床评价、ISO 14971、IEC 62304、ISO 13485 与网络安全材料。

---

## 6. 结论

本工作完成了一个合规、可复现、成本可控的胸部 X 光 10 类多标签分类基线：双 ConvNeXt-Tiny 集成在官方验证集上达到图级宏平均 AUC 0.8070、Study 级 0.7994；推理权重 212.4 MB，本机 76 ms/张，提交一致性与格式自检通过。

该成果可以作为论文/技术报告、院内科研与质控工具、以及后续产品化的起点；但在外部验证、校准、逐类临床指标与监管完成之前，不应作为临床诊断依据。

---

## 附录 A：仓库结构

```
cxr-multilabel/
├── README.md
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── requirements.txt
├── common.py                  # 标签解析、manifest、数据集、模型工厂、AUC
├── train.py                   # 训练
├── predict.py                 # 唯一推理实现 + 提交生成
├── model.py                   # 模型定义入口
├── utils.py                   # 数据/预处理入口
├── cost.py                    # 耗时 + 一致性核验
├── data_check.py              # 数据体检
├── validate_submission.py     # 提交格式自检
├── tune_gate.py               # 按类阈值（本次未使用）
├── eval_agg.py                # 聚合方式对比
├── bench.py                   # 训练吞吐测速
├── make_cache.py              # 开发期缓存
├── tools/
│   └── eval_study.py          # Study 级逐类评估
├── docs/
│   ├── TECHNICAL_REPORT.md
│   ├── MODEL_CARD.md
│   └── COMPLIANCE.md
├── results/
│   ├── dataset_stats.json
│   ├── submission_stats.json
│   ├── eval_study.json
│   └── eval_study_per_class.csv
└── licenses/
    └── LICENSE-torchvision.txt
```

## 附录 B：数据与权重声明

- 官方比赛数据不随仓库分发；请从赛事渠道获取，并遵守赛事数据使用协议。
- 推理权重（`*.pt`）不随仓库分发；仓库只提供训练与评估代码。
- 预训练权重来自 torchvision 官方 ImageNet-1K，许可证为 BSD-3-Clause，全文见 `licenses/LICENSE-torchvision.txt`。
- 本仓库代码许可证见 `LICENSE`；代码许可证不覆盖比赛数据与模型权重。
