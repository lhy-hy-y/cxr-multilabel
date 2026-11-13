# Chest X-ray Multi-label Classification

A reproducible, competition-compliant baseline for 10-class chest X-ray multi-label classification, with study-level submission, cost benchmarking, and per-class evaluation.

> **Not a medical device.** This repository is for research, education, and benchmarking. It must not be used for diagnosis, triage, treatment decisions, or report signing.

## Results at a glance

Validation split: 859 studies / 1,422 images / 228 subjects (subject-disjoint from train and test).

| Model | Image-level macro AUC | Study-level macro AUC |
|---|---:|---:|
| ConvNeXt-Tiny, seed 42 | 0.8004 | — |
| ConvNeXt-Tiny, seed 123 | 0.8013 | — |
| Equal-weight ensemble | **0.8070** | **0.7994** |

- Study aggregation: arithmetic mean over images; mean > meanmax > max > noisy-or on this split.
- Per-class study AUC: 0.6786 (class 1, 24 positives) to 0.9062 (class 9, 260 positives).
- Inference weights: 212.4 MB; FP16; no TTA.
- Local cost: 76 ms/image (p95 88 ms) on RTX 4060 Laptop, batch=1.
- Submission self-check: 14,140 rows = 1,414 studies × 10 classes; 100/100 consistency check passed.

Full details: [Technical report](docs/TECHNICAL_REPORT.md) · [Model card](docs/MODEL_CARD.md) · [Compliance notes](docs/COMPLIANCE.md)

## What is in this repository

- `common.py` — label parsing, manifest building, dataset, transforms, model factory, macro AUC.
- `train.py` — BCE + positive weighting, AdamW, cosine schedule, AMP, EMA, validation-based checkpoint selection.
- `predict.py` — the single inference implementation; ensemble + study aggregation + submission writer.
- `cost.py` — timing and consistency check against `submission.csv` using the same inference functions.
- `data_check.py` — scans images/labels and writes manifests.
- `validate_submission.py` — submission format self-check.
- `eval_agg.py` — compares study-level aggregations.
- `tools/eval_study.py` — image-level and study-level evaluation with per-class AUC/AUPRC.
- `results/` — dataset statistics and evaluation outputs from the validation split.
- `docs/` — technical report, model card, compliance notes.

## What is not included

- **Competition data.** Download it from the official channel and follow its data-use terms.
- **Model weights.** The two `*.pt` checkpoints are not distributed here. Train them with the commands below, or place them under `model/`.
- **Submission file.** Generate it with `predict.py`.

## Repository layout

```
.
├── README.md
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── requirements.txt
├── common.py
├── train.py
├── predict.py
├── model.py
├── utils.py
├── cost.py
├── data_check.py
├── validate_submission.py
├── tune_gate.py
├── eval_agg.py
├── bench.py
├── make_cache.py
├── tools/
│   └── eval_study.py
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

## Quick start

### 1. Environment

The official competition image already provides PyTorch + torchvision. For local development:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` intentionally does not pin `torch` / `torchvision`, so the official image is not downgraded. Install the CUDA build that matches your driver.

### 2. Data layout

```text
data/
├── train/
│   └── subject_XXXXX/
│       └── study_YYYYY/
│           └── pic_ZZZZZZ.jpg
├── valid/
├── test/
├── train.csv
├── valid.csv
└── test.csv
```

Labels may be a long table (one `Predict_class` per row) or a wide table (one column per class).

### 3. Build manifests

```bash
python data_check.py --images data/train --labels data/train.csv --out-dir runs/data_check/train
python data_check.py --images data/valid --labels data/valid.csv --out-dir runs/data_check/valid
python data_check.py --images data/test  --out-dir runs/data_check/test
```

Each run writes `manifest.csv`, `class_ids.json`, `study_ids.txt`, label distribution, co-occurrence, and `data_check.json`.

### 4. Train

```bash
python train.py \
  --train-manifest runs/data_check/train/manifest.csv \
  --val-manifest   runs/data_check/valid/manifest.csv \
  --backbone convnext_tiny --img-size 320 \
  --epochs 8 --batch-size 16 --seed 42 \
  --out-dir runs/r4_convnext_320

python train.py \
  --train-manifest runs/data_check/train/manifest.csv \
  --val-manifest   runs/data_check/valid/manifest.csv \
  --backbone convnext_tiny --img-size 320 \
  --epochs 8 --batch-size 16 --seed 123 \
  --out-dir runs/r6_convnext_320
```

The best checkpoint is selected by validation macro AUC (raw and EMA are both evaluated). Resources: `best.pt`, `log.csv`, `summary.json`.

### 5. Predict

```bash
python predict.py \
  --ckpt runs/r4_convnext_320/best.pt runs/r6_convnext_320/best.pt \
  --manifest runs/data_check/test/manifest.csv \
  --out submission.csv --weights 0.5,0.5
```

- The model output is averaged over the ensemble.
- Images of the same Study are averaged.
- Threshold is 0.5; `Gate` is left empty.
- No TTA.

### 6. Validate and benchmark

```bash
python validate_submission.py --csv submission.csv --ids runs/data_check/test/study_ids.txt

python cost.py submission.csv data/test <pictures_100> <pictures_5> \
  --ckpt runs/r4_convnext_320/best.pt runs/r6_convnext_320/best.pt \
  --manifest runs/data_check/test/manifest.csv
```

`cost.py` checks binary decisions and probability ranking against `submission.csv`, and reports average latency. It imports the same `predict.py` functions used to generate the submission.

### 7. Study-level evaluation

```bash
python tools/eval_study.py \
  --ckpt runs/r4_convnext_320/best.pt runs/r6_convnext_320/best.pt \
  --manifest runs/data_check/valid/manifest.csv \
  --out-json runs/eval_study.json \
  --out-csv  runs/eval_study_per_class.csv
```

This reports image-level and study-level macro AUC, per-class AUC/AUPRC, and a comparison of mean/max/meanmax/noisy-or aggregation.

## Reproducibility notes

- Seeds: 42 and 123.
- The official validation split is never merged into training.
- The submission uses equal-weight ensemble averaging, study-level mean, threshold 0.5, and no TTA.
- Full training logs were not preserved in the submission package; default hyperparameters are documented in `train.py` and the technical report.
- Validation metrics in `results/` were recomputed from the official split and the two checkpoints.

## Limitations

- Single-center data; no external validation.
- Small validation set; rare classes have very few positives.
- Weak image/study labels; possible label noise.
- No probability calibration, no clinical operating points, no reader study.
- Competition JPEG input, not raw DICOM.
- No view conditioning, although ~31% of training images are lateral.
- Not a medical device and not clinically validated.

See [docs/MODEL_CARD.md](docs/MODEL_CARD.md) and [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md) for details.

## License

- Code: MIT, see [LICENSE](LICENSE).
- torchvision ImageNet-1K weights: BSD-3-Clause, see [licenses/LICENSE-torchvision.txt](licenses/LICENSE-torchvision.txt).
- Competition data and trained weights are not distributed here and remain subject to their original terms.
- Third-party notices: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@misc{lin2026cxrmultilabel,
  title  = {Chest X-ray Multi-label Classification: Technical Report and Open Baseline},
  author = {Lin, Hongyi},
  year   = {2026},
  note   = {NCCCU AI Challenge 2026}
}
```
