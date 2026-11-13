# Model Card — Chest X-ray Multi-label Classifier

## Model details

| Item | Value |
|---|---|
| Architecture | ConvNeXt-Tiny ×2 (torchvision) |
| Input | 320 × 320 RGB |
| Output | 10 logits → sigmoid probabilities |
| Pretraining | ImageNet-1K (`ConvNeXt_Tiny_Weights.IMAGENET1K_V1`) |
| Training data | Official competition train split only (18,513 studies / 30,344 images) |
| Ensemble | Equal-weight probability average of seed 42 and seed 123 checkpoints |
| Aggregation | Arithmetic mean over images of the same Study |
| Threshold | 0.5 (Gate column empty) |
| Precision | FP16 inference |
| Weight size | 212.4 MB total |
| License | Code: MIT; weights: not distributed; data: competition terms |

## Intended use

- Research and education on chest X-ray multi-label classification.
- Benchmarking, ablation studies, and reproducibility checks.
- A technical foundation for a research/quality-control/pre-screening tool after local validation.
- Offline batch inference on de-identified images.

## Out-of-scope use

- **Not a medical device.** Do not use for diagnosis, triage, treatment decisions, or report signing.
- Do not use on non-chest radiographs, CT, MRI, ultrasound, or photographs.
- Do not use as the sole basis for any clinical decision.
- Do not deploy on patient data without institutional review, privacy review, and clinical validation.
- Do not claim diagnostic performance from the validation numbers in this repository.

## Training data

- Official competition train split: 18,513 studies, 11,055 subjects, 30,344 images.
- Labels: 10-class multi-label; 62.6% of studies have one label, 37.4% have two or more.
- View distribution (images with a recorded view): AP 12,152; PA 7,604; LATERAL 6,554; LL 2,831; PA LLD 1. Another 1,202 images have a missing or other `ViewPosition` value.
- No external medical data, no medical pretraining, no self-supervised pretraining, no distillation.
- The official validation split was not used for training; it was used for model selection.

## Evaluation

Validation split: 859 studies / 1,422 images / 228 subjects; subject-disjoint from train and test.

| Metric | Value |
|---|---:|
| Single model r4 (seed 42), image-level macro AUC | 0.8004 |
| Single model r6 (seed 123), image-level macro AUC | 0.8013 |
| Ensemble, image-level macro AUC | 0.8070 |
| Ensemble, study-level macro AUC | 0.7994 |
| Ensemble, study-level macro AUC (max aggregation) | 0.7939 |
| Ensemble, study-level macro AUC (meanmax aggregation) | 0.7985 |
| Ensemble, study-level macro AUC (noisy-or aggregation) | 0.7695 |

Per-class study-level AUC ranges from 0.6786 (class 1, 24 positives) to 0.9062 (class 9, 260 positives). See `results/eval_study_per_class.csv`.

## Limitations

- Single-center, single-source data; no external validation.
- Small validation set; rare classes have very few positives, so per-class metrics are unstable.
- Weak image/study-level labels; label noise and co-occurrence bias are possible.
- No probability calibration; threshold 0.5 is not a clinical operating point.
- No sensitivity/specificity/PPV/NPV, decision-curve analysis, or reader study.
- Input is competition JPEG, not raw DICOM; windowing, rescale slope/intercept, and device differences are not handled.
- No lesion localization or heatmaps.
- View position is not used as a model input, although ~31% of training images are lateral.
- Competition rules prohibit medical pretraining and external data; performance is therefore not representative of a product built with broader data.

## Ethical and regulatory notes

- The model must not be presented as a diagnosis or as a replacement for radiologist review.
- Any clinical use requires institutional review, privacy/security review, clinical validation, and the applicable medical-device regulatory path.
- Do not attempt to re-identify individuals from predictions or intermediate features.
- Report performance with the population, scanner, view, and label definitions used; do not generalize the numbers to other settings.

## Citation

If you use this code or results, cite the repository and the technical report:

```
Lin Hongyi. Chest X-ray Multi-label Classification: Technical Report and Open Baseline.
NCCCU AI Challenge 2026. 2026.
```

## Contact

Open an issue in the repository for questions about reproducibility.
