# Compliance Notes

This document records the design choices that keep the submission reproducible and within the competition rules. It is not a legal opinion.

## 1. Data and pretraining

- Only the official competition train split is used for supervised fine-tuning.
- The official validation split is not merged into training; it is used for model selection and early stopping.
- No external medical dataset, no medical pretraining checkpoint, no self-supervised pretraining on the official train set, and no knowledge distillation.
- The only pretrained weights are torchvision ImageNet-1K weights:
  - file: `convnext_tiny-983f1562.pth`
  - SHA-256: `983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d`
  - license: BSD-3-Clause; full text in `licenses/LICENSE-torchvision.txt`.
- The ImageNet-1K pretraining corpus is general natural images, not medical images.

## 2. Inference determinism

- Fixed preprocessing: Resize 365 → CenterCrop 320 → ImageNet normalization.
- Fixed threshold: 0.5; the `Gate` column is empty.
- No test-time augmentation.
- No test-time statistics, label-dependent post-processing, or parameter fitting.
- Study aggregation: arithmetic mean of image probabilities; deterministic and parameter-free.
- `torch.backends.cudnn.benchmark = False` and `torch.backends.cudnn.deterministic = True` in `load_checkpoint`.
- `predict.py` is the only inference implementation; `cost.py` imports and calls the same functions.

## 3. Submission format

- Long table with exactly 10 rows per Study.
- Columns: `Subject_id, Study_id, Predict_class, Probability, Gate`.
- `Predict_class` uses official ids 1–10, mapped back from internal indices 0–9.
- `Probability` is rounded to 6 decimals and clipped to [0, 1].
- `Gate` is empty for all rows (default 0.5).
- `validate_submission.py` checks column order, class coverage, duplicates, probability range, and Study coverage.

## 4. Resource limits

- Inference weights: 2 × 106.2 MB = 212.4 MB.
- Full extracted package: about 321.5 MB (limit: 500 MB).
- Inference precision: FP16.
- Local timing: 76 ms/image (p95 88 ms) on RTX 4060 Laptop, batch=1, no TTA; official T4 re-test is pending.

## 5. Privacy and data handling

- The competition data is not redistributed in this repository.
- Model weights are not redistributed in this repository.
- Do not commit patient data, DICOM files, submission files containing real identifiers, or local absolute paths to a public repository.
- For any hospital deployment, apply de-identification, access control, audit logging, encryption, and data-retention policies consistent with local law and institutional review.

## 6. What this repository does not establish

- It does not establish clinical safety or effectiveness.
- It does not establish regulatory clearance or approval.
- It does not establish generalization to other hospitals, scanners, populations, or label definitions.
- It does not establish that AUC 0.80 is sufficient for any clinical or screening use.

## 7. Third-party licenses

- torchvision / PyTorch: BSD-style licenses; see the upstream projects.
- ConvNeXt-Tiny ImageNet-1K weights: torchvision, BSD-3-Clause; full text in `licenses/LICENSE-torchvision.txt`.
- The repository's own code is released under the MIT license in `LICENSE`.

## 8. Data-use reminder

Before any commercial or clinical use, confirm in writing:

1. whether the competition data license permits commercial use and redistribution;
2. whether models trained on the competition data may be used or distributed commercially;
3. whether additional ethics approval, data-processing agreements, or regulatory filings are required.
