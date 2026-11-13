# Third-party notices

This repository uses the following third-party software and weights. It does
not redistribute the competition dataset or the trained checkpoints.

## PyTorch / torchvision

- Project: PyTorch, torchvision
- License: BSD-style (see upstream repositories)
- Usage: model definition, training, inference, image transforms

## ConvNeXt-Tiny ImageNet-1K weights

- File: `convnext_tiny-983f1562.pth`
- Source: torchvision official weights
- URL: https://download.pytorch.org/models/convnext_tiny-983f1562.pth
- SHA-256: `983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d`
- License: BSD-3-Clause
- Full text: `licenses/LICENSE-torchvision.txt`
- Pretraining corpus: ImageNet-1K (general natural images, no medical images)

## ConvNeXt paper

- Title: A ConvNet for the 2020s
- Authors: Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer,
  Trevor Darrell, Saining Xie
- Usage: architecture reference

## scikit-learn

- License: BSD-3-Clause
- Usage: ROC AUC and average precision in evaluation scripts

## Notes

- The competition dataset is not included. Its use is governed by the
  competition's data-use terms.
- The trained checkpoints are not included. They were trained on the
  competition data and may be subject to the same terms.
- The MIT license in `LICENSE` applies only to the code in this repository.
