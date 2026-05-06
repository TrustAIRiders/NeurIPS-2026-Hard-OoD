# OoD Hard Dataset Pipeline

This repository contains a minimal pipeline for generating out-of-distribution (OoD) scores, estimating shared sample difficulty across models and detectors, selecting hard ImageNet-based subsets, and evaluating OoD detection on those subsets.

Download all datasets first and place them under: `/datasets/`

- `A_get_scores.py` computes OoD scores for all benchmarks and saves detector outputs to `./scores`.

- `B_shered_analysis.py` reads `./scores` and saves shared-difficulty analyses to `./shared_analysis`, including AUROC values and per-dataset Spearman, heatmap, and 10-bin agreement p-value figures.

- `C_generate_hard_dataset.py` uses the shared-difficulty scores to construct hard ID/OOD image lists and saves them to `./ImageNet_dataset`.

- `D_ood_detection.py` evaluates OoD detection on the hard subsets using new models. It reads lists from `./ImageNet_dataset`, reads image data from `/datasets/`, and saves results to `./ood_on_hard/`.

- `E_cifar10.py` runs the CIFAR-10/CIFAR-100/SVHN experiment with the same general metodology `./cifar10/`.
