# OoD Hard Dataset Pipeline

Minimal pipeline for computing out-of-distribution (OoD) scores, estimating shared sample difficulty, selecting hard ImageNet-based subsets, and evaluating OoD detection on those subsets.

## Setup

Download all datasets first and place them under: `/datasets/` 

Then install Python dependencies: `pip install -r requirements.txt`

If your datasets are stored elsewhere, update `DATASET_PATH` in the scripts.

The main memory-related settings are defined near the top of the scripts:
```python
BATCH_SIZE = 64
NUM_WORKERS = 40
```

## Scripts

Run the scripts in order:
- `python A_get_ood_scores_pytorch_ood.py` computes OoD scores and saves them to `./scores`.
- `python B_generate_difficulty_heatmaps.py` reads `./scores` and saves shared-difficulty analyses to `./shared_analysis`.
- `python C_generate_hard_dataset.py` builds hard and easy ID/OOD lists and saves them to `./ImageNet_dataset`.
- `python D_ood_detection.py` evaluates OoD detection on hard subsets and saves results to `./analyze_dataset/ood_hard`.
- `python E_cifar10_ood_minimal.py` runs the CIFAR-10/CIFAR-100/SVHN experiment and saves results to `./cifar10_ood_analysis`.

## Hard dataset
The generated hard/easy lists are saved in `./ImageNet_dataset/` and use the format:

```text
image_path,label
```

OOD samples are saved with label `-1`.

## Datasets

Used datasets:
- ImageNet-1K - Deng et al., 2009. ImageNet: A Large-Scale Hierarchical Image Database.
- ImageNet-O - Hendrycks et al., 2021. Natural Adversarial Examples.
- Describable Textures Dataset, DTD - Cimpoi et al., 2014. Describing Textures in the Wild.
- SUN2012 - Xiao et al., 2010. SUN Database: Large-Scale Scene Recognition from Abbey to Zoo.
- iNaturalist - Van Horn et al., 2018. The iNaturalist Species Classification and Detection Dataset.
- Places365 - Zhou et al., 2018. Places: A 10 Million Image Database for Scene Recognition.
- NINCO - Bitterwolf et al., 2023. In or Out? Fixing ImageNet Out-of-Distribution Detection Evaluation.
- ImageNet-OOD - Yang et al., 2024. ImageNet-OOD: Deciphering Modern Out-of-Distribution Detection Algorithms.

Expected dataset paths:

```text
/datasets/ImageNet2012
/datasets/imagenet-o
/datasets/textures/images
/datasets/SUN2012/Images
/datasets/iNaturalist
/datasets/places365
/datasets/NINCO/NINCO/NINCO_OOD_classes
/datasets/imagenetood
```