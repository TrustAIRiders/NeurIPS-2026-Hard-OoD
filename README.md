# Hard OOD Benchmark

This repository contains the code and benchmark splits accompanying the paper:

**Surprisingly Consistent Failures of Post-Hoc OoD Detectors Reveal Security Concerns in Open-Set Recognition With Modern CNNs and ViTs, Stemming From Representation Mismatch**

**Kamil Szyc · Tomasz Walkowiak · Henryk Maciejewski** -- Wrocław University of Science and Technology

**[NeurIPS 2026, Evaluations & Datasets Track](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets)**

## Start here

The benchmark provides easy and hard splits for testing how well OOD detectors separate ImageNet-1K images from images outside its label space. **[Run the quick demo](quick_demo/README.md)** to evaluate MSP, ReAct, and SHE on the complete hard split with `ConvNeXt_L`, `ResNet152`, and `ViT_L_16`. The demo includes data preparation instructions and example AUROCs. It does not require regenerating the benchmark.

### Use the benchmark with your own detector

Obtain the ImageNet-1K validation, NINCO, and ImageNet-OOD images. Map the `/datasets/` prefix in each [split list](benchmark_splits/) to your local dataset root. Each line contains `image_path,label`. ID labels are ImageNet class indices; OOD labels are `-1`.

| Evaluation | ID list (ImageNet-1K validation) | OOD list (NINCO and ImageNet-OOD) |
| --- | --- | --- |
| Easy | [id_test_easy_list.txt](benchmark_splits/id_test_easy_list.txt) | [ood_easy_list.txt](benchmark_splits/ood_easy_list.txt) |
| Hard | [id_test_hard_list.txt](benchmark_splits/id_test_hard_list.txt) | [ood_hard_list.txt](benchmark_splits/ood_hard_list.txt) |

Score the images with your detector and compute one AUROC per row, using all 10,000 ID and 10,000 OOD images. Treat OOD as positive and orient the score so that larger values indicate stronger OOD evidence.

The split lists are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Images are obtained separately from the source datasets. See [croissant.json](benchmark_splits/croissant.json) for dataset metadata.

## Research pipeline

Run the scripts from the repository root with the main requirements:

```bash
python3 -m pip install -r requirements.txt
```

The default dataset root is `/datasets/`. Batch size, worker count, and model settings are defined in the relevant scripts. The ImageNet loaders in A and D also expect `imagenet_class_index.json` and `ILSVRC2012_val_labels.json` under `/datasets/ImageNet2012/`. F uses `imagenet_class_index.json` for the training centroids.

| Script | Purpose | Output |
| --- | --- | --- |
| [A_get_scores.py](A_get_scores.py) | Compute per-image OOD scores for ImageNet-1K and the OOD datasets. | `scores/` |
| [B_shered_analysis.py](B_shered_analysis.py) | Analyze agreement in sample difficulty using scores from A. | `shared_analysis/` |
| [C_generate_hard_dataset.py](C_generate_hard_dataset.py) | Regenerate the four split lists from scores produced by A. Running C writes to the same paths as the released lists. | `benchmark_splits/` |
| [D_test_ood_hard.py](D_test_ood_hard.py) | Evaluate additional architectures on the hard split (requires CUDA). | `ood_on_hard/` |
| [E_cifar10.py](E_cifar10.py) | Run the separate CIFAR-10 experiment with CIFAR-100 and SVHN as OOD datasets. | `cifar10/` |
| [F_get_clip_coca_features.py](F_get_clip_coca_features.py) | Extract frozen CLIP and CoCa features for the released splits and compute ImageNet training class centroids. | `clip_coca_features/` |
| [G_test_clip_coca_geometry.py](G_test_clip_coca_geometry.py) | Use the features and centroids from F to compute easy and hard AUROCs and bootstrap confidence intervals. | `clip_coca_geometry/` |
| [H_leave_one_family_out.py](H_leave_one_family_out.py) | Use scores from A to select samples with two detector families and evaluate the excluded family on held-out models. | `leave_one_family_out/` |

## Datasets

The full pipeline uses the datasets below. Paths are relative to `/datasets/`.

| Dataset | Reference | Path |
| --- | --- | --- |
| ImageNet-1K | Deng et al., 2009 | `ImageNet2012/` |
| ImageNet-O | Hendrycks et al., 2021 | `imagenet-o/` |
| DTD | Cimpoi et al., 2014 | `textures/images/` |
| SUN2012 | Xiao et al., 2010 | `SUN2012/Images/` |
| iNaturalist | Van Horn et al., 2018 | `iNaturalist/` |
| Places365 | Zhou et al., 2018 | `places365/` |
| NINCO | Bitterwolf et al., 2023 | `NINCO/NINCO/NINCO_OOD_classes/` |
| ImageNet-OOD | Yang et al., 2024 | `imagenetood/` |
