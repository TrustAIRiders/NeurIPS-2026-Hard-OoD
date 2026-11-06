# Quick demo

Evaluate the MSP, ReAct, and SHE detectors from `pytorch-ood` on the released hard split using three pretrained ImageNet-1K classifiers: `ConvNeXt_L`, `ResNet152`, and `ViT_L_16`. The demo reads `id_test_hard_list.txt` and `ood_hard_list.txt` through `quick_demo/benchmark_splits`, a symlink to the [released splits](../benchmark_splits/). You can run it without running the research pipeline.

## Run

From the repository root, use Python 3.12 and the demo's own requirements. `curl` is needed for downloads.

```bash
cd quick_demo
python3 -m pip install -r requirements.txt
python3 prepare_data.py
python3 run_demo.py
```

If you already have the extracted datasets in the layout below, skip `prepare_data.py`. CUDA is used when available, and model weights are downloaded on first use. Reduce `BATCH_SIZE` in `run_demo.py` if the batch does not fit in memory.

## Data

The demo uses ImageNet-1K (train and validation), NINCO, and ImageNet-OOD. Use `prepare_data.py` or symlink existing datasets into this layout under `quick_demo/`:

```text
datasets/
  ImageNet2012/ILSVRC/Data/CLS-LOC/train/   # one directory per ImageNet class
  ImageNet2012/ILSVRC/Data/CLS-LOC/val/     # validation images directly in val/
  NINCO/NINCO/NINCO_OOD_classes/
  imagenetood/
```

ImageNet-1K requires [ImageNet access](https://image-net.org/download.php). The preparation script can also extract archives placed in `datasets/`.

## Results

For each model, the demo evaluates **MSP**, then **ReAct**, then **SHE**. Each detector scores the images independently. MSP and ReAct need no fitting. Before evaluating SHE, the demo makes one pass over the full ImageNet-1K training set to fit its class patterns.

Each AUROC compares 10,000 ID images with 10,000 OOD images from the hard split, with OOD as the positive class. After each model and detector pair, the script prints the hard AUROC. The final table appears in the terminal, and the results are saved to `results/auroc.csv`.

### Example output

| Model | Method | Hard AUROC |
| --- | --- | ---: |
| `ConvNeXt_L` | MSP | 0.1671 |
| `ConvNeXt_L` | ReAct | 0.2826 |
| `ConvNeXt_L` | SHE | 0.3725 |
| `ResNet152` | MSP | 0.2161 |
| `ResNet152` | ReAct | 0.3254 |
| `ResNet152` | SHE | 0.5048 |
| `ViT_L_16` | MSP | 0.1694 |
| `ViT_L_16` | ReAct | 0.1856 |
| `ViT_L_16` | SHE | 0.2726 |
