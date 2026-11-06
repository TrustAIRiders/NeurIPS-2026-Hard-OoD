import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_ood.detector import MaxSoftmax, ReAct, SHE
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models
from tqdm import tqdm

from prepare_data import DATA_ROOT, HERE


BATCH_SIZE = 512
NUM_WORKERS = 32
OUT_DIR = HERE / "results"

MODELS = {
    "ConvNeXt_L": (
        models.convnext_large,
        models.ConvNeXt_Large_Weights.IMAGENET1K_V1,
    ),
    "ResNet152": (
        models.resnet152,
        models.ResNet152_Weights.IMAGENET1K_V1,
    ),
    "ViT_L_16": (
        models.vit_l_16,
        models.ViT_L_16_Weights.IMAGENET1K_V1,
    ),
}
SPLITS = {
    "id_hard": "id_test_hard_list.txt",
    "ood_hard": "ood_hard_list.txt",
}


class ImageList(Dataset):
    def __init__(self, list_path, transform):
        self.transform = transform
        with list_path.open() as handle:
            self.samples = [
                (DATA_ROOT / Path(path).relative_to("/datasets"), int(label))
                for path, label in csv.reader(handle)
            ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with Image.open(path) as image:
            image = self.transform(image.convert("RGB"))
        return image, target


def save_results(rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "auroc.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Batch size: {BATCH_SIZE}", flush=True)

    rows = []
    for model_name, (constructor, weights) in MODELS.items():
        print(f"\nLoading {model_name}...", flush=True)

        transform = weights.transforms()
        loaders = {
            split: DataLoader(
                ImageList(HERE / "benchmark_splits" / filename, transform),
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
            )
            for split, filename in SPLITS.items()
        }

        backbone = constructor(weights=weights).to(device).eval()
        if model_name == "ConvNeXt_L":
            head = backbone.classifier[2]
            backbone.classifier[2] = torch.nn.Identity()
        elif model_name == "ResNet152":
            head = backbone.fc
            backbone.fc = torch.nn.Identity()
        else:
            head = backbone.heads.head
            backbone.heads.head = torch.nn.Identity()

        detectors = {
            "MSP": MaxSoftmax(torch.nn.Sequential(backbone, head).eval()),
            "ReAct": ReAct(backbone, head, threshold=1.0),
            "SHE": SHE(backbone, head),
        }

        for method, detector in detectors.items():
            n_fit = 0

            if method == "SHE":
                print(
                    f"{model_name} / SHE: preparing training data...",
                    flush=True,
                )
                train = datasets.ImageFolder(
                    DATA_ROOT / "ImageNet2012/ILSVRC/Data/CLS-LOC/train",
                    transform=transform,
                )
                train_loader = DataLoader(
                    train,
                    batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS,
                )
                n_fit = len(train)

                with tqdm(
                    train_loader,
                    desc=f"{model_name} | SHE fit",
                    unit="batch",
                    leave=False,
                ) as progress:
                    detector.fit(progress, device=str(device))

                print(
                    f"{model_name} / SHE: fitted on {n_fit:,} images.",
                    flush=True,
                )
                del train_loader, train

            scores = {}
            with tqdm(
                total=sum(len(loader) for loader in loaders.values()),
                desc=f"{model_name} | {method}",
                unit="batch",
                leave=False,
            ) as progress:
                with torch.inference_mode():
                    for split, loader in loaders.items():
                        progress.set_postfix_str(split)
                        batches = []
                        for images, _ in loader:
                            batches.append(detector(images.to(device)).cpu())
                            progress.update(1)
                        scores[split] = torch.cat(batches).numpy()

            id_scores = scores["id_hard"]
            ood_scores = scores["ood_hard"]
            labels = np.r_[
                np.zeros(len(id_scores)),
                np.ones(len(ood_scores)),
            ]
            auroc = float(
                roc_auc_score(labels, np.r_[id_scores, ood_scores])
            )
            rows.append(dict(
                model=model_name,
                detector=method,
                split="hard",
                auroc=auroc,
                n_id=len(id_scores),
                n_ood=len(ood_scores),
                n_fit=n_fit,
            ))

            save_results(rows)
            print(
                f"{model_name:<16} {method:<6} "
                f"Hard: {auroc:.4f}",
                flush=True,
            )

        del detectors, detector, backbone, head

    print("\nFinal AUROC")
    print(f"{'Model':<16} {'Method':<8} {'Hard':>8}")
    print("-" * 34)
    for row in rows:
        print(f"{row['model']:<16} {row['detector']:<8} {row['auroc']:>8.4f}")

    print(f"\nSaved: {OUT_DIR / 'auroc.csv'}")


if __name__ == "__main__":
    main()
