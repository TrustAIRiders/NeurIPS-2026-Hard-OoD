import hashlib
import json
import os
import random
import sys
import unicodedata
from importlib.metadata import version
from pathlib import Path

NUM_WORKERS = 8
os.environ["MKL_NUM_THREADS"] = str(NUM_WORKERS)
os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_WORKERS)
os.environ["OMP_NUM_THREADS"] = str(NUM_WORKERS)

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import torch
from torch.utils.data import DataLoader, Dataset
import open_clip


ImageFile.LOAD_TRUNCATED_IMAGES = True

SEED = 0
BATCH_SIZE = 32
DATASET_PATH = "/datasets/"  # Set to the directory containing the dataset folders
LISTS_DIR = Path("./benchmark_splits")
OUT_DIR = Path("./clip_coca_features")

MODEL_CONFIGS = {
    "CLIP_ViT_H_14": ("ViT-H-14", "laion2b_s32b_b79k"),
    "CoCa_ViT_L_14": ("coca_ViT-L-14", "laion2b_s13b_b90k"),
}
SPLIT_FILES = {
    "id_test_hard": LISTS_DIR / "id_test_hard_list.txt",
    "id_test_easy": LISTS_DIR / "id_test_easy_list.txt",
    "ood_hard": LISTS_DIR / "ood_hard_list.txt",
    "ood_easy": LISTS_DIR / "ood_easy_list.txt",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")


def normalize_rows(values):
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms <= 1e-12):
        raise ValueError("Image features and centroids must be finite and nonzero.")
    return values / norms


def canonical_filename(path):
    filename = os.path.relpath(path, DATASET_PATH).replace(os.sep, "/")
    return unicodedata.normalize("NFC", filename)


def resolve_list_path(value):
    text = str(value).replace("\\", "/")
    if text.startswith("/") and "/datasets/" in text:
        text = text.split("/datasets/", 1)[1]
    path = Path(DATASET_PATH) / text
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    return str(path)


class ImageListDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as image:
            image = self.transform(image.convert("RGB"))
        return image, self.labels[idx], self.paths[idx]


def load_train_samples():
    root = Path(DATASET_PATH) / "ImageNet2012"
    with (root / "imagenet_class_index.json").open() as handle:
        mapping = json.load(handle)
    class_count = len(mapping)
    if not mapping or set(mapping) != {str(i) for i in range(class_count)}:
        raise ValueError("Expected contiguous class labels starting at zero.")
    paths, labels = [], []
    for class_id in range(class_count):
        directory = root / "ILSVRC/Data/CLS-LOC/train" / mapping[str(class_id)][0]
        images = sorted(path for path in directory.iterdir() if path.is_file())
        if not images:
            raise ValueError(f"Missing training images: {directory}")
        paths.extend(str(path) for path in images)
        labels.extend([class_id] * len(images))
    return paths, labels


def load_split_samples(list_path):
    frame = pd.read_csv(list_path, header=None, names=["image_path", "label"])
    paths = [resolve_list_path(value) for value in frame["image_path"]]
    labels = pd.to_numeric(frame["label"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
        raise ValueError(f"Expected integer labels: {list_path}")
    if len(set(paths)) != len(paths):
        raise ValueError(f"Duplicate images: {list_path}")
    return paths, labels.astype(int).tolist()


def make_loader(samples, preprocess):
    return DataLoader(
        ImageListDataset(*samples, transform=preprocess),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
    )


def iter_features(loader, model, device):
    with torch.inference_mode():
        for batch_idx, (images, labels, paths) in enumerate(loader):
            sys.stdout.write(f"{batch_idx + 1}/{len(loader)}\r")
            sys.stdout.flush()
            features = model.encode_image(images.to(device, non_blocking=True))
            yield features.float().cpu().numpy(), labels.numpy(), paths
    sys.stdout.write("\n")


def build_train_centroids(feature_batches, class_count):
    sums = None
    counts = np.zeros(class_count, dtype=np.int64)
    for features, labels, _ in feature_batches:
        features = normalize_rows(features)
        if sums is None:
            sums = np.zeros((class_count, features.shape[1]), dtype=np.float64)
        np.add.at(sums, labels, features)
        np.add.at(counts, labels, 1)
    if np.any(counts == 0):
        raise ValueError(f"Missing ImageNet classes: {np.flatnonzero(counts == 0).tolist()}")
    centroids = normalize_rows(sums / counts[:, None])
    return pd.DataFrame({
        "label": np.arange(class_count),
        "features": list(centroids),
        "count": counts,
    })


def collect_features(feature_batches):
    rows = []
    for features, labels, paths in feature_batches:
        for path, label, feature in zip(paths, labels, features):
            filename = canonical_filename(path)
            rows.append({
                "filename": filename,
                "id": hashlib.sha1(filename.encode("utf-8")).hexdigest(),
                "label": int(label),
                "features": feature,
            })
    return pd.DataFrame(rows).sort_values("filename").reset_index(drop=True)


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model_name in MODEL_CONFIGS:
        if (OUT_DIR / model_name).exists():
            raise FileExistsError(f"Output already exists: {OUT_DIR / model_name}. Use a new OUT_DIR for a new run.")
    train_samples = load_train_samples()
    class_count = len(set(train_samples[1]))
    split_samples = {name: load_split_samples(path) for name, path in SPLIT_FILES.items()}
    all_paths = [path for paths, _ in split_samples.values() for path in paths]
    if len(set(all_paths)) != len(all_paths) or set(all_paths).intersection(train_samples[0]):
        raise ValueError("Training and evaluation lists must be disjoint, and evaluation splits must not overlap.")
    for name, (_, labels) in split_samples.items():
        if name.startswith("ood"):
            valid = all(label == -1 for label in labels)
        else:
            valid = all(0 <= label < class_count for label in labels)
        if not valid:
            raise ValueError(f"Unexpected labels in split: {name}")
    split_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in SPLIT_FILES.items()}

    for model_name, (architecture, checkpoint) in MODEL_CONFIGS.items():
        print("Processing model:", model_name)
        model, _, preprocess = open_clip.create_model_and_transforms(architecture, pretrained=checkpoint)
        model = model.eval().to(device)
        model.requires_grad_(False)
        model_out = OUT_DIR / model_name
        model_out.mkdir(parents=True, exist_ok=False)

        print("\tProcessing ImageNet training centroids")
        loader = make_loader(train_samples, preprocess)
        centroids = build_train_centroids(iter_features(loader, model, device), class_count)
        centroids.to_pickle(model_out / "train_centroids.pickle")
        del loader

        for split_name, samples in split_samples.items():
            print("\tProcessing split:", split_name)
            loader = make_loader(samples, preprocess)
            frame = collect_features(iter_features(loader, model, device))
            frame.to_pickle(model_out / f"{split_name}.pickle")
            del loader

        config = {
            "model_name": model_name,
            "architecture": architecture,
            "checkpoint": checkpoint,
            "preprocess": repr(preprocess),
            "dataset_path": DATASET_PATH,
            "train_sample_count": len(train_samples[0]),
            "split_sha256": split_hashes,
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "device": str(device),
            "precision": "fp32",
            "float32_matmul_precision": "high",
            "versions": {name: version(name) for name in ["torch", "torchvision", "open_clip_torch", "numpy", "pandas", "Pillow"]},
        }
        (model_out / "run_config.json").write_text(json.dumps(config, indent=2))
        print("\tSaved:", model_out)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
