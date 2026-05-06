import hashlib
import json
import os
import random
import unicodedata
from pathlib import Path

NUM_WORKERS = 40
os.environ["MKL_NUM_THREADS"] = str(NUM_WORKERS)
os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_WORKERS)
os.environ["OMP_NUM_THREADS"] = str(NUM_WORKERS)

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from pytorch_ood.detector import (
    DICE,
    EnergyBased,
    Entropy,
    KLMatching,
    KNN,
    Mahalanobis,
    MaxLogit,
    MaxSoftmax,
    ODIN,
    RMD,
    ReAct,
    TemperatureScaling,
    ViM,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

SEED = 0
BATCH_SIZE = 64
OUT_DIR = "./scores"
IMAGENET_STD_FALLBACK = [0.229, 0.224, 0.225]
VIM_SUBSPACE_RATIO = 0.50

DATASET_PATH = "/datasets/"  # Set to the directory containing the dataset folders
DATASET_IMAGE_NET_2012_PATH = os.path.join(DATASET_PATH, "ImageNet2012")
DATASET_IMAGE_NET_O_PATH = os.path.join(DATASET_PATH, "imagenet-o")
DATASET_TEXTURES_PATH = os.path.join(DATASET_PATH, "textures/images")
DATASET_SUN2012_PATH = os.path.join(DATASET_PATH, "SUN2012/Images")
DATASET_INATURALIST_PATH = os.path.join(DATASET_PATH, "iNaturalist")
DATASET_PLACES_PATH = os.path.join(DATASET_PATH, "places365")
DATASET_NINCO_PATH = os.path.join(DATASET_PATH, "NINCO/NINCO/NINCO_OOD_classes")
DATASET_IMAGENETOOD_PATH = os.path.join(DATASET_PATH, "imagenetood")

DEFAULT_MODEL_NAMES = [
    "ResNet152",
    "EfficientNet_B3",
    "ConvNeXt_L",
    "ViT_L_16",
    "Swin_V2_B",
    "MaxVit_T",
    "EfficientNet_V2_M",
    "MobileNet_v3_L",
    "RegNet_Y_800MF",
]

DEFAULT_DATASET_NAMES = [
    "test",
    "imagenet_o",
    "textures",
    "sun2012",
    "iNaturalist",
    "places365",
    "ninco",
    "imagenetood",
]


np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_float32_matmul_precision("high")



def normalize_path(path):
    path = os.path.abspath(path).replace(os.sep, "/")
    return unicodedata.normalize("NFC", path)


def stable_image_id(path):
    return hashlib.sha1(normalize_path(path).encode("utf-8")).hexdigest()


class SortedImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.samples = sorted(self.samples, key=lambda item: normalize_path(item[0]))
        self.imgs = self.samples

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target, path


class ImageNetKaggleWithPaths(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform

        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as handle:
            syn_to_class = {value[0]: int(class_id) for class_id, value in json.load(handle).items()}

        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as handle:
            val_to_syn = json.load(handle)

        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)

        if split == "train":
            for syn_id in sorted(os.listdir(samples_dir)):
                syn_folder = os.path.join(samples_dir, syn_id)
                if not os.path.isdir(syn_folder):
                    continue
                for filename in sorted(os.listdir(syn_folder)):
                    path = os.path.join(syn_folder, filename)
                    if os.path.isfile(path):
                        self.samples.append(path)
                        self.targets.append(syn_to_class[syn_id])
            return

        for filename in sorted(os.listdir(samples_dir)):
            path = os.path.join(samples_dir, filename)
            if os.path.isfile(path):
                self.samples.append(path)
                self.targets.append(syn_to_class[val_to_syn[filename]])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[idx], path


class ImageNetOODWithPaths(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        entries = []

        for filename in sorted(os.listdir(root)):
            path = os.path.join(root, filename)
            if os.path.isfile(path):
                entries.append((path, filename.split("_", 1)[0]))

        synsets = sorted({syn_id for _, syn_id in entries})
        syn_to_class = {syn_id: idx for idx, syn_id in enumerate(synsets)}

        self.samples = [path for path, _ in entries]
        self.targets = [syn_to_class[syn_id] for _, syn_id in entries]
        self.synsets = synsets
        self.syn_to_class = syn_to_class

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[idx], path


class DatasetSpec:
    def __init__(self, name, dataset):
        self.name = name
        self.dataset = dataset


class DetectorSpec:
    def __init__(self, name, build, fit_loader_name=None, needs_grad=False):
        self.name = name
        self.build = build
        self.fit_loader_name = fit_loader_name
        self.needs_grad = needs_grad


class ModelBundle:
    def __init__(self, name, model, preprocess, backbone, head, linear_head):
        self.name = name
        self.model = model
        self.preprocess = preprocess
        self.backbone = backbone
        self.head = head
        self.linear_head = linear_head


class ResNetBackbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        return torch.flatten(x, 1)


class EfficientNetBackbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        return torch.flatten(x, 1)


class ConvNeXtBackbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = self.model.classifier[0](x)
        x = self.model.classifier[1](x)
        return x


class ViTBackbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model._process_input(x)
        batch_size = x.shape[0]
        cls = self.model.class_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.model.encoder(x)[:, 0]
        if hasattr(self.model.heads, "pre_logits"):
            x = self.model.heads.pre_logits(x)
        if hasattr(self.model.heads, "act"):
            x = self.model.heads.act(x)
        return x


class SwinBackbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.norm(x)
        x = self.model.permute(x)
        x = self.model.avgpool(x)
        return self.model.flatten(x)


class MaxVitBackbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model.stem(x)
        for block in self.model.blocks:
            x = block(x)
        x = self.model.classifier[0](x)
        x = self.model.classifier[1](x)
        x = self.model.classifier[2](x)
        x = self.model.classifier[3](x)
        x = self.model.classifier[4](x)
        return x


class MobileNetV3LargeBackbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.classifier[0](x)
        x = self.model.classifier[1](x)
        x = self.model.classifier[2](x)
        return x


class RegNetBackbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model.stem(x)
        x = self.model.trunk_output(x)
        x = self.model.avgpool(x)
        return torch.flatten(x, 1)


class SequentialHead(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)


class LinearHead(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return self.layer(x)


def collate_xy_only(batch):
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]
    return default_collate(xs), default_collate(ys)


def make_loader(dataset, batch_size, num_workers, xy_only=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=collate_xy_only if xy_only else None,
    )


def save_scores_df(df, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out_path)


def extract_norm_std(preprocess):
    std = getattr(preprocess, "std", None)
    if std is not None:
        return list(std)

    for transform in getattr(preprocess, "transforms", []) or []:
        std = getattr(transform, "std", None)
        if std is not None:
            return list(std)

    return IMAGENET_STD_FALLBACK



def vim_subspace_dim(feature_dim):
    dim = int(round(feature_dim * VIM_SUBSPACE_RATIO))
    return min(max(64, dim), feature_dim - 1)


def rows_from_batch(paths, targets, scores):
    return [
        {
            "image_id": stable_image_id(path),
            "image_path": normalize_path(path),
            "ood_score": float(score),
            "true_label": int(target),
        }
        for path, target, score in zip(paths, targets, scores)
    ]


def finalize_scores_df(rows):
    df = pd.DataFrame(rows, columns=["image_id", "image_path", "ood_score", "true_label"])
    df = df.sort_values("image_path").reset_index(drop=True)
    df.insert(0, "row_id", np.arange(1, len(df) + 1, dtype=np.int64))
    return df


def build_resnet152(device):
    weights = torchvision.models.ResNet152_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet152(weights=weights).eval().to(device)
    backbone = ResNetBackbone(model).eval().to(device)
    head = LinearHead(model.fc).eval().to(device)
    return ModelBundle("ResNet152", model, weights.transforms(), backbone, head, model.fc)


def build_efficientnet_b3(device):
    weights = torchvision.models.EfficientNet_B3_Weights.IMAGENET1K_V1
    model = torchvision.models.efficientnet_b3(weights=weights).eval().to(device)
    backbone = EfficientNetBackbone(model).eval().to(device)
    head = SequentialHead(model.classifier).eval().to(device)
    return ModelBundle("EfficientNet_B3", model, weights.transforms(), backbone, head, model.classifier[1])


def build_convnext_l(device):
    weights = torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1
    model = torchvision.models.convnext_large(weights=weights).eval().to(device)
    backbone = ConvNeXtBackbone(model).eval().to(device)
    head = LinearHead(model.classifier[2]).eval().to(device)
    return ModelBundle("ConvNeXt_L", model, weights.transforms(), backbone, head, model.classifier[2])


def build_vit_l_16(device):
    weights = torchvision.models.ViT_L_16_Weights.IMAGENET1K_V1
    model = torchvision.models.vit_l_16(weights=weights).eval().to(device)
    backbone = ViTBackbone(model).eval().to(device)
    linear_head = model.heads.head if hasattr(model.heads, "head") else next(
        module for module in reversed(list(model.heads.children())) if isinstance(module, nn.Linear)
    )
    head = LinearHead(linear_head).eval().to(device)
    return ModelBundle("ViT_L_16", model, weights.transforms(), backbone, head, linear_head)


def build_swin_v2_b(device):
    weights = torchvision.models.Swin_V2_B_Weights.IMAGENET1K_V1
    model = torchvision.models.swin_v2_b(weights=weights).eval().to(device)
    backbone = SwinBackbone(model).eval().to(device)
    head = LinearHead(model.head).eval().to(device)
    return ModelBundle("Swin_V2_B", model, weights.transforms(), backbone, head, model.head)


def build_maxvit_t(device):
    weights = torchvision.models.MaxVit_T_Weights.IMAGENET1K_V1
    model = torchvision.models.maxvit_t(weights=weights).eval().to(device)
    backbone = MaxVitBackbone(model).eval().to(device)
    head = LinearHead(model.classifier[5]).eval().to(device)
    return ModelBundle("MaxVit_T", model, weights.transforms(), backbone, head, model.classifier[5])


def build_efficientnet_v2_m(device):
    weights = torchvision.models.EfficientNet_V2_M_Weights.IMAGENET1K_V1
    model = torchvision.models.efficientnet_v2_m(weights=weights).eval().to(device)
    backbone = EfficientNetBackbone(model).eval().to(device)
    head = SequentialHead(model.classifier).eval().to(device)
    return ModelBundle("EfficientNet_V2_M", model, weights.transforms(), backbone, head, model.classifier[1])


def build_mobilenet_v3_large(device):
    weights = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
    model = torchvision.models.mobilenet_v3_large(weights=weights).eval().to(device)
    backbone = MobileNetV3LargeBackbone(model).eval().to(device)
    head = LinearHead(model.classifier[3]).eval().to(device)
    return ModelBundle("MobileNet_v3_L", model, weights.transforms(), backbone, head, model.classifier[3])


def build_regnet_y_800mf(device):
    weights = torchvision.models.RegNet_Y_800MF_Weights.IMAGENET1K_V1
    model = torchvision.models.regnet_y_800mf(weights=weights).eval().to(device)
    backbone = RegNetBackbone(model).eval().to(device)
    head = LinearHead(model.fc).eval().to(device)
    return ModelBundle("RegNet_Y_800MF", model, weights.transforms(), backbone, head, model.fc)


MODEL_BUILDERS = {
    "ResNet152": build_resnet152,
    "EfficientNet_B3": build_efficientnet_b3,
    "ConvNeXt_L": build_convnext_l,
    "ViT_L_16": build_vit_l_16,
    "Swin_V2_B": build_swin_v2_b,
    "MaxVit_T": build_maxvit_t,
    "EfficientNet_V2_M": build_efficientnet_v2_m,
    "MobileNet_v3_L": build_mobilenet_v3_large,
    "RegNet_Y_800MF": build_regnet_y_800mf,
}


def build_dataset_specs(preprocess):
    return {
        "train": DatasetSpec("train", ImageNetKaggleWithPaths(DATASET_IMAGE_NET_2012_PATH, "train", transform=preprocess)),
        "test": DatasetSpec("test", ImageNetKaggleWithPaths(DATASET_IMAGE_NET_2012_PATH, "val", transform=preprocess)),
        "imagenet_o": DatasetSpec("imagenet_o", SortedImageFolderWithPaths(DATASET_IMAGE_NET_O_PATH, transform=preprocess)),
        "textures": DatasetSpec("textures", SortedImageFolderWithPaths(DATASET_TEXTURES_PATH, transform=preprocess)),
        "sun2012": DatasetSpec("sun2012", SortedImageFolderWithPaths(DATASET_SUN2012_PATH, transform=preprocess)),
        "iNaturalist": DatasetSpec("iNaturalist", SortedImageFolderWithPaths(DATASET_INATURALIST_PATH, transform=preprocess)),
        "places365": DatasetSpec("places365", SortedImageFolderWithPaths(DATASET_PLACES_PATH, transform=preprocess)),
        "ninco": DatasetSpec("ninco", SortedImageFolderWithPaths(DATASET_NINCO_PATH, transform=preprocess)),
        "imagenetood": DatasetSpec("imagenetood", ImageNetOODWithPaths(DATASET_IMAGENETOOD_PATH, transform=preprocess)),
    }


def build_detector_specs(bundle, norm_std):
    vim_dim = vim_subspace_dim(bundle.linear_head.in_features)
    return [
        DetectorSpec("MSP", lambda: MaxSoftmax(bundle.model)),
        DetectorSpec("TemperatureScaling", lambda: TemperatureScaling(bundle.model), fit_loader_name="train"),
        DetectorSpec("KLMatching", lambda: KLMatching(bundle.model), fit_loader_name="train"),
        DetectorSpec("Entropy", lambda: Entropy(bundle.model)),
        DetectorSpec("MaxLogit", lambda: MaxLogit(bundle.model)),
        DetectorSpec("Energy", lambda: EnergyBased(bundle.model)),
        DetectorSpec("RMD", lambda: RMD(bundle.backbone), fit_loader_name="train"),
        DetectorSpec("ViM", lambda: ViM(bundle.backbone, d=vim_dim, w=bundle.linear_head.weight, b=bundle.linear_head.bias), fit_loader_name="train"),
        DetectorSpec("ReAct", lambda: ReAct(bundle.backbone, bundle.head, threshold=1.0)),
        DetectorSpec("DICE", lambda: DICE(bundle.backbone, w=bundle.linear_head.weight, b=bundle.linear_head.bias, p=0.65), fit_loader_name="train"),
        DetectorSpec("KNN", lambda: KNN(bundle.backbone), fit_loader_name="train"),
        DetectorSpec("ODIN", lambda: ODIN(bundle.model, eps=0.002, temperature=1000.0, norm_std=norm_std), needs_grad=True),
        DetectorSpec("Mahalanobis", lambda: Mahalanobis(bundle.backbone, eps=0.002, norm_std=norm_std), fit_loader_name="train", needs_grad=True),
    ]


def fit_detector_if_needed(detector, spec, loaders, device):
    if spec.fit_loader_name is not None:
        detector.fit(loaders[spec.fit_loader_name], device=str(device))


def score_loader(detector, loader, device, needs_grad):
    rows = []

    if needs_grad:
        torch.set_grad_enabled(True)
        context = torch.enable_grad()
    else:
        torch.set_grad_enabled(False)
        context = torch.inference_mode()

    with context:
        for x, y, paths in loader:
            x = x.to(device, non_blocking=True)
            scores = detector(x).detach().cpu().numpy()
            rows.extend(rows_from_batch(paths, y.cpu().numpy(), scores))

    if needs_grad:
        torch.set_grad_enabled(False)

    return finalize_scores_df(rows)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_names = DEFAULT_MODEL_NAMES
    dataset_names = DEFAULT_DATASET_NAMES
    for model_name in model_names:
        print("Processing model:", model_name)
        bundle = MODEL_BUILDERS[model_name](device)
        norm_std = extract_norm_std(bundle.preprocess)
        dataset_specs = build_dataset_specs(bundle.preprocess)
        train_spec = dataset_specs["train"]
        eval_specs = [dataset_specs[name] for name in dataset_names]

        loaders = {
            "train": make_loader(train_spec.dataset, BATCH_SIZE, NUM_WORKERS, xy_only=True),
        }

        detector_specs = build_detector_specs(bundle, norm_std)
        model_out = Path(OUT_DIR) / model_name
        model_out.mkdir(parents=True, exist_ok=True)

        run_meta = {
            "seed": SEED,
            "model": model_name,
            "device": str(device),
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "datasets": [spec.name for spec in eval_specs],
            "detectors": [spec.name for spec in detector_specs],
            "train_samples": len(train_spec.dataset),
            "vim_subspace_dim": vim_subspace_dim(bundle.linear_head.in_features),
            "output_columns": ["row_id", "image_id", "image_path", "ood_score", "true_label"],
        }
        (model_out / "run_config.json").write_text(json.dumps(run_meta, indent=2))

        for detector_spec in detector_specs:
            print("\tProcessing detector:", detector_spec.name)
            detector = detector_spec.build()
            fit_detector_if_needed(detector, detector_spec, loaders, device)
            detector_out = model_out / detector_spec.name
            detector_out.mkdir(parents=True, exist_ok=True)

            for dataset_spec in eval_specs:
                print("\t\tProcessing dataset:", dataset_spec.name)
                out_path = detector_out / f"{dataset_spec.name}.pickle"
                loader = make_loader(dataset_spec.dataset, BATCH_SIZE, NUM_WORKERS)
                df_scores = score_loader(
                    detector,
                    loader,
                    device,
                    needs_grad=detector_spec.needs_grad,
                )
                save_scores_df(df_scores, str(out_path))


if __name__ == "__main__":
    main()
