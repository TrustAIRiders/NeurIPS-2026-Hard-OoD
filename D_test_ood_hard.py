import os
import json
import random
from pathlib import Path

NUM_WORKERS = 40
os.environ['MKL_NUM_THREADS'] = f'{NUM_WORKERS}'
os.environ['NUMEXPR_NUM_THREADS'] = f'{NUM_WORKERS}'
os.environ['OMP_NUM_THREADS'] = f'{NUM_WORKERS}'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from sklearn.metrics import roc_auc_score, roc_curve
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from pytorch_ood.detector import ASH, DICE, EnergyBased, Entropy, KLMatching, KNN, Mahalanobis, MaxLogit, MaxSoftmax, ODIN, RMD, ReAct, SHE, TemperatureScaling, ViM
ImageFile.LOAD_TRUNCATED_IMAGES = True
SEED = 0
BATCH_SIZE = 64
OUT_DIR = Path('./ood_on_hard')
DATASET_PATH = "/datasets/"  # Set to the directory containing the dataset folders
DATASET_IMAGE_NET_2012_PATH = os.path.join(DATASET_PATH, 'ImageNet2012')
LISTS_DIR = Path('./benchmark_splits')
ID_HARD_LIST = LISTS_DIR / 'id_test_hard_list.txt'
OOD_HARD_LIST = LISTS_DIR / 'ood_hard_list.txt'
SEPARATOR = ','
IMAGENET_STD_FALLBACK = [0.229, 0.224, 0.225]
VIM_SUBSPACE_RATIO = 0.5
MODEL_NAMES = ['densenet201.tv_in1k', 'regnety_064.ra3_in1k', 'convnext_zepto_rms.ra4_e3600_r224_in1k', 'vit_medium_patch16_rope_reg1_gap_256.sbb_in1k', 'pvt_v2_b5.in1k', 'coat_lite_medium.in1k', 'inception_next_base.sail_in1k_384', 'nfnet_l0.ra2_in1k', 'fastvit_sa24.apple_in1k', 'mobilevitv2_175.cvnets_in1k']
DETECTOR_NAMES = ['MSP', 'TemperatureScaling', 'KLMatching', 'Entropy', 'MaxLogit', 'Energy', 'ODIN', 'Mahalanobis', 'RMD', 'ViM', 'KNN', 'SHE', 'ASH', 'ReAct', 'DICE']
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_float32_matmul_precision('high')

class ImageNetKaggleWithPaths(Dataset):

    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        with open(os.path.join(root, 'imagenet_class_index.json'), 'r') as handle:
            json_file = json.load(handle)
        syn_to_class = {value[0]: int(class_id) for class_id, value in json_file.items()}
        with open(os.path.join(root, 'ILSVRC2012_val_labels.json'), 'r') as handle:
            val_to_syn = json.load(handle)
        samples_dir = os.path.join(root, 'ILSVRC/Data/CLS-LOC', split)
        if split == 'train':
            for syn_id in sorted(os.listdir(samples_dir)):
                syn_folder = os.path.join(samples_dir, syn_id)
                if not os.path.isdir(syn_folder):
                    continue
                target = syn_to_class[syn_id]
                for sample in sorted(os.listdir(syn_folder)):
                    sample_path = os.path.join(syn_folder, sample)
                    if os.path.isfile(sample_path):
                        self.samples.append(sample_path)
                        self.targets.append(target)
        elif split == 'val':
            for entry in sorted(os.listdir(samples_dir)):
                sample_path = os.path.join(samples_dir, entry)
                if not os.path.isfile(sample_path):
                    continue
                syn_id = val_to_syn[entry]
                self.samples.append(sample_path)
                self.targets.append(syn_to_class[syn_id])
        else:
            raise ValueError(f'Unknown split: {split}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return (image, self.targets[idx], path)


def resolve_image_path(path):
    path = str(path)
    if path.startswith("/datasets/"):
        path = path[len("/datasets/"):]
    return os.path.join(DATASET_PATH, path)


class PathLabelListDataset(Dataset):

    def __init__(self, list_path, transform=None, sep=','):
        df = pd.read_csv(list_path, header=None, names=['image_path', 'label'], sep=sep)
        df['image_path'] = df['image_path'].astype(str).map(resolve_image_path)
        df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(-1).astype(int)
        self.paths = df['image_path'].tolist()
        self.labels = df['label'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return (image, label, path)

class DetectorSpec:

    def __init__(self, name, build, fit_loader_name=None, needs_grad=False):
        self.name = name
        self.build = build
        self.fit_loader_name = fit_loader_name
        self.needs_grad = needs_grad

class ModelBundle:

    def __init__(self, name, model, preprocess, norm_std, prelogit_backbone, prelogit_head, raw_backbone, raw_head, linear_head):
        self.name = name
        self.model = model
        self.preprocess = preprocess
        self.norm_std = norm_std
        self.prelogit_backbone = prelogit_backbone
        self.prelogit_head = prelogit_head
        self.raw_backbone = raw_backbone
        self.raw_head = raw_head
        self.linear_head = linear_head

class TimmPrelogitBackbone(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.model.forward_head(x, pre_logits=True)
        if x.ndim > 2:
            x = torch.flatten(x, 1)
        return x

class TimmRawBackbone(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.forward_features(x)

class TimmRawHead(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.forward_head(x, pre_logits=False)

class LinearHead(nn.Module):

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return self.layer(x)

def collate_xy_only(batch):
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]
    return (default_collate(xs), default_collate(ys))

def make_loader(dataset, batch_size, num_workers, *, xy_only=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0, collate_fn=collate_xy_only if xy_only else None)

def extract_classifier_linear(model):
    classifier = None
    if hasattr(model, 'get_classifier'):
        try:
            classifier = model.get_classifier()
        except Exception:
            classifier = None
    if isinstance(classifier, nn.Linear):
        return classifier
    if isinstance(classifier, nn.Module):
        linear_layers = [m for m in classifier.modules() if isinstance(m, nn.Linear)]
        if linear_layers:
            return linear_layers[-1]
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not linear_layers:
        raise RuntimeError('Could not find final nn.Linear classifier head.')
    return linear_layers[-1]

def extract_bias_tensor(linear_head):
    if linear_head.bias is not None:
        return linear_head.bias
    return torch.zeros(linear_head.out_features, dtype=linear_head.weight.dtype, device=linear_head.weight.device)

def vim_subspace_dim(feature_dim):
    dim = int(round(feature_dim * VIM_SUBSPACE_RATIO))
    dim = max(64, dim)
    return min(dim, feature_dim - 1)

def build_timm_bundle(model_name, device):
    model = timm.create_model(model_name, pretrained=True).eval().to(device)
    data_config = resolve_model_data_config(model)
    preprocess = create_transform(**data_config, is_training=False)
    norm_std = list(data_config.get('std', IMAGENET_STD_FALLBACK))
    linear_head = extract_classifier_linear(model).eval().to(device)
    prelogit_backbone = TimmPrelogitBackbone(model).eval().to(device)
    prelogit_head = LinearHead(linear_head).eval().to(device)
    raw_backbone = TimmRawBackbone(model).eval().to(device)
    raw_head = TimmRawHead(model).eval().to(device)
    return ModelBundle(name=model_name, model=model, preprocess=preprocess, norm_std=norm_std, prelogit_backbone=prelogit_backbone, prelogit_head=prelogit_head, raw_backbone=raw_backbone, raw_head=raw_head, linear_head=linear_head)

def build_detector_specs(bundle):
    bias = extract_bias_tensor(bundle.linear_head)
    vim_dim = vim_subspace_dim(bundle.linear_head.in_features)
    return [DetectorSpec('MSP', lambda: MaxSoftmax(bundle.model)), DetectorSpec('TemperatureScaling', lambda: TemperatureScaling(bundle.model), fit_loader_name='train'), DetectorSpec('KLMatching', lambda: KLMatching(bundle.model), fit_loader_name='train'), DetectorSpec('Entropy', lambda: Entropy(bundle.model)), DetectorSpec('MaxLogit', lambda: MaxLogit(bundle.model)), DetectorSpec('Energy', lambda: EnergyBased(bundle.model)), DetectorSpec('ODIN', lambda: ODIN(bundle.model, eps=0.002, temperature=1000.0, norm_std=bundle.norm_std), needs_grad=True), DetectorSpec('Mahalanobis', lambda: Mahalanobis(bundle.prelogit_backbone, eps=0.002, norm_std=bundle.norm_std), fit_loader_name='train', needs_grad=True), DetectorSpec('RMD', lambda: RMD(bundle.prelogit_backbone), fit_loader_name='train'), DetectorSpec('ViM', lambda: ViM(bundle.prelogit_backbone, d=vim_dim, w=bundle.linear_head.weight, b=bias), fit_loader_name='train'), DetectorSpec('KNN', lambda: KNN(bundle.prelogit_backbone), fit_loader_name='train'), DetectorSpec('SHE', lambda: SHE(bundle.prelogit_backbone, bundle.prelogit_head), fit_loader_name='train'), DetectorSpec('ASH', lambda: ASH(bundle.raw_backbone, bundle.raw_head, variant='ash-s', percentile=0.65)), DetectorSpec('ReAct', lambda: ReAct(bundle.prelogit_backbone, bundle.prelogit_head, threshold=1.0)), DetectorSpec('DICE', lambda: DICE(bundle.prelogit_backbone, w=bundle.linear_head.weight, b=bias, p=0.65), fit_loader_name='train')]

def fit_detector_if_needed(detector, spec, loaders, device):
    if spec.fit_loader_name is None:
        return
    detector.fit(loaders[spec.fit_loader_name], device=str(device))

def collect_scores(detector, loader, device, needs_grad):
    scores_all = []
    context = torch.enable_grad() if needs_grad else torch.inference_mode()
    with context:
        for batch in loader:
            x, _, _ = batch
            x = x.to(device, non_blocking=True)
            scores = detector(x)
            if isinstance(scores, (tuple, list)):
                scores = scores[0]
            scores = scores.detach().float().cpu().numpy().reshape(-1)
            scores_all.append(scores)
    if not scores_all:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(scores_all, axis=0)

def compute_auroc_fpr95(id_scores, ood_scores):
    labels = np.concatenate([np.zeros(len(id_scores), dtype=np.int64), np.ones(len(ood_scores), dtype=np.int64)], axis=0)
    scores = np.concatenate([id_scores, ood_scores], axis=0)
    auroc = float(roc_auc_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    valid = np.where(tpr >= 0.95)[0]
    fpr95 = float(np.min(fpr[valid])) if len(valid) > 0 else 1.0
    return (auroc, fpr95)

def save_metrics_tables(df, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    long_path = out_dir / 'ood_hard_metrics_long.csv'
    df.to_csv(long_path, index=False)
    ok_df = df.loc[df['status'] == 'ok'].copy()
    auroc_matrix = ok_df.pivot(index='model_name', columns='detector_name', values='auroc').reindex(index=MODEL_NAMES, columns=DETECTOR_NAMES)
    fpr95_matrix = ok_df.pivot(index='model_name', columns='detector_name', values='fpr95').reindex(index=MODEL_NAMES, columns=DETECTOR_NAMES)
    auroc_matrix.to_csv(out_dir / 'ood_hard_auroc_matrix.csv')
    fpr95_matrix.to_csv(out_dir / 'ood_hard_fpr95_matrix.csv')
    return (auroc_matrix, fpr95_matrix)

def plot_heatmap(matrix, out_path, title, cmap, value_format='.3f'):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    values = matrix.to_numpy(dtype=float)
    fig_w = max(14, 0.9 * matrix.shape[1] + 6)
    fig_h = max(8, 0.6 * matrix.shape[0] + 4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220)
    im = ax.imshow(values, aspect='auto', cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns.tolist(), rotation=45, ha='right')
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index.tolist())
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = values[i, j]
            text = 'NaN' if np.isnan(value) else format(value, value_format)
            color = 'white' if not np.isnan(value) and value > 0.55 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('score', rotation=90)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required. Expected device cuda:0.')
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    available_models = set(timm.list_models(pretrained=True))
    missing_models = [m for m in MODEL_NAMES if m not in available_models]
    if missing_models:
        raise RuntimeError(f'These timm models are not available in the current installation: {missing_models}')
    rows = []
    for model_name in MODEL_NAMES:
        print("Processing model:", model_name)
        bundle = build_timm_bundle(model_name, device)
        train_dataset = ImageNetKaggleWithPaths(DATASET_IMAGE_NET_2012_PATH, 'train', transform=bundle.preprocess)
        id_hard_dataset = PathLabelListDataset(ID_HARD_LIST, transform=bundle.preprocess, sep=SEPARATOR)
        ood_hard_dataset = PathLabelListDataset(OOD_HARD_LIST, transform=bundle.preprocess, sep=SEPARATOR)
        loaders = {'train': make_loader(train_dataset, BATCH_SIZE, NUM_WORKERS, xy_only=True), 'id_hard': make_loader(id_hard_dataset, BATCH_SIZE, NUM_WORKERS), 'ood_hard': make_loader(ood_hard_dataset, BATCH_SIZE, NUM_WORKERS)}
        detector_specs = build_detector_specs(bundle)
        detector_specs = [spec for spec in detector_specs if spec.name in DETECTOR_NAMES]
        for detector_spec in detector_specs:
            print("\tProcessing detector:", detector_spec.name)
            try:
                detector = detector_spec.build()
                fit_detector_if_needed(detector, detector_spec, loaders, device)
                id_scores = collect_scores(detector=detector, loader=loaders['id_hard'], device=device, needs_grad=detector_spec.needs_grad)
                ood_scores = collect_scores(detector=detector, loader=loaders['ood_hard'], device=device, needs_grad=detector_spec.needs_grad)
                auroc, fpr95 = compute_auroc_fpr95(id_scores=id_scores, ood_scores=ood_scores)
                rows.append({'model_name': model_name, 'detector_name': detector_spec.name, 'auroc': auroc, 'fpr95': fpr95, 'n_id_hard': int(len(id_scores)), 'n_ood_hard': int(len(ood_scores)), 'status': 'ok', 'error': ''})
            except Exception as exc:
                rows.append({'model_name': model_name, 'detector_name': detector_spec.name, 'auroc': np.nan, 'fpr95': np.nan, 'n_id_hard': np.nan, 'n_ood_hard': np.nan, 'status': 'error', 'error': repr(exc)})
            finally:
                torch.cuda.empty_cache()
        del bundle
        torch.cuda.empty_cache()
    results_df = pd.DataFrame(rows)
    auroc_matrix, fpr95_matrix = save_metrics_tables(results_df, OUT_DIR)
    plot_heatmap(matrix=auroc_matrix, out_path=OUT_DIR / 'ood_hard_auroc_heatmap.png', title='OOD hard: AUROC', cmap='viridis')
    plot_heatmap(matrix=fpr95_matrix, out_path=OUT_DIR / 'ood_hard_fpr95_heatmap.png', title='OOD hard: FPR95 (lower is better)', cmap='viridis_r')
if __name__ == '__main__':
    main()
