import gc
import json
import os
import random
import re
from functools import lru_cache
from pathlib import Path

NUM_WORKERS = 5
os.environ["MKL_NUM_THREADS"] = str(NUM_WORKERS)
os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_WORKERS)
os.environ["OMP_NUM_THREADS"] = str(NUM_WORKERS)

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFile
from scipy.stats import chi2_contingency, spearmanr
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASET_PATH = '/datasets/'  # TODO: replace with dataset directory
OUT_ROOT = Path('./cifar10')
SCORES_DIR = OUT_ROOT / 'scores'
SHARED_DIR = OUT_ROOT / 'shared_analysis'
HARD_DIR = OUT_ROOT / 'hard_report'
ID_DATASET_NAME = 'test'
TRAIN_DATASET_NAME = 'train'
OOD_DATASET_NAMES = ['cifar100', 'svhn']
MODEL_NAMES = ['cifar10_repvgg_a2', 'cifar10_mobilenetv2_x1_4', 'cifar10_resnet56', 'cifar10_vgg16_bn']
DETECTOR_NAMES = ['MaxSoftmax', 'Energy', 'KNN', 'Mahalanobis']
SUBSET_SIZE = 500
FIGURE_SAMPLE_SIZE = 100
RANDOM_SEED = 0
BATCH_SIZE = 512
KNN_BATCH_SIZE = 512
NUM_WORKERS = 8
KNN_K = 50
MAHALANOBIS_RIDGE = 1e-4
DOWNLOAD_DATASETS = True
EXPORT_IMAGE_CACHE = True
FORCE_RECOMPUTE = False
USE_AMP = True
HUB_DIR = None
DEVICE = 'auto'
DPI = 220
HEATMAP_DPI = 160
MIN_SHARED_SAMPLES = 30
N_BINS = 10
P_VALUE_MIN = 1e-10
P_VALUE_MAX = 1e-1
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

META_COLUMNS = {'image_id', 'image_path', 'true_label', 'dataset_name'}


def safe_name(value):
    value = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value)).strip('_')
    return value or 'value'


def detector_key(model_name, detector_name):
    return f'{model_name}__{detector_name}'


def split_detector_key(value):
    if '__' not in value:
        return value, ''
    return tuple(value.split('__', 1))


def short_model_name(model_name):
    return str(model_name).replace('cifar10_', '')


def short_detector_key(value):
    model_name, detector_name = split_detector_key(value)
    return f'{short_model_name(model_name)} + {detector_name}'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass


def resolve_device():
    if DEVICE == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if DEVICE == 'cuda' and not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device(DEVICE)


class IndexedVisionDataset(Dataset):
    def __init__(self, base_dataset, dataset_name, transform, image_cache_dir):
        self.base_dataset = base_dataset
        self.dataset_name = dataset_name
        self.transform = transform
        self.image_cache_dir = image_cache_dir

    def __len__(self):
        return len(self.base_dataset)

    def image_id(self, idx):
        return f'{self.dataset_name}_{idx:06d}'

    def image_path(self, idx):
        return str((self.image_cache_dir / self.dataset_name / f'{idx:06d}.png').resolve())

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, int(label), self.image_id(idx), self.image_path(idx)


class LastLinearFeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.layer = self.find_last_linear(model)
        self.features = None
        self.handle = self.layer.register_forward_pre_hook(self.capture)

    def find_last_linear(self, model):
        layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
        if not layers:
            raise RuntimeError('No linear layer found for feature extraction')
        return layers[-1]

    def capture(self, module, inputs):
        x = inputs[0]
        if x.ndim > 2:
            x = torch.flatten(x, 1)
        self.features = x.detach()

    def __call__(self, x):
        self.features = None
        logits = self.model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        if self.features is None:
            raise RuntimeError('Feature hook did not capture activations')
        return logits, self.features

    def close(self):
        self.handle.remove()


def collate_batch(batch):
    images = torch.stack([x[0] for x in batch], dim=0)
    labels = torch.tensor([x[1] for x in batch], dtype=torch.long)
    image_ids = [x[2] for x in batch]
    image_paths = [x[3] for x in batch]
    return images, labels, image_ids, image_paths


def make_loader(dataset, batch_size, num_workers, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=False,
        collate_fn=collate_batch,
    )


def build_raw_datasets():
    dataset_root = Path(DATASET_PATH)
    dataset_root.mkdir(parents=True, exist_ok=True)
    return {
        TRAIN_DATASET_NAME: torchvision.datasets.CIFAR10(root=str(dataset_root), train=True, download=DOWNLOAD_DATASETS, transform=None),
        ID_DATASET_NAME: torchvision.datasets.CIFAR10(root=str(dataset_root), train=False, download=DOWNLOAD_DATASETS, transform=None),
        'cifar100': torchvision.datasets.CIFAR100(root=str(dataset_root), train=False, download=DOWNLOAD_DATASETS, transform=None),
        'svhn': torchvision.datasets.SVHN(root=str(dataset_root), split='test', download=DOWNLOAD_DATASETS, transform=None),
    }


def build_datasets(raw_datasets):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    image_cache_dir = OUT_ROOT / 'cache' / 'images'
    return {name: IndexedVisionDataset(raw, name, transform, image_cache_dir) for name, raw in raw_datasets.items()}


def export_image_cache(raw_datasets, wrapped, dataset_names, force):
    for dataset_name in dataset_names:
        raw = raw_datasets[dataset_name]
        ds = wrapped[dataset_name]
        out_dir = ds.image_cache_dir / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(out_dir.glob('*.png')))
        if existing == len(raw) and not force:
            continue
        for idx in range(len(raw)):
            out_path = Path(ds.image_path(idx))
            if out_path.exists() and not force:
                continue
            image, _ = raw[idx]
            if not isinstance(image, Image.Image):
                image = Image.fromarray(np.asarray(image))
            image.convert('RGB').save(out_path)


def load_model(model_name, device):
    if HUB_DIR:
        torch.hub.set_dir(HUB_DIR)
    model = torch.hub.load('chenyaofo/pytorch-cifar-models', model_name, pretrained=True, trust_repo=True)
    model.eval().to(device)
    return model


def feature_cache_path(model_name, dataset_name):
    return OUT_ROOT / 'cache' / 'features' / model_name / f'{dataset_name}.npz'


def save_feature_cache(cache, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        dataset_name=np.array(cache['dataset_name'], dtype=object),
        model_name=np.array(cache['model_name'], dtype=object),
        image_ids=cache['image_ids'].astype(object),
        image_paths=cache['image_paths'].astype(object),
        labels=cache['labels'].astype(np.int64),
        logits=cache['logits'].astype(np.float32),
        features=cache['features'].astype(np.float32),
    )


def load_feature_cache(path):
    data = np.load(path, allow_pickle=True)
    return {
        'dataset_name': str(data['dataset_name'].item()) if 'dataset_name' in data.files else path.stem,
        'model_name': str(data['model_name'].item()) if 'model_name' in data.files else path.parent.name,
        'image_ids': data['image_ids'].astype(str),
        'image_paths': data['image_paths'].astype(str),
        'labels': data['labels'].astype(np.int64),
        'logits': data['logits'].astype(np.float32),
        'features': data['features'].astype(np.float32),
    }


def extract_or_load_features(model_name, extractor, dataset_name, loader, device):
    path = feature_cache_path(model_name, dataset_name)
    if path.exists() and not FORCE_RECOMPUTE:
        return load_feature_cache(path)

    all_ids = []
    all_paths = []
    all_labels = []
    all_logits = []
    all_features = []

    with torch.inference_mode():
        for images, labels, image_ids, image_paths in loader:
            images = images.to(device, non_blocking=True)
            amp_enabled = bool(USE_AMP and device.type == 'cuda')
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp_enabled):
                logits, features = extractor(images)
            all_ids.extend(image_ids)
            all_paths.extend([str(Path(p).resolve()) for p in image_paths])
            all_labels.append(labels.cpu().numpy().astype(np.int64))
            all_logits.append(logits.detach().float().cpu().numpy())
            all_features.append(features.detach().float().cpu().numpy())

    cache = {
        'dataset_name': dataset_name,
        'model_name': model_name,
        'image_ids': np.asarray(all_ids, dtype=object),
        'image_paths': np.asarray(all_paths, dtype=object),
        'labels': np.concatenate(all_labels, axis=0),
        'logits': np.concatenate(all_logits, axis=0),
        'features': np.concatenate(all_features, axis=0),
    }
    save_feature_cache(cache, path)
    return cache


def fit_reference_state(train_cache, model_name, device):
    ref_dir = OUT_ROOT / 'cache' / 'references' / model_name
    means_path = ref_dir / 'class_means.npy'
    precision_path = ref_dir / 'precision.npy'

    train_features = torch.from_numpy(train_cache['features']).float().to(device)
    train_labels = torch.from_numpy(train_cache['labels']).long().to(device)
    train_features_normalized = F.normalize(train_features, dim=1)

    if means_path.exists() and precision_path.exists() and not FORCE_RECOMPUTE:
        class_means = torch.from_numpy(np.load(means_path)).float().to(device)
        precision = torch.from_numpy(np.load(precision_path)).float().to(device)
        return {
            'train_features': train_features,
            'train_features_normalized': train_features_normalized,
            'class_means': class_means,
            'precision': precision,
        }

    class_means = []
    for class_idx in range(10):
        mask = train_labels == class_idx
        if not bool(mask.any()):
            raise RuntimeError(f'No training samples for class {class_idx}')
        class_means.append(train_features[mask].mean(dim=0))
    class_means = torch.stack(class_means, dim=0)

    centered = train_features - class_means[train_labels]
    denom = max(1, int(centered.shape[0] - 10))
    cov = centered.T.matmul(centered) / float(denom)
    diag_mean = torch.diag(cov).mean().clamp_min(1e-12)
    cov = cov + torch.eye(cov.shape[0], device=device, dtype=cov.dtype) * float(MAHALANOBIS_RIDGE) * diag_mean

    try:
        precision = torch.linalg.inv(cov)
    except RuntimeError:
        precision = torch.linalg.pinv(cov)

    ref_dir.mkdir(parents=True, exist_ok=True)
    np.save(means_path, class_means.detach().float().cpu().numpy())
    np.save(precision_path, precision.detach().float().cpu().numpy())

    return {
        'train_features': train_features,
        'train_features_normalized': train_features_normalized,
        'class_means': class_means,
        'precision': precision,
    }


def logsumexp_np(logits):
    logits = logits.astype(np.float64)
    max_v = np.max(logits, axis=1, keepdims=True)
    return max_v[:, 0] + np.log(np.exp(logits - max_v).sum(axis=1))


def compute_msp_scores(logits):
    logits_t = torch.from_numpy(logits).float()
    max_prob = torch.softmax(logits_t, dim=1).max(dim=1).values.numpy()
    return (1.0 - max_prob).astype(np.float32)


def compute_energy_scores(logits):
    return (-logsumexp_np(logits)).astype(np.float32)


def compute_knn_scores(features, state, device):
    scores = []
    train = state['train_features_normalized']
    kk = min(int(KNN_K), int(train.shape[0]))
    for start in range(0, len(features), KNN_BATCH_SIZE):
        batch = torch.from_numpy(features[start:start + KNN_BATCH_SIZE]).float().to(device)
        batch = F.normalize(batch, dim=1)
        distances = torch.cdist(batch, train)
        kth = torch.topk(distances, k=kk, dim=1, largest=False).values[:, -1]
        scores.append(kth.detach().float().cpu().numpy())
    return np.concatenate(scores, axis=0).astype(np.float32)


def compute_mahalanobis_scores(features, state, device):
    scores = []
    means = state['class_means']
    precision = state['precision']
    for start in range(0, len(features), KNN_BATCH_SIZE):
        batch = torch.from_numpy(features[start:start + KNN_BATCH_SIZE]).float().to(device)
        diff = batch[:, None, :] - means[None, :, :]
        left = torch.matmul(diff, precision)
        distances = (left * diff).sum(dim=2)
        scores.append(distances.min(dim=1).values.detach().float().cpu().numpy())
    return np.concatenate(scores, axis=0).astype(np.float32)


def score_path(model_name, detector_name, dataset_name):
    return SCORES_DIR / model_name / detector_name / f'{dataset_name}.pickle'


def build_score_df(cache, model_name, detector_name, scores):
    df = pd.DataFrame({
        'image_id': cache['image_ids'].astype(str),
        'image_path': cache['image_paths'].astype(str),
        'ood_score': scores.astype(float),
        'true_label': cache['labels'].astype(int),
        'dataset_name': cache['dataset_name'],
        'model_name': model_name,
        'detector_name': detector_name,
    })
    return df.sort_values('image_id').reset_index(drop=True)


def save_score_df(df, model_name, detector_name, dataset_name):
    path = score_path(model_name, detector_name, dataset_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)


def compute_and_save_scores(cache, state, model_name, device):
    scores_by_detector = {
        'MaxSoftmax': compute_msp_scores(cache['logits']),
        'Energy': compute_energy_scores(cache['logits']),
        'KNN': compute_knn_scores(cache['features'], state, device),
        'Mahalanobis': compute_mahalanobis_scores(cache['features'], state, device),
    }
    for detector_name, scores in scores_by_detector.items():
        save_score_df(build_score_df(cache, model_name, detector_name, scores), model_name, detector_name, cache['dataset_name'])


def missing_score_paths(model_names, detector_names, dataset_names):
    missing = []
    for model_name in model_names:
        for detector_name in detector_names:
            for dataset_name in dataset_names:
                path = score_path(model_name, detector_name, dataset_name)
                if not path.exists() or FORCE_RECOMPUTE:
                    missing.append(path)
    return missing


def run_scoring():
    dataset_names = [ID_DATASET_NAME] + OOD_DATASET_NAMES
    if not missing_score_paths(MODEL_NAMES, DETECTOR_NAMES, dataset_names):
        return

    raw_datasets = build_raw_datasets()
    datasets = build_datasets(raw_datasets)

    if EXPORT_IMAGE_CACHE:
        export_image_cache(raw_datasets, datasets, [TRAIN_DATASET_NAME] + dataset_names, FORCE_RECOMPUTE)

    device = resolve_device()

    for model_name in MODEL_NAMES:
        if not missing_score_paths([model_name], DETECTOR_NAMES, dataset_names):
            continue

        model = load_model(model_name, device)
        extractor = LastLinearFeatureExtractor(model)
        loaders = {name: make_loader(datasets[name], BATCH_SIZE, NUM_WORKERS) for name in [TRAIN_DATASET_NAME] + dataset_names}
        train_cache = extract_or_load_features(model_name, extractor, TRAIN_DATASET_NAME, loaders[TRAIN_DATASET_NAME], device)
        state = fit_reference_state(train_cache, model_name, device)

        for dataset_name in dataset_names:
            if not missing_score_paths([model_name], DETECTOR_NAMES, [dataset_name]):
                continue
            cache = extract_or_load_features(model_name, extractor, dataset_name, loaders[dataset_name], device)
            compute_and_save_scores(cache, state, model_name, device)

        extractor.close()
        del state, train_cache, extractor, model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def load_scores_root():
    mapping = {}
    for path in SCORES_DIR.rglob('*.pickle'):
        rel = path.relative_to(SCORES_DIR).parts
        if len(rel) < 3:
            continue
        mapping[(rel[0], rel[1], path.stem)] = path
    if not mapping:
        raise RuntimeError(f'No .pickle score files found under {SCORES_DIR.resolve()}')

    cache = {}

    def get_df(model_name, detector_name, dataset_name):
        key = (model_name, detector_name, dataset_name)
        if key in cache:
            return cache[key]
        df = pd.read_pickle(mapping[key])
        df = df[['image_id', 'image_path', 'ood_score', 'true_label']].copy()
        df['image_id'] = df['image_id'].astype(str)
        df['image_path'] = df['image_path'].astype(str)
        df['ood_score'] = pd.to_numeric(df['ood_score'], errors='coerce')
        df['true_label'] = pd.to_numeric(df['true_label'], errors='coerce').fillna(-1).astype(int)
        df['dataset_name'] = dataset_name
        df['model_name'] = model_name
        df['detector_name'] = detector_name
        df = df.dropna(subset=['image_id', 'ood_score'])
        df = df.drop_duplicates(subset=['image_id'], keep='first').reset_index(drop=True)
        cache[key] = df
        return df

    return mapping, get_df


def required_detector_keys(model_names, detector_names):
    return [detector_key(model_name, detector_name) for model_name in model_names for detector_name in detector_names]


def join_scores_for_dataset(mapping, get_df, dataset_name, detector_keys, how='inner'):
    score_series = []
    meta_frames = []
    for key in detector_keys:
        model_name, detector_name = split_detector_key(key)
        if (model_name, detector_name, dataset_name) not in mapping:
            continue
        df = get_df(model_name, detector_name, dataset_name)
        score_series.append(df.set_index('image_id')['ood_score'].rename(key))
        meta_frames.append(df.set_index('image_id')[['image_path', 'true_label', 'dataset_name']])
    if not score_series:
        return pd.DataFrame()
    scores = pd.concat(score_series, axis=1)
    meta = pd.concat(meta_frames, axis=0).groupby(level=0).first()
    return meta.join(scores, how=how).sort_index().reset_index().rename(columns={'index': 'image_id'})


def build_hard_percentiles(wide, detector_keys, is_id):
    out = pd.DataFrame(index=wide.index)
    for key in detector_keys:
        if key not in wide.columns:
            continue
        values = pd.to_numeric(wide[key], errors='coerce')
        out[key] = values.rank(pct=True, method='average', ascending=bool(is_id))
    return out


def build_reference_summary(wide, detector_keys, is_id):
    keys = [key for key in detector_keys if key in wide.columns]
    hard = build_hard_percentiles(wide, keys, is_id)
    easy = 1.0 - hard
    summary = wide[['image_id', 'image_path', 'true_label', 'dataset_name']].copy()
    summary['available_pair_count'] = hard.notna().sum(axis=1).astype(int)
    summary['mean_hard_all'] = hard.mean(axis=1, skipna=True)
    summary['mean_easy_all'] = easy.mean(axis=1, skipna=True)
    summary['hard_std_all'] = hard.std(axis=1, skipna=True).fillna(0.0)
    summary['hard_rank'] = summary['mean_hard_all'].rank(method='first', ascending=False).astype(int)
    summary['easy_rank'] = summary['mean_easy_all'].rank(method='first', ascending=False).astype(int)
    return summary.sort_values(['hard_rank', 'dataset_name', 'image_id']).reset_index(drop=True)


def select_top(summary, size, mode):
    if len(summary) < size:
        raise RuntimeError(f'Requested {size} samples, but only {len(summary)} are available')
    score_col = 'mean_hard_all' if mode == 'hard' else 'mean_easy_all'
    out = summary.sort_values([score_col, 'hard_std_all', 'dataset_name', 'image_id'], ascending=[False, True, True, True]).head(size).copy().reset_index(drop=True)
    out['selection_rank'] = np.arange(1, len(out) + 1)
    out['selection_mode'] = mode
    out['selection_size'] = int(size)
    out['selection_score_value'] = pd.to_numeric(out[score_col], errors='coerce')
    return out


def scores_for_ids(mapping, get_df, det_key, dataset_name, image_ids):
    model_name, detector_name = split_detector_key(det_key)
    df = get_df(model_name, detector_name, dataset_name)
    if image_ids is None:
        values = df['ood_score'].to_numpy(dtype=float)
    else:
        series = df.set_index('image_id')['ood_score']
        values = series.reindex([str(x) for x in image_ids]).dropna().to_numpy(dtype=float)
    return values[np.isfinite(values)]


def auroc_for_scores(id_scores, ood_scores):
    id_scores = np.asarray(id_scores, dtype=float)
    ood_scores = np.asarray(ood_scores, dtype=float)
    id_scores = id_scores[np.isfinite(id_scores)]
    ood_scores = ood_scores[np.isfinite(ood_scores)]
    if len(id_scores) < 2 or len(ood_scores) < 2:
        return {'auroc': np.nan, 'n_id': int(len(id_scores)), 'n_ood': int(len(ood_scores))}
    y_true = np.concatenate([np.zeros(len(id_scores), dtype=int), np.ones(len(ood_scores), dtype=int)])
    y_score = np.concatenate([id_scores, ood_scores])
    return {'auroc': float(roc_auc_score(y_true, y_score)), 'n_id': int(len(id_scores)), 'n_ood': int(len(ood_scores))}


def compute_full_aurocs(mapping, get_df):
    rows = []
    for ood_dataset in OOD_DATASET_NAMES:
        for model_name in MODEL_NAMES:
            for detector_name in DETECTOR_NAMES:
                key = detector_key(model_name, detector_name)
                id_scores = scores_for_ids(mapping, get_df, key, ID_DATASET_NAME, None)
                ood_scores = scores_for_ids(mapping, get_df, key, ood_dataset, None)
                rows.append({'ood_dataset': ood_dataset, 'model_name': model_name, 'detector_name': detector_name, 'detector_key': key, **auroc_for_scores(id_scores, ood_scores)})
    return pd.DataFrame(rows)


def compute_spearman(hard):
    keys = list(hard.columns)
    n = len(keys)
    rho = np.full((n, n), np.nan)
    for i in range(n):
        x = pd.to_numeric(hard.iloc[:, i], errors='coerce').to_numpy(dtype=float)
        rho[i, i] = 1.0
        for j in range(i + 1, n):
            y = pd.to_numeric(hard.iloc[:, j], errors='coerce').to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            if int(ok.sum()) < MIN_SHARED_SAMPLES:
                continue
            r, _ = spearmanr(x[ok], y[ok])
            if np.isfinite(r):
                rho[i, j] = r
                rho[j, i] = r
    return rho


def compute_binned_agreement(hard):
    keys = list(hard.columns)
    n = len(keys)
    agreement = np.full((n, n), np.nan)
    p_values = np.full((n, n), np.nan)
    counts = np.zeros((n, n), dtype=int)
    for i in range(n):
        x = pd.to_numeric(hard.iloc[:, i], errors='coerce').to_numpy(dtype=float)
        agreement[i, i] = 1.0
        p_values[i, i] = P_VALUE_MIN
        counts[i, i] = int(np.isfinite(x).sum())
        for j in range(i + 1, n):
            y = pd.to_numeric(hard.iloc[:, j], errors='coerce').to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            count = int(ok.sum())
            counts[i, j] = count
            counts[j, i] = count
            if count < MIN_SHARED_SAMPLES:
                continue
            x_bins = np.clip((x[ok] * N_BINS).astype(int), 0, N_BINS - 1)
            y_bins = np.clip((y[ok] * N_BINS).astype(int), 0, N_BINS - 1)
            contingency = np.zeros((N_BINS, N_BINS), dtype=int)
            for xb, yb in zip(x_bins, y_bins):
                contingency[xb, yb] += 1
            agreement_value = float(np.mean(x_bins == y_bins))
            agreement[i, j] = agreement_value
            agreement[j, i] = agreement_value
            try:
                _, p_value, _, _ = chi2_contingency(contingency)
                if np.isfinite(p_value):
                    p_value = float(np.clip(p_value, P_VALUE_MIN, P_VALUE_MAX))
                    p_values[i, j] = p_value
                    p_values[j, i] = p_value
            except Exception:
                pass
    return agreement, p_values, counts


def color_ticks(ax, keys, axis):
    detector_names = [split_detector_key(key)[1] for key in keys]
    unique = sorted(set(detector_names))
    cmap = plt.get_cmap('tab20')
    colors = {name: cmap(i % 20) for i, name in enumerate(unique)}
    ticks = ax.get_xticklabels() if axis == 'x' else ax.get_yticklabels()
    for tick, detector_name in zip(ticks, detector_names):
        tick.set_color(colors[detector_name])


def save_spearman_png(path, dataset_name, keys, rho):
    n = len(keys)
    size = max(10, min(30, 0.42 * n + 5))
    fig, ax = plt.subplots(figsize=(size, size), dpi=HEATMAP_DPI)
    im = ax.imshow(rho, cmap='coolwarm', vmin=-1, vmax=1, interpolation='nearest')
    labels = [short_detector_key(key) for key in keys]
    ax.set_title(f'{dataset_name} | Spearman')
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    color_ticks(ax, keys, 'x')
    color_ticks(ax, keys, 'y')
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def save_pvalue_png(path, dataset_name, keys, agreement, p_values, counts):
    n = len(keys)
    data = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                data[i, j] = P_VALUE_MIN
            elif counts[i, j] < MIN_SHARED_SAMPLES:
                data[i, j] = np.nan
            elif np.isfinite(p_values[i, j]):
                data[i, j] = p_values[i, j]
            else:
                data[i, j] = P_VALUE_MAX
    fig, ax = plt.subplots(figsize=(max(10, 0.42 * n + 5), max(10, 0.42 * n + 5)), dpi=HEATMAP_DPI)
    cmap = plt.get_cmap('RdYlGn_r').copy()
    cmap.set_bad(color='lightgray')
    from matplotlib.colors import LogNorm
    im = ax.imshow(data, cmap=cmap, norm=LogNorm(vmin=P_VALUE_MIN, vmax=P_VALUE_MAX), interpolation='nearest')
    labels = [short_detector_key(key) for key in keys]
    ax.set_title(f'Binned Agreement ({N_BINS} bins) | color = p-value | dataset={dataset_name}')
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    color_ticks(ax, keys, 'x')
    color_ticks(ax, keys, 'y')

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label('p-value (log scale)')
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def save_difficulty_png(path, dataset_name, keys, hard):
    mean = hard.mean(axis=1, skipna=True)
    order = mean.sort_values(ascending=False).index
    matrix = hard.loc[order, keys].T
    width = max(10, min(32, 0.02 * matrix.shape[1] + 10))
    height = max(8, min(50, 0.35 * matrix.shape[0] + 3))
    fig, ax = plt.subplots(figsize=(width, height), dpi=HEATMAP_DPI)
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap='RdYlGn_r', vmin=0, vmax=1, aspect='auto', interpolation='nearest')
    ax.set_title(f'{dataset_name} | difficulty heatmap')
    ax.set_xticks([])
    ax.set_yticks(np.arange(len(keys)))
    ax.set_yticklabels([short_detector_key(key) for key in keys], fontsize=6)
    color_ticks(ax, keys, 'y')
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def save_shared_outputs(mapping, get_df):
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    all_keys = required_detector_keys(MODEL_NAMES, DETECTOR_NAMES)
    full_metrics = compute_full_aurocs(mapping, get_df)
    full_metrics.to_csv(SHARED_DIR / 'auroc_raw.csv', index=False)

    for dataset_name in [ID_DATASET_NAME] + OOD_DATASET_NAMES:
        wide = join_scores_for_dataset(mapping, get_df, dataset_name, all_keys)
        keys = [key for key in all_keys if key in wide.columns]
        if len(keys) < 2:
            continue
        hard = build_hard_percentiles(wide, keys, dataset_name == ID_DATASET_NAME)
        rho = compute_spearman(hard)
        agreement, p_values, counts = compute_binned_agreement(hard)
        name = safe_name(dataset_name)
        save_spearman_png(SHARED_DIR / f'{name}_sperman.png', dataset_name, keys, rho)
        save_pvalue_png(SHARED_DIR / f'{name}_pvalue.png', dataset_name, keys, agreement, p_values, counts)
        save_difficulty_png(SHARED_DIR / f'{name}_heatmap.png', dataset_name, keys, hard)


def save_path_label_txt(df, path, force_label=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    work = df.copy()
    if force_label is None:
        work['label_out'] = pd.to_numeric(work['true_label'], errors='coerce').fillna(-1).astype(int)
    else:
        work['label_out'] = int(force_label)
    with path.open('w', encoding='utf-8') as f:
        for row in work[['image_path', 'label_out']].itertuples(index=False):
            f.write(f'{row.image_path},{int(row.label_out)}\n')


@lru_cache(maxsize=20000)
def read_tile(path, tile_px):
    try:
        with Image.open(path) as img:
            img = img.convert('RGB')
            img = img.resize((tile_px, tile_px), Image.Resampling.LANCZOS)
            return np.asarray(img, dtype=np.uint8)
    except Exception:
        img = Image.new('RGB', (tile_px, tile_px), (245, 245, 245))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, tile_px - 1, tile_px - 1], outline=(180, 180, 180), width=max(1, tile_px // 24))
        draw.text((max(2, tile_px // 8), tile_px // 2 - 5), 'missing', fill=(80, 80, 80))
        return np.asarray(img, dtype=np.uint8)


def make_mosaic(paths, rows, cols, tile_px, line_px=1):
    paths = list(paths)[:rows * cols]
    height = rows * tile_px + (rows + 1) * line_px
    width = cols * tile_px + (cols + 1) * line_px
    canvas = np.full((height, width, 3), 12, dtype=np.uint8)
    for idx in range(rows * cols):
        r = idx // cols
        c = idx % cols
        y0 = line_px + r * (tile_px + line_px)
        x0 = line_px + c * (tile_px + line_px)
        if idx < len(paths):
            tile = read_tile(str(paths[idx]), tile_px)
        else:
            tile = np.full((tile_px, tile_px, 3), 245, dtype=np.uint8)
        canvas[y0:y0 + tile_px, x0:x0 + tile_px] = tile
    return canvas


def draw_auroc_table(ax, metrics):
    ax.set_facecolor('white')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)
    work = metrics.set_index('detector_name').reindex(DETECTOR_NAMES).reset_index()
    rows = [[str(r.detector_name), f'{100.0 * float(r.auroc):.1f}' if np.isfinite(float(r.auroc)) else 'nan'] for r in work.itertuples(index=False)]
    table = ax.table(cellText=rows, colLabels=['method', 'AUROC'], cellLoc='center', colLoc='center', loc='center', bbox=[0.05, 0.08, 0.90, 0.84])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for row_col, cell in table.get_celld().items():
        cell.set_linewidth(0.7)
        if row_col[0] == 0:
            cell.set_text_props(weight='bold')


def sample_for_mosaic(df, n):
    if 'selection_rank' in df.columns:
        return df.sort_values('selection_rank', ascending=True).head(n).reset_index(drop=True)
    return df.head(n).reset_index(drop=True)


def draw_sample_figure(id_df, ood_df, metrics, path, title, id_title, ood_title):
    path.parent.mkdir(parents=True, exist_ok=True)
    id_sample = sample_for_mosaic(id_df, FIGURE_SAMPLE_SIZE)
    ood_sample = sample_for_mosaic(ood_df, FIGURE_SAMPLE_SIZE)
    fig = plt.figure(figsize=(7.8, 3.15), dpi=DPI, facecolor='#ffffff')
    ax_left = fig.add_axes([0.015, 0.08, 0.34, 0.80])
    ax_right = fig.add_axes([0.645, 0.08, 0.34, 0.80])
    for ax, sample, subtitle in [(ax_left, id_sample, id_title), (ax_right, ood_sample, ood_title)]:
        mosaic = make_mosaic(sample['image_path'].astype(str).tolist(), 10, 10, 26, line_px=1)
        ax.imshow(mosaic)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(subtitle, fontsize=8.5, pad=1)
    ax_table = fig.add_axes([0.345, 0.255, 0.31, 0.47], zorder=10)
    draw_auroc_table(ax_table, metrics)
    fig.text(0.5, 0.98, title, ha='center', va='top', fontsize=8.8)
    fig.savefig(path, bbox_inches='tight', facecolor='#ffffff', pad_inches=0.02)
    plt.close(fig)


def save_auroc_heatmap(df, path, title):
    if df.empty:
        return
    matrix = df.pivot_table(index='left_out_model', columns='detector_name', values='auroc', aggfunc='mean').reindex(index=MODEL_NAMES, columns=DETECTOR_NAMES)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, max(3.2, 0.52 * len(matrix.index) + 1.5)), dpi=HEATMAP_DPI)
    values = matrix.to_numpy(dtype=float)
    im = ax.imshow(values, vmin=0, vmax=1, cmap='viridis', interpolation='nearest')
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=35, ha='right', fontsize=8)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels([short_model_name(m) for m in matrix.index], fontsize=8)
    ax.set_title(title, fontsize=11)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{100.0 * v:.1f}', ha='center', va='center', fontsize=8, color='white' if v < 0.55 or v > 0.82 else 'black')
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def compute_subset_aurocs(mapping, get_df, model_name, ood_dataset, id_df, ood_df):
    rows = []
    id_ids = id_df['image_id'].astype(str).tolist()
    ood_ids = ood_df['image_id'].astype(str).tolist()
    for detector_name in DETECTOR_NAMES:
        key = detector_key(model_name, detector_name)
        id_scores = scores_for_ids(mapping, get_df, key, ID_DATASET_NAME, id_ids)
        ood_scores = scores_for_ids(mapping, get_df, key, ood_dataset, ood_ids)
        rows.append({'model_name': model_name, 'detector_name': detector_name, 'detector_key': key, **auroc_for_scores(id_scores, ood_scores)})
    return rows


def process_leave_one_model(mapping, get_df, ood_dataset):
    rows = []
    figures_dir = HARD_DIR / 'figures'
    lists_dir = HARD_DIR / 'lists'
    metrics_dir = HARD_DIR / 'metrics'

    for left_model in MODEL_NAMES:
        selection_models = [model for model in MODEL_NAMES if model != left_model]
        selection_keys = required_detector_keys(selection_models, DETECTOR_NAMES)
        id_wide = join_scores_for_dataset(mapping, get_df, ID_DATASET_NAME, selection_keys)
        ood_wide = join_scores_for_dataset(mapping, get_df, ood_dataset, selection_keys)
        id_summary = build_reference_summary(id_wide, selection_keys, True)
        ood_summary = build_reference_summary(ood_wide, selection_keys, False)
        id_hard = select_top(id_summary, SUBSET_SIZE, 'hard')
        ood_hard = select_top(ood_summary, SUBSET_SIZE, 'hard')
        metrics_rows = compute_subset_aurocs(mapping, get_df, left_model, ood_dataset, id_hard, ood_hard)
        metrics = pd.DataFrame(metrics_rows)

        for row in metrics_rows:
            row.update({
                'ood_dataset': ood_dataset,
                'left_out_model': left_model,
                'selection_models': ';'.join(selection_models),
                'selection_mode': 'hard',
                'subset_size': SUBSET_SIZE,
            })
            rows.append(row)

        prefix = f'{safe_name(ood_dataset)}__{safe_name(short_model_name(left_model))}'
        draw_sample_figure(
            id_hard,
            ood_hard,
            metrics,
            figures_dir / f'{prefix}_hard.png',
            f'{ood_dataset.upper()} | hard n={SUBSET_SIZE} | eval: {short_model_name(left_model)}',
            f'CIFAR-10 hard (100/{SUBSET_SIZE})',
            f'{ood_dataset.upper()} hard (100/{SUBSET_SIZE})',
        )
        save_path_label_txt(id_hard, lists_dir / f'{prefix}_id_hard_list.txt', None)
        save_path_label_txt(ood_hard, lists_dir / f'{prefix}_ood_hard_list.txt', -1)

    out = pd.DataFrame(rows)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(metrics_dir / f'{safe_name(ood_dataset)}_hard_metrics.csv', index=False)
    save_auroc_heatmap(out, figures_dir / f'{safe_name(ood_dataset)}_hard_auroc_heatmap.png', f'Hard AUROC | one-model-leave | {ood_dataset.upper()} | n={SUBSET_SIZE}')
    return out


def save_aggregate_heatmap(summary, path):
    if summary.empty:
        return
    work = summary.copy()
    work['label'] = work['ood_dataset'].astype(str)
    matrix = work.pivot(index='label', columns='detector_name', values='mean').reindex(columns=DETECTOR_NAMES)
    std = work.pivot(index='label', columns='detector_name', values='std').reindex(index=matrix.index, columns=matrix.columns)
    fig, ax = plt.subplots(figsize=(8.0, max(3.0, 0.48 * len(matrix.index) + 1.5)), dpi=HEATMAP_DPI)
    values = matrix.to_numpy(dtype=float)
    im = ax.imshow(values, vmin=0, vmax=1, cmap='viridis', interpolation='nearest')
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=35, ha='right', fontsize=8)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_title('AUROC mean±std over left-out models', fontsize=11)
    std_values = std.to_numpy(dtype=float)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            s = std_values[i, j]
            if np.isfinite(v):
                text = f'{100.0 * v:.1f}±{100.0 * s:.1f}' if np.isfinite(s) else f'{100.0 * v:.1f}'
                ax.text(j, i, text, ha='center', va='center', fontsize=7, color='white' if v < 0.55 or v > 0.82 else 'black')
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def run_hard_report(mapping, get_df):
    parts = []
    for ood_dataset in OOD_DATASET_NAMES:
        parts.append(process_leave_one_model(mapping, get_df, ood_dataset))
    metrics = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    metrics_dir = HARD_DIR / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_dir / 'one_model_leave_hard_metrics_all.csv', index=False)
    if not metrics.empty:
        summary = metrics.groupby(['ood_dataset', 'detector_name'], as_index=False).agg(mean=('auroc', 'mean'), std=('auroc', 'std'), min=('auroc', 'min'), max=('auroc', 'max'), n=('auroc', 'count'))
        summary.to_csv(metrics_dir / 'one_model_leave_hard_auroc_mean_std.csv', index=False)
        save_aggregate_heatmap(summary, HARD_DIR / 'figures' / 'aggregate_hard_auroc_mean_std.png')


def save_config():
    config = {
        'dataset_path': DATASET_PATH,
        'out_root': str(OUT_ROOT),
        'models': MODEL_NAMES,
        'detectors': DETECTOR_NAMES,
        'id_dataset': ID_DATASET_NAME,
        'ood_datasets': OOD_DATASET_NAMES,
        'subset_size': SUBSET_SIZE,
        'figure_sample_size': FIGURE_SAMPLE_SIZE,
        'batch_size': BATCH_SIZE,
        'knn_k': KNN_K,
        'mahalanobis_ridge': MAHALANOBIS_RIDGE,
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUT_ROOT / 'run_config.json').open('w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)


def main():
    set_seed(RANDOM_SEED)
    save_config()
    run_scoring()
    mapping, get_df = load_scores_root()
    save_shared_outputs(mapping, get_df)
    run_hard_report(mapping, get_df)


if __name__ == '__main__':
    main()
