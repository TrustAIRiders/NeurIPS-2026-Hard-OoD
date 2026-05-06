import os

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from scipy.stats import chi2_contingency, spearmanr
from sklearn.metrics import roc_auc_score


SCORES_DIR = './scores'
OUT_DIR = './shared_analysis'
ID_DATASET_NAME = 'test'
DPI = 140
MIN_SHARED_SAMPLES = 30
MAX_LABEL_LEN = 80
N_BINS = 10
P_VALUE_MIN = 1e-10
P_VALUE_MAX = 1e-1

MODEL_ORDER = [
    'MobileNet_v3_L',
    'RegNet_Y_800MF',
    'ResNet152',
    'EfficientNet_B3',
    'ConvNeXt_L',
    'EfficientNet_V2_M',
    'ViT_L_16',
    'Swin_V2_B',
    'MaxVit_T',
]

MODEL_RANK = {name: i for i, name in enumerate(MODEL_ORDER)}
META_COLUMNS = {'image_path', 'true_label'}

warnings.filterwarnings('ignore')


def safe_name(value):
    return ''.join(c if c.isalnum() or c in '._-' else '_' for c in str(value))


def short_label(value):
    value = str(value)
    if len(value) <= MAX_LABEL_LEN:
        return value
    return value[:MAX_LABEL_LEN - 3] + '...'


def detector_key(model_name, detector_name):
    return f'{model_name}__{detector_name}'


def split_detector_key(value):
    if '__' not in value:
        return value, ''
    return tuple(value.split('__', 1))


def ordered_datasets(dataset_names):
    dataset_names = [str(x) for x in dataset_names]
    other = sorted([x for x in dataset_names if x != ID_DATASET_NAME])
    if ID_DATASET_NAME in dataset_names:
        return [ID_DATASET_NAME] + other
    return other


def normalize_image_id(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    if not value or value.lower() in {'nan', 'none', '<na>', 'nat', 'null'}:
        return None
    try:
        number = float(value)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except Exception:
        pass
    return value


def load_scores():
    root = Path(SCORES_DIR)
    mapping = {}

    for path in root.rglob('*.pickle'):
        parts = path.relative_to(root).parts
        if len(parts) < 3:
            continue
        model_name = parts[0]
        detector_name = parts[1]
        dataset_name = path.stem
        mapping[(model_name, detector_name, dataset_name)] = path

    if not mapping:
        raise RuntimeError(f'No score files found in {root.resolve()}')

    detectors = sorted({(model, detector) for model, detector, _ in mapping})
    datasets = ordered_datasets({dataset for _, _, dataset in mapping})

    def get_df(model_name, detector_name, dataset_name):
        path = mapping[(model_name, detector_name, dataset_name)]
        df = pd.read_pickle(path)
        df = df[['image_id', 'image_path', 'ood_score', 'true_label']].copy()
        df['image_id'] = df['image_id'].map(normalize_image_id)
        df['image_path'] = df['image_path'].astype(str)
        df['ood_score'] = pd.to_numeric(df['ood_score'], errors='coerce')
        df['true_label'] = pd.to_numeric(df['true_label'], errors='coerce').fillna(-1).astype(int)
        df = df.dropna(subset=['image_id', 'ood_score'])
        df = df.drop_duplicates(subset=['image_id'], keep='first')
        return df

    return mapping, detectors, datasets, get_df


def compute_auroc(mapping, detectors, datasets, get_df):
    rows = []

    for model_name, detector_name in detectors:
        if (model_name, detector_name, ID_DATASET_NAME) not in mapping:
            continue

        id_df = get_df(model_name, detector_name, ID_DATASET_NAME)
        id_scores = id_df['ood_score'].to_numpy(dtype=float)
        id_scores = id_scores[np.isfinite(id_scores)]

        for dataset_name in datasets:
            if dataset_name == ID_DATASET_NAME:
                continue
            if (model_name, detector_name, dataset_name) not in mapping:
                continue

            ood_df = get_df(model_name, detector_name, dataset_name)
            ood_scores = ood_df['ood_score'].to_numpy(dtype=float)
            ood_scores = ood_scores[np.isfinite(ood_scores)]

            if len(id_scores) < 2 or len(ood_scores) < 2:
                continue

            y = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
            s = np.concatenate([id_scores, ood_scores])
            auroc = float(roc_auc_score(y, s))

            rows.append({
                'dataset_name': dataset_name,
                'model_name': model_name,
                'detector_name': detector_name,
                'detector_key': detector_key(model_name, detector_name),
                'auroc': auroc,
                'n_id': int(len(id_scores)),
                'n_ood': int(len(ood_scores)),
            })

    return pd.DataFrame(rows)


def join_dataset_scores(mapping, detectors, datasets, get_df):
    joined = {}

    for dataset_name in datasets:
        score_parts = []
        meta_parts = []

        for model_name, detector_name in detectors:
            if (model_name, detector_name, dataset_name) not in mapping:
                continue

            df = get_df(model_name, detector_name, dataset_name)
            key = detector_key(model_name, detector_name)
            score_parts.append(df.set_index('image_id')['ood_score'].rename(key))
            meta_parts.append(df.set_index('image_id')[['image_path', 'true_label']])

        if not score_parts:
            continue

        scores = pd.concat(score_parts, axis=1)
        meta = pd.concat(meta_parts, axis=0).groupby(level=0).first()
        joined[dataset_name] = meta.join(scores, how='outer').sort_index()

    return joined


def available_keys(wide):
    return [c for c in wide.columns if c not in META_COLUMNS]


def ordered_keys_for_dataset(metrics, dataset_name, keys):
    keys = set(keys)

    if metrics.empty:
        rows = pd.DataFrame({'detector_key': list(keys)})
        rows['model_name'] = rows['detector_key'].map(lambda x: split_detector_key(x)[0])
        rows['detector_name'] = rows['detector_key'].map(lambda x: split_detector_key(x)[1])
        rows['auroc'] = 0.0
    elif dataset_name == ID_DATASET_NAME:
        rows = metrics.groupby(['model_name', 'detector_name', 'detector_key'], as_index=False).agg(auroc=('auroc', 'mean'))
    else:
        rows = metrics.loc[metrics['dataset_name'] == dataset_name].copy()

    rows = rows.loc[rows['detector_key'].isin(keys)].copy()
    rows['model_rank'] = rows['model_name'].map(lambda x: MODEL_RANK.get(str(x), len(MODEL_RANK)))
    rows = rows.sort_values(['model_rank', 'model_name', 'auroc', 'detector_name'], ascending=[True, True, False, True])
    return rows['detector_key'].astype(str).tolist()


def label_map(metrics, dataset_name, keys):
    labels = {}

    if metrics.empty:
        for key in keys:
            model_name, detector_name = split_detector_key(key)
            labels[key] = f'{model_name} + {detector_name}'
        return labels

    if dataset_name == ID_DATASET_NAME:
        rows = metrics.groupby('detector_key', as_index=False).agg(auroc=('auroc', 'mean'))
    else:
        rows = metrics.loc[metrics['dataset_name'] == dataset_name, ['detector_key', 'auroc']]

    aurocs = {str(r.detector_key): float(r.auroc) for r in rows.itertuples(index=False)}

    for key in keys:
        model_name, detector_name = split_detector_key(key)
        if key in aurocs and np.isfinite(aurocs[key]):
            labels[key] = f'{model_name} + {detector_name} ({aurocs[key]:.3f})'
        else:
            labels[key] = f'{model_name} + {detector_name}'

    return labels


def hard_percentiles(wide, dataset_name, keys):
    out = pd.DataFrame(index=wide.index)
    ascending = dataset_name == ID_DATASET_NAME

    for key in keys:
        values = pd.to_numeric(wide[key], errors='coerce')
        out[key] = values.rank(pct=True, method='average', ascending=ascending)

    return out


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


def save_spearman_png(path, dataset_name, keys, labels, rho):
    n = len(keys)
    size = max(10, min(30, 0.35 * n + 6))
    fig, ax = plt.subplots(figsize=(size, size), dpi=DPI)
    im = ax.imshow(rho, cmap='coolwarm', vmin=-1, vmax=1, interpolation='nearest')
    shown_labels = [short_label(labels[key]) for key in keys]

    ax.set_title(f'{dataset_name} | Spearman')
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(shown_labels, rotation=90, fontsize=5)
    ax.set_yticklabels(shown_labels, fontsize=5)
    color_ticks(ax, keys, 'x')
    color_ticks(ax, keys, 'y')
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    plt.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def save_pvalue_png(path, dataset_name, keys, labels, agreement, p_values, counts):
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

    size = max(10, min(30, 0.35 * n + 6))
    fig, ax = plt.subplots(figsize=(size, size), dpi=DPI)
    cmap = plt.get_cmap('RdYlGn_r').copy()
    cmap.set_bad(color='lightgray')
    im = ax.imshow(data, cmap=cmap, norm=LogNorm(vmin=P_VALUE_MIN, vmax=P_VALUE_MAX), interpolation='nearest')
    shown_labels = [short_label(labels[key]) for key in keys]

    ax.set_title(f'Binned Agreement ({N_BINS} bins) | color = p-value | dataset={dataset_name}')
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(shown_labels, rotation=90, fontsize=5)
    ax.set_yticklabels(shown_labels, fontsize=5)
    color_ticks(ax, keys, 'x')
    color_ticks(ax, keys, 'y')

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label('p-value (log scale)')
    plt.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def save_difficulty_png(path, dataset_name, keys, labels, hard):
    mean = hard.mean(axis=1, skipna=True)
    image_order = mean.sort_values(ascending=False).index
    matrix = hard.loc[image_order, keys].T
    labels_ordered = [short_label(labels[key]) for key in keys]

    width = max(10, min(32, 0.02 * matrix.shape[1] + 10))
    height = max(8, min(50, 0.28 * matrix.shape[0] + 3))
    cmap = 'RdYlGn' if dataset_name == ID_DATASET_NAME else 'RdYlGn_r'

    fig, ax = plt.subplots(figsize=(width, height), dpi=DPI)
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest')
    ax.set_title(f'{dataset_name} | difficulty heatmap')
    ax.set_xticks([])
    ax.set_yticks(np.arange(len(keys)))
    ax.set_yticklabels(labels_ordered, fontsize=5)
    color_ticks(ax, keys, 'y')
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    plt.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def save_dataset_outputs(dataset_name, wide, metrics, out_dir):
    keys = ordered_keys_for_dataset(metrics, dataset_name, available_keys(wide))
    if len(keys) < 2:
        return

    labels = label_map(metrics, dataset_name, keys)
    hard = hard_percentiles(wide, dataset_name, keys)
    rho = compute_spearman(hard)
    agreement, p_values, counts = compute_binned_agreement(hard)
    name = safe_name(dataset_name)

    save_spearman_png(out_dir / f'{name}_sperman.png', dataset_name, keys, labels, rho)
    save_pvalue_png(out_dir / f'{name}_pvalue.png', dataset_name, keys, labels, agreement, p_values, counts)
    save_difficulty_png(out_dir / f'{name}_heatmap.png', dataset_name, keys, labels, hard)


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping, detectors, datasets, get_df = load_scores()
    metrics = compute_auroc(mapping, detectors, datasets, get_df)
    joined = join_dataset_scores(mapping, detectors, datasets, get_df)

    metrics.to_csv(out_dir / 'auroc_raw.csv', index=False)

    for dataset_name in ordered_datasets(joined.keys()):
        save_dataset_outputs(dataset_name, joined[dataset_name], metrics, out_dir)


if __name__ == '__main__':
    main()
