from pathlib import Path

import pandas as pd


SCORES_ROOT = Path('./scores')
ID_DATASET_NAME = 'test'
OOD_DATASET_NAMES = ['imagenetood', 'ninco']
OUT_DIR = Path('./ImageNet_dataset')

N_ID = 10000
N_OOD = 10000
OUTPUT_SEPARATOR = ','


def detector_key(model_name, detector_name):
    return f'{model_name}__{detector_name}'


def split_detector_key(value):
    if '__' not in value:
        return value, ''
    return tuple(value.split('__', 1))


def load_scores_root(root):
    root_path = Path(root)
    if not root_path.exists():
        raise RuntimeError(f'Scores root does not exist: {root_path.resolve()}')

    mapping = {}

    for path in root_path.rglob('*.pickle'):
        parts = path.relative_to(root_path).parts
        if len(parts) < 3:
            continue
        model_name = parts[0]
        detector_name = parts[1]
        dataset_name = path.stem
        mapping[(model_name, detector_name, dataset_name)] = path

    if not mapping:
        raise RuntimeError(f'No .pickle files found under {root_path.resolve()}')

    detectors = sorted({(model_name, detector_name) for model_name, detector_name, _ in mapping})
    dataset_names = sorted({dataset_name for _, _, dataset_name in mapping})

    if ID_DATASET_NAME not in dataset_names:
        raise RuntimeError(f"Missing ID dataset '{ID_DATASET_NAME}' in {root_path.resolve()}")

    missing_ood = [ds for ds in OOD_DATASET_NAMES if ds not in dataset_names]
    if missing_ood:
        raise RuntimeError(f'Missing OOD datasets: {missing_ood}. Available datasets: {dataset_names}')

    cache = {}

    def get_df(model_name, detector_name, dataset_name):
        key = (model_name, detector_name, dataset_name)
        if key in cache:
            return cache[key]
        if key not in mapping:
            raise KeyError(key)

        df = pd.read_pickle(mapping[key])
        required = {'image_id', 'image_path', 'ood_score', 'true_label'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f'{mapping[key]} is missing columns: {sorted(missing)}')

        df = df[['image_id', 'image_path', 'ood_score', 'true_label']].copy()
        df['image_id'] = df['image_id'].astype(str)
        df['image_path'] = df['image_path'].astype(str)
        df['ood_score'] = pd.to_numeric(df['ood_score'], errors='coerce')
        df['true_label'] = pd.to_numeric(df['true_label'], errors='coerce').fillna(-1).astype(int)
        df = df.dropna(subset=['image_id', 'ood_score'])
        df = df.drop_duplicates(subset=['image_id'], keep='first')
        cache[key] = df
        return df

    return {
        'mapping': mapping,
        'detectors': detectors,
        'get_df': get_df,
    }


def get_eligible_detector_keys(data, required_datasets):
    eligible = []

    for model_name, detector_name in data['detectors']:
        required_keys = [(model_name, detector_name, ds) for ds in required_datasets]
        if all(key in data['mapping'] for key in required_keys):
            eligible.append(detector_key(model_name, detector_name))

    return sorted(eligible)


def join_scores_for_dataset(data, dataset_name, detector_keys):
    score_parts = []
    meta_parts = []

    for key in detector_keys:
        model_name, detector_name = split_detector_key(key)
        map_key = (model_name, detector_name, dataset_name)
        if map_key not in data['mapping']:
            continue

        df = data['get_df'](model_name, detector_name, dataset_name)
        score_parts.append(df.set_index('image_id')['ood_score'].rename(key))
        meta_parts.append(df.set_index('image_id')[['image_path', 'true_label']])

    if not score_parts:
        return pd.DataFrame()

    scores = pd.concat(score_parts, axis=1)
    meta = pd.concat(meta_parts, axis=0).groupby(level=0).first()
    wide = meta.join(scores, how='inner').sort_index()
    return wide.reset_index().rename(columns={'index': 'image_id'})


def build_hard_percentiles(wide, detector_keys, is_id):
    out = pd.DataFrame(index=wide.index)

    for key in detector_keys:
        series = pd.to_numeric(wide[key], errors='coerce')
        out[key] = series.rank(pct=True, method='average', ascending=bool(is_id))

    return out


def build_reference_summary(wide, detector_keys, is_id, dataset_name):
    detector_keys = [key for key in detector_keys if key in wide.columns]
    if not detector_keys:
        return pd.DataFrame()

    hard = build_hard_percentiles(wide, detector_keys, is_id)
    easy = 1.0 - hard

    columns = ['image_id', 'image_path', 'true_label']
    if 'dataset_name' in wide.columns:
        columns.append('dataset_name')

    summary = wide[columns].copy()

    if 'dataset_name' not in summary.columns:
        summary['dataset_name'] = dataset_name

    summary['available_pair_count'] = hard.notna().sum(axis=1).astype(int)
    summary['mean_hard_all'] = hard.mean(axis=1, skipna=True)
    summary['median_hard_all'] = hard.median(axis=1, skipna=True)
    summary['hard_std_all'] = hard.std(axis=1, skipna=True)
    summary['mean_easy_all'] = easy.mean(axis=1, skipna=True)
    summary['hard_rank'] = summary['mean_hard_all'].rank(method='first', ascending=False).astype(int)
    summary['easy_rank'] = summary['mean_easy_all'].rank(method='first', ascending=False).astype(int)

    return summary.sort_values(['hard_rank', 'dataset_name', 'image_id']).reset_index(drop=True)


def build_joint_ood_wide(data, detector_keys):
    parts = []

    for dataset_name in OOD_DATASET_NAMES:
        wide = join_scores_for_dataset(data, dataset_name, detector_keys)
        if wide.empty:
            continue
        wide = wide.copy()
        wide['dataset_name'] = dataset_name
        parts.append(wide)

    if not parts:
        raise RuntimeError('OOD wide table is empty.')

    return pd.concat(parts, axis=0, ignore_index=True)


def build_joint_ood_reference_summary(joint_wide, detector_keys):
    summary = build_reference_summary(
        joint_wide,
        detector_keys,
        is_id=False,
        dataset_name='+'.join(OOD_DATASET_NAMES),
    )

    if summary.empty:
        raise RuntimeError('Joint OOD reference summary is empty.')

    summary['ood_pool_name'] = '+'.join(OOD_DATASET_NAMES)
    summary['ranking_scope'] = 'joint_ood_pool'
    return summary


def select_top(summary, k, mode):
    if summary.empty:
        raise RuntimeError('Summary dataframe is empty.')
    if len(summary) < k:
        raise RuntimeError(f'Requested top-{k}, but only {len(summary)} images are available.')

    if mode == 'hard':
        score_name = 'mean_hard_all'
    elif mode == 'easy':
        score_name = 'mean_easy_all'
    else:
        raise ValueError(f'Unknown mode: {mode}')

    subset = summary.sort_values(
        [score_name, 'hard_std_all', 'dataset_name', 'image_id'],
        ascending=[False, True, True, True],
    ).head(k).reset_index(drop=True)

    subset['selection_rank'] = range(1, len(subset) + 1)
    subset['selection_mode'] = mode
    subset['selection_score_name'] = score_name
    subset['selection_score_value'] = pd.to_numeric(subset[score_name], errors='coerce')
    return subset


def enrich_selection_with_pair_results(selection, wide, detector_keys, is_id):
    detector_keys = [key for key in detector_keys if key in wide.columns]
    join_cols = ['image_id', 'dataset_name']

    raw_scores = wide[join_cols + detector_keys].copy()
    raw_scores = raw_scores.rename(columns={key: f'score__{key}' for key in detector_keys})

    hard = build_hard_percentiles(wide, detector_keys, is_id)
    hard.columns = [f'hard_pct__{c}' for c in hard.columns]

    easy = 1.0 - build_hard_percentiles(wide, detector_keys, is_id)
    easy.columns = [f'easy_pct__{c}' for c in easy.columns]

    extra = pd.concat(
        [
            wide[join_cols].reset_index(drop=True),
            raw_scores.drop(columns=join_cols).reset_index(drop=True),
            hard.reset_index(drop=True),
            easy.reset_index(drop=True),
        ],
        axis=1,
    )

    return selection.merge(extra, on=join_cols, how='left')


def save_path_label_txt(df, out_path, force_label=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = df.copy()

    if force_label is None:
        work['label_out'] = pd.to_numeric(work['true_label'], errors='coerce').fillna(-1).astype(int)
    else:
        work['label_out'] = int(force_label)

    work['image_path'] = work['image_path'].astype(str)

    with out_path.open('w', encoding='utf-8') as f:
        for row in work[['image_path', 'label_out']].itertuples(index=False):
            f.write(f'{row.image_path}{OUTPUT_SEPARATOR}{row.label_out}\n')


def save_selection_csv(df, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_cols = [
        'selection_rank',
        'selection_mode',
        'selection_score_name',
        'selection_score_value',
        'image_id',
        'image_path',
        'true_label',
        'dataset_name',
        'available_pair_count',
        'mean_hard_all',
        'mean_easy_all',
        'median_hard_all',
        'hard_std_all',
    ]
    base_cols = [c for c in base_cols if c in df.columns]
    score_cols = sorted([c for c in df.columns if c.startswith('score__')])
    hard_cols = sorted([c for c in df.columns if c.startswith('hard_pct__')])
    easy_cols = sorted([c for c in df.columns if c.startswith('easy_pct__')])
    ordered_cols = base_cols + score_cols + hard_cols + easy_cols
    remaining_cols = [c for c in df.columns if c not in ordered_cols]

    df[ordered_cols + remaining_cols].to_csv(out_path, index=False)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_scores_root(SCORES_ROOT)
    required_datasets = [ID_DATASET_NAME] + OOD_DATASET_NAMES
    detector_keys = get_eligible_detector_keys(data, required_datasets)

    if not detector_keys:
        raise RuntimeError(f'No eligible model-detector pairs found with scores for all required datasets: {required_datasets}')

    id_wide = join_scores_for_dataset(data, ID_DATASET_NAME, detector_keys)
    if id_wide.empty:
        raise RuntimeError('ID wide table is empty.')
    id_wide = id_wide.copy()
    id_wide['dataset_name'] = ID_DATASET_NAME

    id_summary = build_reference_summary(id_wide, detector_keys, is_id=True, dataset_name=ID_DATASET_NAME)
    id_hard = select_top(id_summary, N_ID, mode='hard')
    id_easy = select_top(id_summary, N_ID, mode='easy')

    joint_ood_wide = build_joint_ood_wide(data, detector_keys)
    ood_summary = build_joint_ood_reference_summary(joint_ood_wide, detector_keys)
    ood_hard = select_top(ood_summary, N_OOD, mode='hard')
    ood_easy = select_top(ood_summary, N_OOD, mode='easy')

    id_hard = enrich_selection_with_pair_results(id_hard, id_wide, detector_keys, is_id=True)
    id_easy = enrich_selection_with_pair_results(id_easy, id_wide, detector_keys, is_id=True)
    ood_hard = enrich_selection_with_pair_results(ood_hard, joint_ood_wide, detector_keys, is_id=False)
    ood_easy = enrich_selection_with_pair_results(ood_easy, joint_ood_wide, detector_keys, is_id=False)

    save_path_label_txt(id_hard, OUT_DIR / 'id_test_hard_list.txt')
    save_path_label_txt(id_easy, OUT_DIR / 'id_test_easy_list.txt')
    save_path_label_txt(ood_hard, OUT_DIR / 'ood_hard_list.txt', force_label=-1)
    save_path_label_txt(ood_easy, OUT_DIR / 'ood_easy_list.txt', force_label=-1)

    save_selection_csv(id_hard, OUT_DIR / 'id_test_hard_selection.csv')
    save_selection_csv(id_easy, OUT_DIR / 'id_test_easy_selection.csv')
    save_selection_csv(ood_hard, OUT_DIR / 'ood_hard_selection.csv')
    save_selection_csv(ood_easy, OUT_DIR / 'ood_easy_selection.csv')


if __name__ == '__main__':
    main()
