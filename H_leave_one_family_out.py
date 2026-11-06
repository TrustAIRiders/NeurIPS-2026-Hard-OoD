import json
import os
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


SCORES_ROOT = Path("./scores")
OUT_DIR = Path("./leave_one_family_out")

EVALUATION_MODELS = {"ConvNeXt_L", "ViT_L_16"}
SELECTION_MODELS = {
    "EfficientNet_B3", "EfficientNet_V2_M", "MaxVit_T", "MobileNet_v3_L",
    "RegNet_Y_800MF", "ResNet152", "Swin_V2_B",
}

SUBSET_SIZE = 10_000

DATASETS = ("test", "ninco", "imagenetood")
OOD_DATASETS = ("ninco", "imagenetood")

FAMILIES = {
    "output": {
        "MSP", "MaxSoftmax", "TemperatureScaling", "KLMatching",
        "Entropy", "MaxLogit", "Energy", "ODIN",
    },
    "feature": {"KNN", "Mahalanobis", "RMD", "ViM", "SHE"},
    "activation": {"ReAct", "DICE", "ASH"},
}


def canonical_path(value):
    return unicodedata.normalize("NFC", str(value)).replace("\\", "/")


def canonical_sample_id(value, dataset_name):
    text = canonical_path(value)

    if dataset_name == "test":
        return f"test::{os.path.basename(text)}"

    lower = text.lower()
    markers = {
        "ninco": "ninco/",
        "imagenetood": "imagenetood/",
    }
    marker = markers[dataset_name]
    position = lower.find(marker)
    relative = text[position:] if position >= 0 else text
    return f"{dataset_name}::{relative}"


def detector_family(detector_name):
    for family_name, members in FAMILIES.items():
        if detector_name in members:
            return family_name
    return None


def iter_score_files():
    for path in sorted(SCORES_ROOT.rglob("*.pickle")):
        parts = path.relative_to(SCORES_ROOT).parts
        if len(parts) < 3:
            continue

        model_name = parts[0]
        detector_name = parts[1]
        dataset_name = path.stem
        if dataset_name not in DATASETS:
            continue

        yield path, model_name, detector_name, dataset_name


def discover_inventory():
    records = []
    for path, model_name, detector_name, dataset_name in iter_score_files():
        records.append({
            "path": str(path),
            "model_name": model_name,
            "detector_name": detector_name,
            "family": detector_family(detector_name),
            "dataset_name": dataset_name,
        })

    if not records:
        raise RuntimeError(f"No score files found under {SCORES_ROOT}")

    return pd.DataFrame(records)


def resolve_model_sets(inventory):
    selection = set(SELECTION_MODELS)
    evaluation = set(EVALUATION_MODELS)
    if not selection or not evaluation:
        raise ValueError("Selection and evaluation model sets must be nonempty.")
    overlap = selection & evaluation
    if overlap:
        raise ValueError(f"Selection and evaluation models overlap: {sorted(overlap)}")
    missing = (selection | evaluation) - set(inventory["model_name"])
    if missing:
        raise RuntimeError(f"Missing configured models: {sorted(missing)}")

    used = inventory[
        inventory["model_name"].isin(selection | evaluation)
        & inventory["family"].notna()
    ]
    for model_name in sorted(selection | evaluation):
        present = set(used.loc[used["model_name"] == model_name, "family"])
        missing = set(FAMILIES) - present
        if missing:
            raise RuntimeError(f"Missing detector families for {model_name}: {sorted(missing)}")
    for (model_name, detector_name), group in used.groupby(["model_name", "detector_name"]):
        missing = set(DATASETS) - set(group["dataset_name"])
        if missing:
            raise RuntimeError(f"Missing datasets for {model_name}/{detector_name}: {sorted(missing)}")
        if group["dataset_name"].duplicated().any():
            raise RuntimeError(f"Duplicate score files for {model_name}/{detector_name}")
    return selection, evaluation


def read_score_file(path, dataset_name):
    frame = pd.read_pickle(path)
    required = {"image_path", "ood_score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    work = frame[["image_path", "ood_score"]].copy()
    work["ood_score"] = pd.to_numeric(work["ood_score"], errors="coerce")
    work = work.dropna(subset=["ood_score"])
    work["sample_id"] = work["image_path"].map(
        lambda value: canonical_sample_id(value, dataset_name)
    )
    return work.drop_duplicates("sample_id")[["sample_id", "ood_score"]]


def score_to_hardness(scores, dataset_name):
    ascending = dataset_name == "test"
    return scores.rank(pct=True, method="average", ascending=ascending)


def load_selection_pair_rows(inventory, selection_models):
    selected = inventory[
        inventory["model_name"].isin(selection_models)
        & inventory["family"].notna()
    ]

    parts = []
    for row in selected.itertuples(index=False):
        scores = read_score_file(Path(row.path), row.dataset_name)
        scores["hardness"] = score_to_hardness(scores["ood_score"], row.dataset_name)
        scores["dataset_name"] = row.dataset_name
        scores["model_name"] = row.model_name
        scores["detector_name"] = row.detector_name
        scores["family"] = row.family
        parts.append(scores[[
            "sample_id", "dataset_name", "model_name", "detector_name",
            "family", "hardness",
        ]])

    if not parts:
        raise RuntimeError("No recognized detector scores found for selection models")

    return pd.concat(parts, ignore_index=True)


def build_family_hardness(pair_rows):
    # Equal detector weighting within a model-family block.
    model_family = pair_rows.groupby(
        ["dataset_name", "sample_id", "family", "model_name"],
        as_index=False,
    ).agg(
        model_family_hardness=("hardness", "mean"),
        detector_count=("detector_name", "nunique"),
    )

    expected_models = (
        model_family.groupby(["dataset_name", "family"])["model_name"]
        .nunique()
        .rename("expected_model_count")
        .reset_index()
    )

    # Equal model weighting within each family.
    family_sample = model_family.groupby(
        ["dataset_name", "sample_id", "family"],
        as_index=False,
    ).agg(
        family_hardness=("model_family_hardness", "mean"),
        family_hardness_std=("model_family_hardness", "std"),
        model_count=("model_name", "nunique"),
    )
    family_sample = family_sample.merge(
        expected_models,
        on=["dataset_name", "family"],
        how="left",
        validate="many_to_one",
    )
    family_sample["model_coverage"] = (
        family_sample["model_count"] / family_sample["expected_model_count"]
    )
    family_sample = family_sample[
        family_sample["model_count"] == family_sample["expected_model_count"]
    ].copy()

    return family_sample


def aggregate_selection(family_sample, included_families):
    included_families = tuple(sorted(included_families))
    subset = family_sample[family_sample["family"].isin(included_families)].copy()

    wide = subset.pivot_table(
        index=["dataset_name", "sample_id"],
        columns="family",
        values="family_hardness",
        aggfunc="first",
    )

    missing_families = set(included_families) - set(wide.columns)
    if missing_families:
        raise RuntimeError(
            f"No selection scores for families: {sorted(missing_families)}"
        )

    wide = wide.dropna(subset=list(included_families)).reset_index()
    wide["aggregate_hardness"] = wide[list(included_families)].mean(axis=1)
    wide["family_disagreement"] = wide[list(included_families)].std(axis=1, ddof=0)
    wide["selection_families"] = "+".join(included_families)
    return wide


def select_tail(frame, count, hardest):
    if len(frame) < 2 * count:
        raise RuntimeError(
            f"Need at least {2 * count} eligible samples, found {len(frame)}"
        )

    return (
        frame.sort_values(
            ["aggregate_hardness", "sample_id"],
            ascending=[not hardest, True],
            kind="mergesort",
        )
        .head(count)
        .copy()
    )


def make_splits(selection, subset_size):
    id_pool = selection[selection["dataset_name"] == "test"]
    ood_pool = selection[selection["dataset_name"].isin(OOD_DATASETS)]

    selected = {
        "id_hard": select_tail(id_pool, subset_size, hardest=True),
        "id_easy": select_tail(id_pool, subset_size, hardest=False),
        "ood_hard": select_tail(ood_pool, subset_size, hardest=True),
        "ood_easy": select_tail(ood_pool, subset_size, hardest=False),
    }

    for split_name, frame in selected.items():
        frame["split"] = split_name

    return selected


def evaluate_detector_pair(score_paths, selected_ids):
    score_parts = {}
    for dataset_name, path in score_paths.items():
        score_parts[dataset_name] = read_score_file(path, dataset_name).set_index("sample_id")

    results = {}
    for level in ("hard", "easy"):
        id_ids = selected_ids[f"id_{level}"]
        ood_ids = selected_ids[f"ood_{level}"]

        id_scores = score_parts["test"].reindex(id_ids)["ood_score"].dropna()
        ood_scores = pd.concat([
            score_parts[name].reindex(ood_ids)["ood_score"]
            for name in OOD_DATASETS
        ]).dropna()
        ood_scores = ood_scores[~ood_scores.index.duplicated(keep="first")]

        id_coverage = len(id_scores) / len(id_ids)
        ood_coverage = len(ood_scores) / len(ood_ids)
        if len(id_scores) != len(id_ids) or len(ood_scores) != len(ood_ids):
            raise RuntimeError(
                f"Incomplete {level} evaluation: ID={len(id_scores)}/{len(id_ids)}, "
                f"OOD={len(ood_scores)}/{len(ood_ids)}. Check files: {score_paths}"
            )

        y = np.concatenate([
            np.zeros(len(id_scores), dtype=int),
            np.ones(len(ood_scores), dtype=int),
        ])
        score = np.concatenate([
            id_scores.to_numpy(dtype=float),
            ood_scores.to_numpy(dtype=float),
        ])

        results[f"{level}_auroc"] = float(roc_auc_score(y, score))
        results[f"{level}_id_count"] = int(len(id_scores))
        results[f"{level}_ood_count"] = int(len(ood_scores))
        results[f"{level}_id_coverage"] = float(id_coverage)
        results[f"{level}_ood_coverage"] = float(ood_coverage)

    return results


def evaluate_ood_detection(inventory, evaluation_models, held_out_family, splits):
    selected_ids = {
        name: frame["sample_id"].tolist()
        for name, frame in splits.items()
    }

    candidates = inventory[
        inventory["model_name"].isin(evaluation_models)
        & (inventory["family"] == held_out_family)
    ]

    rows = []
    for (model_name, detector_name), group in candidates.groupby(
        ["model_name", "detector_name"]
    ):
        paths = {
            row.dataset_name: Path(row.path)
            for row in group.itertuples(index=False)
        }
        if not set(DATASETS).issubset(paths):
            raise RuntimeError(f"Missing evaluation datasets for {model_name}/{detector_name}")

        result = evaluate_detector_pair(paths, selected_ids)
        if result is None:
            continue

        rows.append({
            "held_out_family": held_out_family,
            "model_name": model_name,
            "detector_name": detector_name,
            **result,
            "auroc_drop": result["easy_auroc"] - result["hard_auroc"],
        })

    return pd.DataFrame(rows)


def load_evaluation_hardness(inventory, evaluation_models, held_out_family):
    selected = inventory[
        inventory["model_name"].isin(evaluation_models)
        & (inventory["family"] == held_out_family)
    ]

    parts = []
    for row in selected.itertuples(index=False):
        scores = read_score_file(Path(row.path), row.dataset_name)
        scores["hardness"] = score_to_hardness(scores["ood_score"], row.dataset_name)
        scores["dataset_name"] = row.dataset_name
        scores["model_name"] = row.model_name
        scores["detector_name"] = row.detector_name
        parts.append(scores[[
            "sample_id", "dataset_name", "model_name", "detector_name", "hardness",
        ]])

    if not parts:
        return pd.DataFrame()

    pair_rows = pd.concat(parts, ignore_index=True)
    return pair_rows.groupby(
        ["dataset_name", "sample_id", "model_name"],
        as_index=False,
    ).agg(
        heldout_family_hardness=("hardness", "mean"),
        detector_count=("detector_name", "nunique"),
    )


def evaluate_rank_transfer(inventory, evaluation_models, held_out_family, splits):
    hardness = load_evaluation_hardness(
        inventory, evaluation_models, held_out_family
    )
    if hardness.empty:
        return pd.DataFrame()

    split_lookup = pd.concat([
        frame[["sample_id"]].assign(split=split_name)
        for split_name, frame in splits.items()
    ], ignore_index=True)
    hardness = hardness.merge(split_lookup, on="sample_id", how="inner")

    rows = []
    for model_name, model_frame in hardness.groupby("model_name"):
        for group_name, hard_split, easy_split in [
            ("id", "id_hard", "id_easy"),
            ("ood", "ood_hard", "ood_easy"),
        ]:
            group = model_frame[model_frame["split"].isin([hard_split, easy_split])]
            hard_count = int((group["split"] == hard_split).sum())
            easy_count = int((group["split"] == easy_split).sum())
            expected = len(splits[hard_split]) + len(splits[easy_split])
            coverage = len(group) / expected
            if hard_count < 2 or easy_count < 2 or len(group) != expected:
                continue

            y = (group["split"] == hard_split).astype(int).to_numpy()
            score = group["heldout_family_hardness"].to_numpy(dtype=float)
            rows.append({
                "held_out_family": held_out_family,
                "model_name": model_name,
                "group": group_name,
                "hard_vs_easy_auroc": float(roc_auc_score(y, score)),
                "hard_count": hard_count,
                "easy_count": easy_count,
                "coverage": float(coverage),
            })

    return pd.DataFrame(rows)


def set_overlap(left, right):
    left = set(left)
    right = set(right)
    intersection = len(left & right)
    union = len(left | right)
    return {
        "intersection": intersection,
        "jaccard": float(intersection / union) if union else np.nan,
        "left_retention": float(intersection / len(left)) if left else np.nan,
        "right_retention": float(intersection / len(right)) if right else np.nan,
    }


def compare_scenarios(scenarios):
    rows = []
    names = sorted(scenarios)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1:]:
            for split_name in ("id_hard", "id_easy", "ood_hard", "ood_easy"):
                overlap = set_overlap(
                    scenarios[left_name][split_name]["sample_id"],
                    scenarios[right_name][split_name]["sample_id"],
                )
                rows.append({
                    "scenario_a": left_name,
                    "scenario_b": right_name,
                    "split": split_name,
                    **overlap,
                })
    return pd.DataFrame(rows)


def summarize_ood_results(results):
    if results.empty:
        return pd.DataFrame()

    return results.groupby("held_out_family", as_index=False).agg(
        model_detector_pairs=("hard_auroc", "count"),
        mean_hard_auroc=("hard_auroc", "mean"),
        std_hard_auroc=("hard_auroc", "std"),
        median_hard_auroc=("hard_auroc", "median"),
        maximum_hard_auroc=("hard_auroc", "max"),
        fraction_hard_below_05=("hard_auroc", lambda x: float(np.mean(np.asarray(x) < 0.5))),
        mean_easy_auroc=("easy_auroc", "mean"),
        mean_auroc_drop=("auroc_drop", "mean"),
    )


def summarize_rank_results(results):
    if results.empty:
        return pd.DataFrame()

    return results.groupby(["held_out_family", "group"], as_index=False).agg(
        evaluation_models=("hard_vs_easy_auroc", "count"),
        mean_hard_vs_easy_auroc=("hard_vs_easy_auroc", "mean"),
        std_hard_vs_easy_auroc=("hard_vs_easy_auroc", "std"),
        minimum_hard_vs_easy_auroc=("hard_vs_easy_auroc", "min"),
    )


def save_selected_samples(scenarios):
    parts = []
    for scenario_name, splits in scenarios.items():
        for split_name, frame in splits.items():
            work = frame.copy()
            work["scenario"] = scenario_name
            work["split"] = split_name
            parts.append(work)
    pd.concat(parts, ignore_index=True).to_csv(
        OUT_DIR / "selected_samples.csv", index=False
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory = discover_inventory()
    selection_models, evaluation_models = resolve_model_sets(inventory)
    inventory[
        inventory["model_name"].isin(selection_models | evaluation_models)
        & inventory["family"].notna()
    ].to_csv(OUT_DIR / "used_score_files.csv", index=False)

    unknown = inventory[inventory["family"].isna()][
        ["model_name", "detector_name", "dataset_name"]
    ].drop_duplicates()
    unknown.to_csv(OUT_DIR / "skipped_unknown_detectors.csv", index=False)

    pair_rows = load_selection_pair_rows(inventory, selection_models)
    family_sample = build_family_hardness(pair_rows)

    all_families = tuple(sorted(FAMILIES))
    scenarios = {}

    full_selection = aggregate_selection(family_sample, all_families)
    scenarios["all_families"] = make_splits(full_selection, SUBSET_SIZE)

    ood_results = []
    rank_results = []
    for held_out_family in all_families:
        included = set(all_families) - {held_out_family}
        selection = aggregate_selection(family_sample, included)
        splits = make_splits(selection, SUBSET_SIZE)
        scenario_name = f"without_{held_out_family}"
        scenarios[scenario_name] = splits

        family_ood = evaluate_ood_detection(
            inventory, evaluation_models, held_out_family, splits
        )
        if not family_ood.empty:
            ood_results.append(family_ood)

        family_rank = evaluate_rank_transfer(
            inventory, evaluation_models, held_out_family, splits
        )
        if not family_rank.empty:
            rank_results.append(family_rank)

    ood_results = (
        pd.concat(ood_results, ignore_index=True)
        if ood_results else pd.DataFrame()
    )
    rank_results = (
        pd.concat(rank_results, ignore_index=True)
        if rank_results else pd.DataFrame()
    )

    ood_results.to_csv(OUT_DIR / "heldout_family_ood_auroc.csv", index=False)
    summarize_ood_results(ood_results).to_csv(
        OUT_DIR / "heldout_family_ood_summary.csv", index=False
    )
    rank_results.to_csv(OUT_DIR / "heldout_family_rank_transfer.csv", index=False)
    summarize_rank_results(rank_results).to_csv(
        OUT_DIR / "heldout_family_rank_summary.csv", index=False
    )

    compare_scenarios(scenarios).to_csv(
        OUT_DIR / "selection_pairwise_overlap.csv", index=False
    )
    save_selected_samples(scenarios)

    config = {
        "scores_root": str(SCORES_ROOT),
        "output_dir": str(OUT_DIR),
        "selection_models": sorted(selection_models),
        "evaluation_models": sorted(evaluation_models),
        "subset_size": SUBSET_SIZE,
        "minimum_model_coverage": 1.0,
        "minimum_evaluation_coverage": 1.0,
        "score_higher_is_ood": True,
        "families": {name: sorted(values) for name, values in FAMILIES.items()},
        "overlap_reference": (
            "all_families: a new reference selection with equal family weights "
            "on the selection models, not the released benchmark lists"
        ),
        "selection_weighting": (
            "equal detector weight within model-family, equal model weight "
            "within family, equal family weight across included families"
        ),
    }
    (OUT_DIR / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print("Selection models:", sorted(selection_models))
    print("Evaluation models:", sorted(evaluation_models))
    print("Unknown detectors skipped:", sorted(unknown["detector_name"].unique()))
    print()
    print("OOD transfer summary")
    print(summarize_ood_results(ood_results).to_string(index=False))
    print()
    print("Ranking transfer summary")
    print(summarize_rank_results(rank_results).to_string(index=False))


if __name__ == "__main__":
    main()
