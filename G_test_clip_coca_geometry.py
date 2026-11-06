import hashlib
import json
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


SPLIT_FEATURES_ROOT = Path("./clip_coca_features")
LISTS_DIR = Path("./benchmark_splits")
OUT_DIR = Path("./clip_coca_geometry")

MODEL_CONFIGS = {
    "CLIP_ViT_H_14": ("ViT-H-14", "laion2b_s32b_b79k"),
    "CoCa_ViT_L_14": ("coca_ViT-L-14", "laion2b_s13b_b90k"),
}
SPLIT_NAMES = ["id_test_hard", "id_test_easy", "ood_hard", "ood_easy"]

SIMILARITY_BATCH_SIZE = 1024
BOOTSTRAP_REPETITIONS = 1000
RANDOM_SEED = 0


def normalize_rows(values):
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms <= 1e-12):
        raise ValueError("Image features and centroids must be finite and nonzero.")
    return values / norms


def build_train_centroids(model_name):
    model_root = SPLIT_FEATURES_ROOT / model_name
    config_path = model_root / "run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing completed extraction: {config_path}. Run F_get_clip_coca_features.py first.")
    config = json.loads(config_path.read_text())
    architecture, checkpoint = MODEL_CONFIGS[model_name]
    if (config["model_name"], config["architecture"], config["checkpoint"]) != (model_name, architecture, checkpoint):
        raise ValueError(f"Unexpected checkpoint: {config_path}")
    for split_name in SPLIT_NAMES:
        list_path = LISTS_DIR / f"{split_name}_list.txt"
        if hashlib.sha256(list_path.read_bytes()).hexdigest() != config["split_sha256"][split_name]:
            raise ValueError(f"Split changed since feature extraction: {list_path}")

    path = model_root / "train_centroids.pickle"
    frame = pd.read_pickle(path).sort_values("label")
    if frame.empty or frame["label"].tolist() != list(range(len(frame))):
        raise ValueError(f"Expected one centroid for each ImageNet class: {path}")
    counts = frame["count"].to_numpy()
    if np.any(counts <= 0) or int(counts.sum()) != config["train_sample_count"]:
        raise ValueError(f"Unexpected training counts: {path}")
    centroids = np.stack(frame["features"].to_numpy()).astype(np.float32)
    if not np.isfinite(centroids).all() or not np.allclose(np.linalg.norm(centroids, axis=1), 1, atol=1e-5):
        raise ValueError(f"Expected finite, unit-length centroids: {path}")
    return centroids, int(counts.sum()), path


def list_filename(value):
    text = str(value).replace("\\", "/")
    if text.startswith("/") and "/datasets/" in text:
        text = text.split("/datasets/", 1)[1]
    return unicodedata.normalize("NFC", text)


def load_split(model_name, split_name):
    path = SPLIT_FEATURES_ROOT / model_name / f"{split_name}.pickle"
    frame = pd.read_pickle(path)
    expected = pd.read_csv(LISTS_DIR / f"{split_name}_list.txt", header=None, names=["filename", "label"])
    expected["filename"] = expected["filename"].map(list_filename)
    expected = expected.sort_values("filename").reset_index(drop=True)
    if len(frame) != len(expected) or frame["filename"].duplicated().any():
        raise ValueError(f"Expected {len(expected)} unique images: {path}")
    if list(zip(frame["filename"], frame["label"])) != list(zip(expected["filename"], expected["label"])):
        raise ValueError(f"Cached images or labels differ from the released split: {path}")
    features = np.stack(frame["features"].to_numpy()).astype(np.float32)
    return normalize_rows(features)


def nearest_centroid_similarity(features, centroids):
    result = np.empty(len(features), dtype=np.float32)

    for start in range(0, len(features), SIMILARITY_BATCH_SIZE):
        stop = min(start + SIMILARITY_BATCH_SIZE, len(features))
        similarity = features[start:stop] @ centroids.T
        result[start:stop] = similarity.max(axis=1)

    return result


def auroc_from_similarity(id_similarity, ood_similarity):
    y = np.concatenate([
        np.zeros(len(id_similarity), dtype=int),
        np.ones(len(ood_similarity), dtype=int),
    ])
    score = -np.concatenate([id_similarity, ood_similarity])
    return float(roc_auc_score(y, score))


def bootstrap_auroc(id_similarity, ood_similarity, seed):
    rng = np.random.default_rng(seed)
    values = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)

    for i in range(BOOTSTRAP_REPETITIONS):
        id_sample = id_similarity[rng.integers(0, len(id_similarity), len(id_similarity))]
        ood_sample = ood_similarity[rng.integers(0, len(ood_similarity), len(ood_similarity))]
        values[i] = auroc_from_similarity(id_sample, ood_sample)

    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def compute_model_result(model_name, model_index):
    centroids, train_sample_count, train_path = build_train_centroids(model_name)
    features = {name: load_split(model_name, name) for name in SPLIT_NAMES}
    similarities = {
        name: nearest_centroid_similarity(values, centroids)
        for name, values in features.items()
    }

    score_frames = []
    for split_name in SPLIT_NAMES:
        path = SPLIT_FEATURES_ROOT / model_name / f"{split_name}.pickle"
        frame = pd.read_pickle(path)[["filename", "id", "label"]].copy()
        frame["split"] = split_name
        frame["centroid_similarity"] = similarities[split_name]
        frame["ood_score"] = -similarities[split_name]
        score_frames.append(frame)
    pd.concat(score_frames, ignore_index=True).to_csv(OUT_DIR / f"{model_name}_scores.csv", index=False)

    easy_auroc = auroc_from_similarity(
        similarities["id_test_easy"],
        similarities["ood_easy"],
    )
    hard_auroc = auroc_from_similarity(
        similarities["id_test_hard"],
        similarities["ood_hard"],
    )
    easy_ci_low, easy_ci_high = bootstrap_auroc(
        similarities["id_test_easy"],
        similarities["ood_easy"],
        seed=RANDOM_SEED + 2 * model_index,
    )
    hard_ci_low, hard_ci_high = bootstrap_auroc(
        similarities["id_test_hard"],
        similarities["ood_hard"],
        seed=RANDOM_SEED + 2 * model_index + 1,
    )

    return {
        "model_name": model_name,
        "train_centroid_path": str(train_path),
        "train_sample_count": train_sample_count,
        "id_easy_centroid_similarity": float(np.mean(similarities["id_test_easy"])),
        "id_hard_centroid_similarity": float(np.mean(similarities["id_test_hard"])),
        "ood_easy_centroid_similarity": float(np.mean(similarities["ood_easy"])),
        "ood_hard_centroid_similarity": float(np.mean(similarities["ood_hard"])),
        "easy_auroc": easy_auroc,
        "easy_ci_low": easy_ci_low,
        "easy_ci_high": easy_ci_high,
        "hard_auroc": hard_auroc,
        "hard_ci_low": hard_ci_low,
        "hard_ci_high": hard_ci_high,
        "auroc_drop": easy_auroc - hard_auroc,
    }


def save_plot(results):
    x = np.arange(len(results))
    width = 0.34

    easy = results["easy_auroc"].to_numpy(dtype=float)
    hard = results["hard_auroc"].to_numpy(dtype=float)
    easy_error = np.vstack([
        easy - results["easy_ci_low"].to_numpy(dtype=float),
        results["easy_ci_high"].to_numpy(dtype=float) - easy,
    ])
    hard_error = np.vstack([
        hard - results["hard_ci_low"].to_numpy(dtype=float),
        results["hard_ci_high"].to_numpy(dtype=float) - hard,
    ])

    fig, ax = plt.subplots(figsize=(6.2, 3.7), dpi=180)
    easy_bars = ax.bar(
        x - width / 2,
        easy,
        width,
        yerr=easy_error,
        capsize=3,
        label="Easy split",
    )
    hard_bars = ax.bar(
        x + width / 2,
        hard,
        width,
        yerr=hard_error,
        capsize=3,
        label="Hard split",
    )
    ax.axhline(0.5, linewidth=1.0, linestyle="--")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Nearest-centroid AUROC")
    ax.set_xticks(x)
    ax.set_xticklabels(results["model_name"], rotation=15, ha="right")
    ax.legend()
    ax.bar_label(easy_bars, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(hard_bars, fmt="%.3f", padding=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "train_centroid_geometry.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame([
        compute_model_result(model_name, model_index)
        for model_index, model_name in enumerate(MODEL_CONFIGS)
    ])
    results.to_csv(OUT_DIR / "train_centroid_geometry.csv", index=False)
    config = {
        "models": list(MODEL_CONFIGS),
        "feature_root": str(SPLIT_FEATURES_ROOT),
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seeds": {name: {"easy": RANDOM_SEED + 2 * i, "hard": RANDOM_SEED + 2 * i + 1} for i, name in enumerate(MODEL_CONFIGS)},
        "ood_score": "negative maximum cosine similarity to ImageNet training-class centroids",
        "positive_class": "OOD",
    }
    (OUT_DIR / "run_config.json").write_text(json.dumps(config, indent=2))
    save_plot(results)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
