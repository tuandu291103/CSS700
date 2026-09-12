"""Strict, identity-preserving readers for the frozen TreeSHAP pilot inputs."""

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from css700.config import resolve_experiment_seeds
from css700.data import load_breast_cancer_splits
from css700.data.splits import split_fingerprint


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prediction_arrays(rows: list[dict], sample_ids: np.ndarray, labels: np.ndarray) -> dict:
    if [int(row["sample_id"]) for row in rows] != sample_ids.tolist():
        raise ValueError("prediction sample IDs are missing, duplicated or reordered")
    if [int(row["true_label"]) for row in rows] != labels.tolist():
        raise ValueError("prediction true labels do not match source split")
    predictions = np.array([int(row["predicted_label"]) for row in rows])
    probabilities = np.array(
        [[float(row[f"probability_class_{c}"]) for c in (0, 1)] for row in rows]
    )
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
    ) or not np.array_equal(probabilities.argmax(axis=1), predictions):
        raise ValueError("invalid saved predictions")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("invalid saved probability sum")
    return {"predictions": predictions, "probabilities": probabilities}


def named_feature_matrix(
    rows: list[dict],
    sample_ids: np.ndarray,
    feature_names: tuple[str, ...],
    predictions: np.ndarray,
    labels: np.ndarray,
    *,
    value_field: str,
) -> np.ndarray:
    positions = {int(value): index for index, value in enumerate(sample_ids)}
    matrix = np.full((len(sample_ids), len(feature_names)), np.nan)
    for row in rows:
        sample_id, feature = int(row["sample_id"]), int(row["feature_index"])
        if sample_id not in positions or not 0 <= feature < len(feature_names):
            raise ValueError("invalid attribution sample or feature index")
        position = positions[sample_id]
        if row["feature_name"] != feature_names[feature]:
            raise ValueError("attribution feature name/index mismatch")
        if (
            any(
                int(row[key]) != predictions[position]
                for key in ("target_class", "predicted_label")
            )
            or int(row["true_label"]) != labels[position]
        ):
            raise ValueError("attribution target or label mismatch")
        if not np.isnan(matrix[position, feature]):
            raise ValueError("duplicate attribution row")
        value = float(row[value_field])
        if not np.isfinite(value):
            raise ValueError("nonfinite attribution")
        matrix[position, feature] = value
    if not np.isfinite(matrix).all():
        raise ValueError("missing attribution rows")
    return matrix


def load_sources(config: dict) -> dict[str, Any]:
    model_seed = config["model_seed"]
    if (
        type(model_seed) is not int
        or model_seed not in (7, 42, 2026)
        or config["data_split_seed"] != 42
    ):
        raise ValueError("registered model seeds are 7, 42, 2026; split seed must remain 42")
    bundle = load_breast_cancer_splits(seed=42)
    if tuple(bundle.target_names) != ("malignant", "benign"):
        raise ValueError("unexpected dataset class names")
    fingerprint = split_fingerprint(
        bundle.train_indices, bundle.validation_indices, bundle.test_indices
    )
    if fingerprint != config["expected_split_sha256"]:
        raise ValueError("split fingerprint mismatch")
    hashes: dict[str, str] = {}

    def source(root: Path, relative: str) -> Path:
        path = (root / relative).resolve()
        hashes[str(path)] = file_sha256(path)
        return path

    def read_json(root: Path, relative: str) -> dict:
        return json.loads(source(root, relative).read_text())

    def read_csv(root: Path, relative: str) -> list[dict]:
        with source(root, relative).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    tree_root, grid_root = Path(config["tree_run"]), Path(config["grid_run"])
    for root in (tree_root, grid_root):
        with np.load(source(root, "splits/indices.npz"), allow_pickle=False) as splits:
            for split in ("train", "validation", "test"):
                if not np.array_equal(splits[split], getattr(bundle, f"{split}_indices")):
                    raise ValueError("source split/order mismatch")
        preprocessing = read_json(root, "preprocessing/standard_scaler.json")
        if preprocessing["fit_split"] != "training" or preprocessing["method"] != "StandardScaler":
            raise ValueError("source scaler was not training-only StandardScaler")
        if tuple(preprocessing["feature_names"]) != bundle.feature_names:
            raise ValueError("source feature order mismatch")
        for key in ("mean", "scale"):
            if not np.array_equal(preprocessing[key], getattr(bundle, f"scaler_{key}")):
                raise ValueError("source scaler parameters mismatch")
    tree_manifest = read_json(tree_root, "manifest.json")
    original_config = source(Path("."), config["tree_training_config"])
    stored_config = source(tree_root, "config.yaml")
    if tree_manifest["config_sha256"] != file_sha256(original_config) or yaml.safe_load(
        original_config.read_text()
    ) != yaml.safe_load(stored_config.read_text()):
        raise ValueError("source tree config hash mismatch")
    if not all(read_json(tree_root, "metrics/summary.json")["technical_checks"].values()):
        raise ValueError("source tree quality gate failed")
    grid_config = yaml.safe_load(source(grid_root, "config.yaml").read_text())
    seeds = resolve_experiment_seeds(grid_config)
    if seeds.model_seed != model_seed or seeds.data_split_seed != 42:
        raise ValueError("source Grid seeds mismatch")
    grid_manifest = read_json(grid_root, "manifest.json")
    predictive = read_json(grid_root, "metrics/predictive_metrics.json")
    if not all(
        predictive["quality_gate"][name]
        for name in (
            "beats_majority_accuracy",
            "beats_majority_macro_f1",
            "checkpoint_reload_exact_within_1e-7",
            "working_tree_clean_at_start",
        )
    ):
        raise ValueError("source Grid predictive quality gate failed")
    tree_rows = read_csv(tree_root, "metrics/test_predictions.csv")
    tree_predictions, model_paths = {}, {}
    for family in ("random_forest", "xgboost"):
        model_paths[family] = source(tree_root, f"models/{family}__seed-{model_seed}.joblib")
        if hashes[str(model_paths[family])] != config["model_sha256"][family]:
            raise ValueError("frozen model hash mismatch; do not deserialize")
        rows = [
            r for r in tree_rows if r["model_name"] == family and int(r["model_seed"]) == model_seed
        ]
        tree_predictions[family] = prediction_arrays(rows, bundle.test_indices, bundle.y_test)
    grid_predictions = prediction_arrays(
        read_csv(grid_root, "model/test_predictions.csv"),
        bundle.test_indices,
        bundle.y_test,
    )
    scores = {}
    for name, filename, field in (
        ("grad_cam", "grad_cam", "raw_attribution"),
        ("integrated_gradients_zero", "integrated_gradients", "signed_attribution"),
        ("occlusion_zero", "occlusion", "signed_target_logit_drop"),
    ):
        if not all(read_json(grid_root, f"metrics/{filename}_summary.json")["checks"].values()):
            raise ValueError(f"source {name} quality gate failed")
        rows = read_csv(grid_root, f"attributions/{filename}_features.csv")
        if filename == "integrated_gradients":
            rows = [row for row in rows if row["baseline"] == "zero"]
        scores[name] = named_feature_matrix(
            rows,
            bundle.test_indices,
            bundle.feature_names,
            grid_predictions["predictions"],
            bundle.y_test,
            value_field=field,
        )
    return {
        "bundle": bundle,
        "file_hashes": hashes,
        "tree_manifest": tree_manifest,
        "grid_manifest": grid_manifest,
        "model_paths": model_paths,
        "tree_predictions": tree_predictions,
        "grid_predictions": grid_predictions,
        "grid_scores": scores,
    }
