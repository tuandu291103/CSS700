"""Verified frozen Spambase inputs for training/validation only; no test tensors."""

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from css700.data.spambase import load_reviewed_spambase_training
from css700.data.spambase_audit import parse_data
from css700.encoding import GridEncoder
from css700.experiments.spambase_preflight import PREFLIGHT_POLICY
from css700.experiments.spambase_split import verified_read

PREFLIGHT_FILES = {
    "config.yaml",
    "manifest.json",
    "summary.json",
    "training_arrays.npz",
    "feature_map.json",
    "training_feature_checks.json",
    "cpu_smoke.json",
}


@dataclass(frozen=True)
class SpambaseGridInputs:
    arrays: dict[str, np.ndarray]
    feature_map: list[dict]
    split_fingerprint: str
    partition_counts: dict[str, int]


def load_spambase_grid_inputs(review_path: Path, expected: str, hashes: dict) -> SpambaseGridInputs:
    """Reuse reviewed scaling verbatim; transform validation only after identity checks."""
    review = yaml.safe_load(verified_read(review_path, expected, hashes))
    if (
        review.get("review_version") != "spambase_training_preflight_v1_review_1"
        or review.get("dataset") != "spambase"
        or review.get("technical_status") != "ok"
        or review.get("acceptance") is not True
        or review.get("acceptance_scope") != "training_preprocessing_grid_cpu_compatibility_only"
        or review.get("decision") != "go_to_frozen_grid_training_protocol"
        or review.get("training_protocol_ready") is not True
        or review.get("test_evaluation_ready") is not False
        or review.get("independent_calculation", {}).get("status") != "passed"
        or set(review.get("file_sha256", {})) != PREFLIGHT_FILES
    ):
        raise ValueError("affirmative pinned training-only preflight review required")
    root = Path(review["run_dir"])
    saved = {n: verified_read(root / n, h, hashes) for n, h in review["file_sha256"].items()}
    manifest, summary = (json.loads(saved[n]) for n in ("manifest.json", "summary.json"))
    if (
        manifest["status"] != "ok"
        or manifest["worktree_dirty_at_start"]
        or manifest["commit_sha"] != review["execution_commit"]
        or summary["status"] != "ok"
        or summary["errors"]
        or not all(summary["checks"].values())
        or summary["decision"] != "go_to_frozen_grid_training_protocol"
        or summary["test_evaluation_ready"] is not False
        or any(
            summary[k] != 0
            for k in ("models_trained", "optimizer_steps", "heldout_rows_transformed_or_encoded")
        )
        or manifest["output_file_sha256"]
        != {n: h for n, h in review["file_sha256"].items() if n != "manifest.json"}
    ):
        raise ValueError("preflight run/review mismatch")
    upstream = {}
    config_bytes = verified_read(
        Path(review["preflight_config"]), review["preflight_config_sha256"], upstream
    )
    config = yaml.safe_load(config_bytes)
    if (
        config_bytes != saved["config.yaml"]
        or config["policy"] != PREFLIGHT_POLICY
        or config["review_path"] != review["split_review"]
        or config["review_sha256"] != review["split_review_sha256"]
    ):
        raise ValueError("preflight config or split identity mismatch")
    source = load_reviewed_spambase_training(
        Path(review["split_review"]), review["split_review_sha256"], hashes=upstream
    )
    hashes.update(upstream)
    if upstream != manifest["source_file_sha256"]:
        raise ValueError("preflight upstream pins changed")
    if not source.split_fingerprint == review["split_fingerprint"] == summary["split_fingerprint"]:
        raise ValueError("preflight fingerprint mismatch")

    with np.load(io.BytesIO(saved["training_arrays.npz"]), allow_pickle=False) as stored:
        arrays = {n: stored[n].copy() for n in stored.files}
    required = {
        "train_indices",
        "y_train",
        "x_standardized",
        "images",
        "occupied_mask",
        "scaler_mean",
        "scaler_var",
        "scaler_scale",
    }
    if set(arrays) != required:
        raise ValueError("preflight arrays schema mismatch")
    for key in ("scaler_mean", "scaler_var", "scaler_scale"):
        if arrays[key].shape != (57,) or not np.isfinite(arrays[key]).all():
            raise ValueError("invalid frozen scaler")
    if np.any(arrays["scaler_scale"] <= 0):
        raise ValueError("frozen scaler requires positive scales")
    if (
        not np.array_equal(arrays["train_indices"], source.train_indices)
        or not np.array_equal(arrays["y_train"], source.y_train)
        or not np.allclose(
            arrays["scaler_mean"], source.x_train_raw.mean(axis=0), rtol=1e-10, atol=1e-12
        )
        or not np.allclose(
            arrays["scaler_var"], source.x_train_raw.var(axis=0), rtol=1e-10, atol=1e-12
        )
        or not np.array_equal(
            arrays["x_standardized"],
            (source.x_train_raw - arrays["scaler_mean"]) / arrays["scaler_scale"],
        )
    ):
        raise ValueError("frozen training rows/scaler do not reproduce")
    encoder = GridEncoder(rows=8, columns=8, padding_value=0).fit(source.feature_names)
    mapping = json.loads(saved["feature_map.json"])
    if (
        mapping != encoder.mapping_records()
        or not np.array_equal(arrays["occupied_mask"], encoder.padding_mask())
        or not np.array_equal(arrays["images"], encoder.transform(arrays["x_standardized"]))
        or arrays["images"].dtype != np.float32
    ):
        raise ValueError("frozen Grid does not reproduce")

    # Full-cohort parsing is integrity/provenance work. Only validation is selected below.
    split_review = yaml.safe_load(
        verified_read(Path(review["split_review"]), review["split_review_sha256"], hashes)
    )
    split_root = Path(split_review["run_dir"])
    ids = json.loads(
        verified_read(
            split_root / "split_row_ids.json",
            split_review["file_sha256"]["split_row_ids.json"],
            hashes,
        )
    )
    rows = json.loads(
        verified_read(
            split_root / "row_assignments.json",
            split_review["file_sha256"]["row_assignments.json"],
            hashes,
        )
    )
    raw_paths = [p for p in upstream if Path(p).name == "spambase.data"]
    if len(raw_paths) != 1:
        raise ValueError("ambiguous raw source")
    values, issues = parse_data(verified_read(Path(raw_paths[0]), upstream[raw_paths[0]], hashes))
    if issues:
        raise ValueError("invalid raw source")
    validation_ids = np.asarray(ids["validation"], dtype=np.int64)
    validation_x = (values[validation_ids, :-1] - arrays["scaler_mean"]) / arrays["scaler_scale"]
    result_arrays = {
        "train_indices": arrays["train_indices"],
        "y_train": arrays["y_train"],
        "train_images": arrays["images"],
        "validation_indices": validation_ids,
        "y_validation": values[validation_ids, -1].astype(np.int64),
        "validation_images": encoder.transform(validation_x),
        "validation_group_weights": np.asarray(
            [rows[i]["group_equal_sensitivity_weight"] for i in validation_ids]
        ),
        "scaler_mean": arrays["scaler_mean"],
        "scaler_var": arrays["scaler_var"],
        "scaler_scale": arrays["scaler_scale"],
        "occupied_mask": arrays["occupied_mask"],
    }
    if not all(np.isfinite(a).all() for a in result_arrays.values()):
        raise ValueError("nonfinite model input")
    return SpambaseGridInputs(
        result_arrays,
        mapping,
        source.split_fingerprint,
        {p: d["rows"] for p, d in source.partition_counts.items()},
    )
