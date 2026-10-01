"""Read only the accepted checkpoint and validation tensors through pinned provenance."""

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from css700.data.spambase_audit import FEATURE_NAMES
from css700.encoding import GridEncoder
from css700.experiments.spambase_grid import validate_training_config, validation_scores
from css700.experiments.spambase_split import verified_read

TRAINING_FILES = {
    "config.yaml",
    "manifest.json",
    "summary.json",
    "inputs.npz",
    "feature_map.json",
    "best_checkpoint.pt",
    "history.json",
    "validation_predictions.npz",
}


@dataclass(frozen=True)
class FrozenValidationCheckpoint:
    arrays: dict[str, np.ndarray]
    predictions: dict[str, np.ndarray]
    feature_map: list[dict]
    checkpoint: dict
    split_fingerprint: str
    source_run: str


def load_validation_checkpoint(
    path: Path, expected: str, hashes: dict
) -> FrozenValidationCheckpoint:
    review = yaml.safe_load(verified_read(path, expected, hashes))
    if (
        review.get("review_version") != "spambase_grid_train_validation_v1_review_1"
        or review.get("dataset") != "spambase"
        or review.get("technical_status") != "ok"
        or review.get("acceptance") is not True
        or review.get("acceptance_scope") != "seed_42_training_and_validation_checkpoint_only"
        or review.get("decision") != "go_to_validation_grad_cam_protocol"
        or review.get("test_evaluation_ready") is not False
        or review.get("independent_calculation", {}).get("status") != "passed"
        or set(review.get("file_sha256", {})) != TRAINING_FILES
    ):
        raise ValueError("affirmative pinned validation-only checkpoint review required")
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
        or summary["test_evaluation_ready"] is not False
        or summary["decision"] != "await_independent_training_review"
        or summary["models_completed"] != 1
        or any(
            summary[k] != 0
            for k in (
                "test_rows_transformed_or_scored",
                "scalers_refitted",
                "explanations_generated",
            )
        )
        or manifest["output_file_sha256"]
        != {n: h for n, h in review["file_sha256"].items() if n != "manifest.json"}
    ):
        raise ValueError("checkpoint review/run mismatch")
    # Hash-only upstream verification: no scaler fits or new train/validation/test encoding.
    upstream = {}
    for p, h in manifest["source_file_sha256"].items():
        verified_read(Path(p), h, upstream)
    hashes.update(upstream)
    config_bytes = verified_read(
        Path(review["training_config"]), review["training_config_sha256"], hashes
    )
    config = yaml.safe_load(config_bytes)
    validate_training_config(config)
    if (
        config_bytes != saved["config.yaml"]
        or config["preflight_review"] != review["preflight_review"]
        or config["preflight_review_sha256"] != review["preflight_review_sha256"]
        or upstream[str(Path(review["training_config"]).resolve())]
        != review["training_config_sha256"]
        or upstream[str(Path(review["preflight_review"]).resolve())]
        != review["preflight_review_sha256"]
    ):
        raise ValueError("training/preflight config identity mismatch")
    for key in (
        "model_seed",
        "epochs_completed",
        "best_epoch",
        "optimizer_steps",
        "stop_reason",
        "split_fingerprint",
    ):
        if summary[key] != review[key]:
            raise ValueError(f"checkpoint review metadata mismatch: {key}")

    with np.load(io.BytesIO(saved["inputs.npz"]), allow_pickle=False) as stored:
        if set(stored.files) != {
            "train_indices",
            "y_train",
            "train_images",
            "validation_indices",
            "y_validation",
            "validation_images",
            "validation_group_weights",
            "scaler_mean",
            "scaler_var",
            "scaler_scale",
            "occupied_mask",
        }:
            raise ValueError("unexpected training input schema")
        # Do not materialize training arrays. There are no test arrays in this source bundle.
        a = {
            n: stored[n].copy()
            for n in (
                "validation_indices",
                "y_validation",
                "validation_images",
                "validation_group_weights",
                "occupied_mask",
            )
        }
    with np.load(io.BytesIO(saved["validation_predictions.npz"]), allow_pickle=False) as stored:
        p = {n: stored[n].copy() for n in stored.files}
    if set(p) != {
        "sample_ids",
        "labels",
        "logits",
        "probabilities",
        "predictions",
        "group_equal_weights",
    }:
        raise ValueError("unexpected validation prediction schema")
    n = summary["partition_counts"]["validation"]
    if (
        a["validation_images"].shape != (n, 1, 8, 8)
        or a["validation_images"].dtype != np.float32
        or a["y_validation"].shape != (n,)
        or a["y_validation"].dtype != np.int64
        or a["validation_indices"].shape != (n,)
        or a["validation_indices"].dtype != np.int64
        or a["validation_group_weights"].shape != (n,)
        or np.any(a["validation_group_weights"] <= 0)
        or p["logits"].shape != (n, 2)
        or p["probabilities"].shape != (n, 2)
        or p["logits"].dtype != np.float32
        or p["probabilities"].dtype != np.float32
        or np.any(p["probabilities"] < 0)
        or np.any(p["probabilities"] > 1)
        or not np.allclose(p["probabilities"].sum(axis=1), 1, rtol=0, atol=1e-6)
        or p["predictions"].shape != (n,)
        or p["predictions"].dtype != np.int64
        or not np.array_equal(np.unique(a["y_validation"]), [0, 1])
        or not all(np.isfinite(v).all() for v in [*a.values(), *p.values()])
    ):
        raise ValueError("invalid validation arrays")
    for original, retained in (
        ("validation_indices", "sample_ids"),
        ("y_validation", "labels"),
        ("validation_group_weights", "group_equal_weights"),
    ):
        if not np.array_equal(a[original], p[retained]):
            raise ValueError("validation identities/labels/weights disagree")
    if not np.array_equal(p["predictions"], p["logits"].argmax(axis=1)) or not np.array_equal(
        p["predictions"], p["probabilities"].argmax(axis=1)
    ):
        raise ValueError("stored predicted targets disagree")
    metrics = validation_scores(p["labels"], p["predictions"])
    if metrics != summary["validation_primary"] or any(
        metrics[k] != review[f"validation_{k}"] for k in ("accuracy", "macro_f1")
    ):
        raise ValueError("stored validation metrics disagree")
    preflight = yaml.safe_load(
        verified_read(Path(review["preflight_review"]), review["preflight_review_sha256"], hashes)
    )
    split = yaml.safe_load(
        verified_read(Path(preflight["split_review"]), preflight["split_review_sha256"], hashes)
    )
    ids = json.loads(
        verified_read(
            Path(split["run_dir"]) / "split_row_ids.json",
            split["file_sha256"]["split_row_ids.json"],
            hashes,
        )
    )
    if not np.array_equal(a["validation_indices"], ids["validation"]):
        raise ValueError("validation IDs do not match the accepted split")
    encoder = GridEncoder(rows=8, columns=8).fit(FEATURE_NAMES)
    mapping = json.loads(saved["feature_map.json"])
    if (
        mapping != encoder.mapping_records()
        or not np.array_equal(a["occupied_mask"], encoder.padding_mask())
        or np.any(a["validation_images"][:, ~a["occupied_mask"]] != 0)
    ):
        raise ValueError("feature identities or zero input padding disagree")
    checkpoint = torch.load(
        io.BytesIO(saved["best_checkpoint.pt"]), map_location="cpu", weights_only=True
    )
    if (
        checkpoint["model_class"] != "SmallTabularImageCNN"
        or checkpoint["input_channels"] != 1
        or checkpoint["num_classes"] != 2
        or checkpoint["model_seed"] != 42
        or checkpoint["best_epoch"] != review["best_epoch"]
        or checkpoint["split_fingerprint"] != review["split_fingerprint"]
        or checkpoint["config_sha256"] != review["training_config_sha256"]
        or not all(bool(torch.isfinite(v).all()) for v in checkpoint["model_state_dict"].values())
    ):
        raise ValueError("checkpoint metadata/state mismatch")
    for values in [*a.values(), *p.values()]:
        values.setflags(write=False)
    return FrozenValidationCheckpoint(
        a, p, mapping, checkpoint, review["split_fingerprint"], str(root)
    )
