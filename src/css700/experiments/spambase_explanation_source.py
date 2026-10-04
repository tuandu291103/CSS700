"""Pinned accepted Grad-CAM source plus training-only deletion references and group IDs."""

import io
import json
from pathlib import Path

import numpy as np
import yaml

from css700.experiments.spambase_checkpoint import load_validation_checkpoint
from css700.experiments.spambase_split import verified_read
from css700.interpretability.activation_boundary import bitwise_equal

CAM_FILES = {
    "aggregate_features.json",
    "config.yaml",
    "feature_map.json",
    "grad_cam_arrays.npz",
    "manifest.json",
    "numerical_diagnostics.json",
    "numerical_logits.npz",
    "per_sample.json",
    "summary.json",
    "validation_examples.png",
}


def load_deletion_source(path, expected, hashes):
    review = yaml.safe_load(verified_read(Path(path), expected, hashes))
    if (
        review.get("review_version") != "spambase_validation_grad_cam_v2_review_1"
        or review.get("acceptance") is not True
        or review.get("acceptance_scope") != "seed_42_validation_grad_cam_tracing_only"
        or review.get("technical_status") != "ok"
        or review.get("decision") != "go_to_validation_faithfulness_protocol"
        or review.get("test_evaluation_ready") is not False
        or review.get("independent_calculation", {}).get("status") != "passed"
        or set(review.get("file_sha256", {})) != CAM_FILES
    ):
        raise ValueError("affirmative pinned validation Grad-CAM v2 review required")
    source = load_validation_checkpoint(
        Path(review["training_review"]), review["training_review_sha256"], hashes
    )
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
        or summary["source_training_run"] != source.source_run
        or summary["test_evaluation_ready"] is not False
        or manifest["output_file_sha256"]
        != {n: h for n, h in review["file_sha256"].items() if n != "manifest.json"}
        or json.loads(saved["feature_map.json"]) != source.feature_map
    ):
        raise ValueError("Grad-CAM acceptance/run identity mismatch")
    for p, h in manifest["source_file_sha256"].items():
        verified_read(Path(p), h, hashes)
    with np.load(io.BytesIO(saved["grad_cam_arrays.npz"]), allow_pickle=False) as arrays:
        cam = {k: arrays[k].copy() for k in arrays.files}
    for k, expected_values in (
        ("sample_ids", source.predictions["sample_ids"]),
        ("labels", source.predictions["labels"]),
        ("target_classes", source.predictions["predictions"]),
        ("logits", source.predictions["logits"]),
        ("inference_logits", source.predictions["logits"]),
        ("group_equal_weights", source.predictions["group_equal_weights"]),
        ("occupied_mask", source.arrays["occupied_mask"]),
    ):
        if not bitwise_equal(cam[k], expected_values):
            raise ValueError(f"Grad-CAM array identity mismatch: {k}")
    scores = cam["feature_scores"]
    if (
        scores.shape != (len(cam["sample_ids"]), 57)
        or not np.isfinite(scores).all()
        or np.any(scores < 0)
    ):
        raise ValueError("invalid raw Grad-CAM scores")
    coordinates = np.empty(57, dtype=np.int64)
    for record in source.feature_map:
        k = record["feature_index"]
        coordinates[k] = record["row"] * 8 + record["column"]
        if not np.array_equal(
            scores[:, k], cam["raw_maps"][:, record["channel"], record["row"], record["column"]]
        ):
            raise ValueError("feature scores do not trace raw maps")
    input_path = Path(source.source_run) / "inputs.npz"
    with np.load(
        io.BytesIO(verified_read(input_path, hashes[str(input_path.resolve())], hashes)),
        allow_pickle=False,
    ) as arrays:
        train = arrays["train_images"]
        train_ids = arrays["train_indices"]
    if (
        train.dtype != np.float32
        or train.shape != (len(train_ids), 1, 8, 8)
        or not np.isfinite(train).all()
    ):
        raise ValueError("invalid saved training images")
    if np.any(train[:, ~source.arrays["occupied_mask"]] != 0):
        raise ValueError("training padding is not zero")
    replacements = np.stack([np.zeros((1, 8, 8), dtype=np.float32), np.median(train, axis=0)])
    row_files = [Path(p) for p in hashes if Path(p).name == "row_assignments.json"]
    if len(row_files) != 1:
        raise ValueError("one pinned accepted split row-assignment source required")
    rows = json.loads(verified_read(row_files[0], hashes[str(row_files[0])], hashes))
    selected = {int(r["row_id"]): r for r in rows if r["partition"] == "validation"}
    ids, labels = source.predictions["sample_ids"], source.predictions["labels"]
    if set(selected) != set(ids.tolist()):
        raise ValueError("validation groups do not cover frozen IDs")
    groups = np.array([selected[int(i)]["group_id"] for i in ids])
    _, codes, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if not np.array_equal(1 / counts[codes], source.predictions["group_equal_weights"]):
        raise ValueError("validation group weights disagree")
    if not np.array_equal([selected[int(i)]["target_class"] for i in ids], labels):
        raise ValueError("validation group labels disagree")
    return source, scores, coordinates, replacements, groups
