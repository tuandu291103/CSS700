"""Append-only numerical diagnostic and single activation-boundary Grad-CAM candidate."""

import io
import json
import platform
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import yaml

from css700.experiments.som_mapping_audit import _git_state
from css700.experiments.spambase_audit import sha256, write_json
from css700.experiments.spambase_checkpoint import load_validation_checkpoint
from css700.experiments.spambase_grad_cam import GRAD_CAM_POLICY
from css700.experiments.spambase_split import verified_read
from css700.interpretability.activation_boundary import MODES, explain_boundary_validation
from css700.interpretability.spambase_grad_cam import map_diagnostics, write_examples

V2_POLICY = {
    **GRAD_CAM_POLICY,
    "candidate": "no_grad_stem_then_activation_leaf_head",
    "candidate_nnpack_enabled": True,
    "candidate_logits": "bitwise_exact_frozen_inference",
    "diagnostic_modes": list(MODES),
}
HELD_FILES = {
    "config.yaml",
    "manifest.json",
    "summary.json",
    "grad_cam_arrays.npz",
    "feature_map.json",
    "per_sample.json",
    "aggregate_features.json",
    "validation_examples.png",
}


def validate_v2_config(config):
    if (
        set(config)
        != {
            "protocol",
            "training_review",
            "training_review_sha256",
            "held_review",
            "held_review_sha256",
            "policy",
        }
        or config["protocol"] != "spambase_validation_grad_cam_v2"
        or config["policy"] != V2_POLICY
    ):
        raise ValueError("frozen v2 numerical/Grad-CAM policy changed")


def load_held_v1(config, source, hashes):
    review = yaml.safe_load(
        verified_read(Path(config["held_review"]), config["held_review_sha256"], hashes)
    )
    if (
        review.get("review_version") != "spambase_validation_grad_cam_v1_review_1"
        or review.get("acceptance") is not False
        or review.get("technical_status") != "quality_gate_failed"
        or review.get("test_evaluation_ready") is not False
        or review.get("failed_checks") != ["gradient_logits_numerically_match"]
        or set(review.get("file_sha256", {})) != HELD_FILES
        or any(review.get(k) != config[k] for k in ("training_review", "training_review_sha256"))
    ):
        raise ValueError("pinned held v1 with same accepted checkpoint required")
    root = Path(review["run_dir"])
    saved = {n: verified_read(root / n, h, hashes) for n, h in review["file_sha256"].items()}
    manifest, summary = (json.loads(saved[n]) for n in ("manifest.json", "summary.json"))
    if (
        manifest["commit_sha"] != review["execution_commit"]
        or manifest["status"] != "quality_gate_failed"
        or summary["status"] != "quality_gate_failed"
        or summary["errors"]
        or [k for k, v in summary["checks"].items() if not v] != review["failed_checks"]
        or summary["source_training_run"] != source.source_run
        or summary["test_evaluation_ready"] is not False
        or json.loads(saved["feature_map.json"]) != source.feature_map
        or manifest["output_file_sha256"]
        != {n: h for n, h in review["file_sha256"].items() if n != "manifest.json"}
    ):
        raise ValueError("held v1 review/run mismatch")
    for p, h in manifest["source_file_sha256"].items():
        verified_read(Path(p), h, hashes)
    with np.load(io.BytesIO(saved["grad_cam_arrays.npz"]), allow_pickle=False) as arrays:
        previous = {k: arrays[k].copy() for k in arrays.files}
    for key, expected in (
        ("sample_ids", source.predictions["sample_ids"]),
        ("labels", source.predictions["labels"]),
        ("target_classes", source.predictions["predictions"]),
        ("inference_logits", source.predictions["logits"]),
        ("occupied_mask", source.arrays["occupied_mask"]),
    ):
        if not np.array_equal(previous[key], expected):
            raise ValueError(f"held v1 identity mismatch: {key}")
    n = len(source.predictions["sample_ids"])
    if (
        previous["raw_maps"].shape != (n, 1, 8, 8)
        or previous["feature_scores"].shape != (n, 57)
        or previous["logits"].shape != (n, 2)
        or not all(np.isfinite(v).all() for v in previous.values())
    ):
        raise ValueError("invalid held v1 arrays")
    return previous


def run_spambase_grad_cam_v2(config_path: Path, output_dir: Path):
    started = time.perf_counter()
    commit, dirty = _git_state()
    root = output_dir / (
        datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
        + "__spambase__grid__seed-42__validation-grad-cam-v2"
    )
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "commit_sha": commit,
        "worktree_dirty_at_start": dirty,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": "validation_activation_boundary_grad_cam_only_test_closed",
        "packages": {n: version(n) for n in ("numpy", "torch", "matplotlib")},
        "platform": platform.platform(),
        "torch_build": torch.__config__.show(),
    }
    write_json(root / "manifest.json", manifest)
    hashes, checks, errors, details = {}, {"clean_worktree": not dirty}, [], {}
    try:
        if dirty:
            raise ValueError("v2 requires a clean prospective code/config commit")
        payload = config_path.read_bytes()
        hashes[str(config_path.resolve())] = sha256(payload)
        (root / "config.yaml").write_bytes(payload)
        config = yaml.safe_load(payload)
        validate_v2_config(config)
        source = load_validation_checkpoint(
            Path(config["training_review"]), config["training_review_sha256"], hashes
        )
        previous = load_held_v1(config, source, hashes)
        if not all(sha256(Path(p).read_bytes()) == h for p, h in hashes.items()):
            raise ValueError("an input changed before v2 execution")
        checks["accepted_checkpoint_and_held_v1_verified"] = True
        outputs, logits, candidate_checks, numerical = explain_boundary_validation(
            source,
            previous,
            batch_size=config["policy"]["batch_size"],
            max_seconds=config["policy"]["max_seconds"],
        )
        checks.update(candidate_checks)
        # A failed gate does not erase the candidate or diagnostic evidence.
        np.savez_compressed(
            root / "grad_cam_arrays.npz",
            **outputs,
            sample_ids=source.predictions["sample_ids"],
            labels=source.predictions["labels"],
            predicted_classes=source.predictions["predictions"],
            group_equal_weights=source.predictions["group_equal_weights"],
            occupied_mask=source.arrays["occupied_mask"],
        )
        np.savez_compressed(
            root / "numerical_logits.npz", **logits, sample_ids=source.predictions["sample_ids"]
        )
        write_json(root / "numerical_diagnostics.json", numerical)
        write_json(root / "feature_map.json", source.feature_map)
        records, aggregate, diagnostics = map_diagnostics(outputs, source)
        write_json(root / "per_sample.json", records)
        write_json(root / "aggregate_features.json", aggregate)
        example_ids = write_examples(root / "validation_examples.png", source, outputs, records)
        details.update(
            source_training_run=source.source_run,
            split_fingerprint=source.split_fingerprint,
            model_seed=42,
            checkpoint_epoch=source.checkpoint["best_epoch"],
            explained_rows=len(records),
            feature_count=57,
            diagnostics=diagnostics,
            example_sample_ids=example_ids,
            numerical=numerical,
        )
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    try:
        checks["all_inputs_unchanged"] = all(
            sha256(Path(p).read_bytes()) == h for p, h in hashes.items()
        )
    except OSError as error:
        checks["all_inputs_unchanged"] = False
        errors.append(str(error))
    passed = not errors and all(checks.values())
    summary = {
        "status": "ok" if passed else "quality_gate_failed",
        "output": str(root),
        "checks": checks,
        "errors": errors,
        **details,
        "decision": "await_independent_v2_review" if passed else "hold_v2_review",
        "test_rows_transformed_or_scored": 0,
        "test_evaluation_ready": False,
        "models_trained": 0,
        "scalers_refitted": 0,
        "optimizer_steps": 0,
        "v1_acceptance_changed": False,
        "interpretation": (
            "Numerical identity and positive spatial tracing, "
            "not faithfulness or causal importance."
        ),
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(root / "summary.json", summary)
    manifest.update(
        status=summary["status"],
        completed_at_utc=datetime.now(UTC).isoformat(),
        source_file_sha256=hashes,
        output_file_sha256={
            p.name: sha256(p.read_bytes())
            for p in sorted(root.iterdir())
            if p.is_file() and p.name != "manifest.json"
        },
    )
    write_json(root / "manifest.json", manifest)
    return summary
