"""Append-only predicted-class Grad-CAM milestone with the Spambase test gate closed."""

import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml

from css700.experiments.som_mapping_audit import _git_state
from css700.experiments.spambase_audit import sha256, write_json
from css700.experiments.spambase_checkpoint import load_validation_checkpoint
from css700.interpretability.spambase_grad_cam import (
    explain_validation,
    map_diagnostics,
    write_examples,
)

GRAD_CAM_POLICY = {
    "split": "validation",
    "samples": "all_accepted_validation_rows_in_source_order",
    "targets": "frozen_predicted_class_logits",
    "layer": "features.2_pre_relu",
    "sign": "relu_positive_evidence",
    "normalization": "display_only_full_grid_max_clamp_1e-12",
    "batch_size": 32,
    "device": "cpu",
    "threads": 1,
    "max_seconds": 120.0,
    "padding": "exclude_from_feature_scores_report_raw_cam_mass",
    "ranking": "descending_raw_stable_feature_index_ties_positive_only_top_lists",
    "aggregate": "row_primary_and_group_equal_by_predicted_class",
    "examples": "first_source_row_per_true_predicted_pair_00_11_01_10",
    "independent_reference_rtol": 1e-5,
    "independent_reference_atol": 1e-7,
    "gradient_logit_rtol": 1e-5,
    "gradient_logit_atol": 1e-6,
    "test_evaluation": False,
    "refit_or_retrain": False,
}


def validate_grad_cam_config(config):
    if (
        set(config) != {"protocol", "training_review", "training_review_sha256", "policy"}
        or config["protocol"] != "spambase_validation_grad_cam_v1"
        or config["policy"] != GRAD_CAM_POLICY
    ):
        raise ValueError("frozen validation Grad-CAM policy changed")


def run_spambase_grad_cam(config_path: Path, output_dir: Path) -> dict:
    started = time.perf_counter()
    commit, dirty = _git_state()
    root = output_dir / (
        datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
        + "__spambase__grid__seed-42__validation-grad-cam-v1"
    )
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "commit_sha": commit,
        "worktree_dirty_at_start": dirty,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": "validation_grad_cam_tracing_only_test_closed",
        "packages": {n: version(n) for n in ("numpy", "torch", "matplotlib")},
    }
    write_json(root / "manifest.json", manifest)
    hashes, checks, errors, details = {}, {"clean_worktree": not dirty}, [], {}
    try:
        if dirty:
            raise ValueError("Grad-CAM requires a clean prospective code/config commit")
        payload = config_path.read_bytes()
        hashes[str(config_path.resolve())] = sha256(payload)
        (root / "config.yaml").write_bytes(payload)
        config = yaml.safe_load(payload)
        validate_grad_cam_config(config)
        source = load_validation_checkpoint(
            Path(config["training_review"]), config["training_review_sha256"], hashes
        )
        if not all(sha256(Path(p).read_bytes()) == h for p, h in hashes.items()):
            raise ValueError("an input changed before explanation")
        checks["accepted_checkpoint_and_validation_source_verified"] = True
        outputs, result_checks = explain_validation(
            source,
            batch_size=config["policy"]["batch_size"],
            max_seconds=config["policy"]["max_seconds"],
            gradient_logit_rtol=config["policy"]["gradient_logit_rtol"],
            gradient_logit_atol=config["policy"]["gradient_logit_atol"],
        )
        checks.update(result_checks)
        # Save even a numerical/identity gate failure; do not silently discard diagnostics.
        np.savez_compressed(
            root / "grad_cam_arrays.npz",
            **outputs,
            sample_ids=source.predictions["sample_ids"],
            labels=source.predictions["labels"],
            predicted_classes=source.predictions["predictions"],
            group_equal_weights=source.predictions["group_equal_weights"],
            occupied_mask=source.arrays["occupied_mask"],
        )
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
            target_names=["non-spam", "spam"],
            diagnostics=diagnostics,
            example_sample_ids=example_ids,
            gradient_vs_frozen_max_logit_difference=float(
                np.max(np.abs(outputs["logits"] - source.predictions["logits"]))
            ),
            top_5_by_predicted_class={
                k: [r for r in v["row_primary"] if r["mean_raw_score"] > 0][:5]
                for k, v in aggregate.items()
                if k != "all_predicted_classes"
            },
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
    accepted = not errors and all(checks.values())
    summary = {
        "status": "ok" if accepted else "quality_gate_failed",
        "output": str(root),
        "checks": checks,
        "errors": errors,
        **details,
        "decision": "await_independent_grad_cam_review" if accepted else "hold_grad_cam_review",
        "test_rows_transformed_or_scored": 0,
        "test_evaluation_ready": False,
        "models_trained": 0,
        "scalers_refitted": 0,
        "optimizer_steps": 0,
        "interpretation": (
            "Spatial positive-evidence tracing, not causal feature importance "
            "or a faithfulness test."
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
