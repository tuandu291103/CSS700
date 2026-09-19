"""Frozen Spambase partition-only milestone; no scaler, encoder, model or explanation fits."""

import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from css700.data.grouped_split import SPLIT_POLICY, allocate_groups, verify_partitions
from css700.data.spambase_audit import inspect_data
from css700.experiments.som_mapping_audit import _git_state
from css700.experiments.spambase_audit import load_pinned_source, sha256, write_json

AUDIT_FILES = {
    "manifest.json",
    "summary.json",
    "source_lock.yaml",
    "integrity_report.json",
    "feature_profiles.json",
    "duplicate_groups.json",
    "row_memberships.json",
}
PREPROCESSING_POLICY = {
    "fit_scope": "training_only_in_future_preflight",
    "scaler": "standard_per_feature_mean_and_population_sd",
    "constant_training_features": "retain_with_scale_one_and_flag",
    "imputation": "none_fail_on_invalid_values",
    "feature_selection": "none_preserve_all_57_in_source_order",
    "training_row_weights": "one_per_original_row",
}
EVALUATION_POLICY = {
    "primary_weights": "one_per_original_row",
    "sensitivity_weights": "one_over_feature_group_size_per_row",
    "uncertainty_resampling_unit": "whole_feature_group_paired_across_methods",
    "uncertainty_execution": "deferred_versioned_budget_before_test_evaluation",
}


def validate_config(config: dict) -> None:
    expected = {
        "protocol",
        "dataset",
        "source_dir",
        "source_lock",
        "source_lock_sha256",
        "readiness_audit",
        "split",
        "preprocessing",
        "evaluation",
    }
    if set(config) != expected or config["protocol"] != "spambase_grouped_split_v1":
        raise ValueError("unexpected split configuration schema/version")
    if config["dataset"] != "spambase" or config["split"] != SPLIT_POLICY:
        raise ValueError("frozen dataset/split policy changed; requires a versioned amendment")
    if config["preprocessing"] != PREPROCESSING_POLICY or config["evaluation"] != EVALUATION_POLICY:
        raise ValueError("frozen preprocessing/evaluation policy changed")
    audit = config["readiness_audit"]
    if set(audit) != {"run_dir", "file_sha256"} or set(audit["file_sha256"]) != AUDIT_FILES:
        raise ValueError("unexpected readiness audit pin schema")


def verified_read(path: Path, expected: str, hashes: dict) -> bytes:
    if path.is_symlink():
        raise ValueError(f"symlink not accepted: {path}")
    payload = path.read_bytes()
    if sha256(payload) != expected:
        raise ValueError(f"pinned input mismatch: {path}")
    hashes[str(path.resolve())] = expected
    return payload


def run_spambase_split(config_path: Path, output_dir: Path) -> dict:
    started = time.perf_counter()
    commit, dirty = _git_state()
    root = output_dir / (
        datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f") + "__spambase__grouped-split-v1"
    )
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "commit_sha": commit,
        "worktree_dirty_at_start": dirty,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "running",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scope": "partition_only_no_preprocessing_or_model_fits",
        "committee_approval_claimed": False,
    }
    write_json(root / "manifest.json", manifest)
    hashes, checks, errors, result = {}, {"clean_worktree": not dirty}, [], {}
    outputs = []
    try:
        if dirty:
            raise ValueError("split execution requires a clean frozen code/config commit")
        config_bytes = config_path.read_bytes()
        hashes[str(config_path.resolve())] = sha256(config_bytes)
        (root / "config.yaml").write_bytes(config_bytes)
        outputs.append("config.yaml")
        config = yaml.safe_load(config_bytes)
        validate_config(config)
        lock_bytes = verified_read(
            Path(config["source_lock"]), config["source_lock_sha256"], hashes
        )
        payloads, source_hashes = load_pinned_source(
            Path(config["source_dir"]), yaml.safe_load(lock_bytes)
        )
        hashes.update(source_hashes)
        audit = config["readiness_audit"]
        saved = {
            name: verified_read(Path(audit["run_dir"]) / name, expected, hashes)
            for name, expected in audit["file_sha256"].items()
        }
        prior_summary, prior_manifest = (
            json.loads(saved[name]) for name in ("summary.json", "manifest.json")
        )
        if (
            prior_summary["status"] != "ok"
            or not all(prior_summary["checks"].values())
            or prior_summary["errors"]
            or prior_manifest["status"] != "ok"
            or prior_manifest["worktree_dirty_at_start"]
            or saved["source_lock.yaml"] != lock_bytes
        ):
            raise ValueError("readiness audit/source lock not accepted")
        # Reconstruct accepted memberships from raw data, not unverified row-group annotations.
        report = inspect_data(payloads["spambase.data"], payloads["spambase.names"])
        if not all(report["schema_checks"].values()):
            raise ValueError("source integrity checks failed")
        if report != json.loads(saved["integrity_report.json"]):
            raise ValueError("recomputed integrity audit differs from accepted report")
        for name in ("row_memberships", "duplicate_groups", "feature_profiles"):
            if report[name] != json.loads(saved[f"{name}.json"]):
                raise ValueError(f"accepted audit export differs: {name}")
        checks["pinned_source_and_readiness_audit_reproduced"] = True
        members = report["row_memberships"]
        row_ids, allocations = allocate_groups(members, seed=config["split"]["seed"])
        # Preserve the first candidate even if a downstream balance check fails.
        write_json(root / "split_row_ids.json", row_ids)
        write_json(root / "group_allocations.json", allocations)
        outputs.extend(["split_row_ids.json", "group_allocations.json"])
        result = verify_partitions(members, row_ids)
        checks.update(result.pop("checks"))
        assignment = {i: name for name, ids in row_ids.items() for i in ids}
        sizes = {a["feature_vector_sha256"]: len(a["row_ids"]) for a in allocations}
        records = [
            {
                **member,
                "partition": assignment[member["row_id"]],
                "primary_weight": 1.0,
                "group_equal_sensitivity_weight": 1.0 / sizes[member["feature_vector_sha256"]],
            }
            for member in members
        ]
        write_json(root / "row_assignments.json", records)
        outputs.append("row_assignments.json")
        result["conflicting_groups"] = [a for a in allocations if a["conflicting_labels"]]
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    try:
        checks["all_inputs_unchanged"] = all(
            sha256(Path(path).read_bytes()) == expected for path, expected in hashes.items()
        )
    except OSError as error:
        checks["all_inputs_unchanged"] = False
        errors.append(str(error))
    accepted = all(checks.values()) and not errors
    summary = {
        "status": "ok" if accepted else "quality_gate_failed",
        "output": str(root),
        "checks": checks,
        "errors": errors,
        **result,
        "decision": "go_to_training_only_preflight" if accepted else "hold_split_policy_review",
        "rows_removed": 0,
        "labels_changed": 0,
        "preprocessors_fitted": 0,
        "models_trained": 0,
        "training_ready": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(root / "summary.json", summary)
    outputs.append("summary.json")
    manifest.update(
        status=summary["status"],
        source_file_sha256=hashes,
        output_file_sha256={name: sha256((root / name).read_bytes()) for name in outputs},
        completed_at_utc=datetime.now(UTC).isoformat(),
    )
    write_json(root / "manifest.json", manifest)
    return summary
