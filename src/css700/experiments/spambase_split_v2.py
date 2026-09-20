"""Separate v2 execution; preserve v1 and require review beyond a technical pass."""

import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from css700.data.grouped_split import SPLIT_POLICY
from css700.data.grouped_split_v2 import SPLIT_POLICY_V2, allocate_groups_v2, verify_partitions_v2
from css700.data.spambase_audit import inspect_data
from css700.experiments.som_mapping_audit import _git_state
from css700.experiments.spambase_audit import load_pinned_source, sha256, write_json
from css700.experiments.spambase_split import validate_config as validate_v1_config
from css700.experiments.spambase_split import verified_read


def validate_config_v2(config: dict) -> None:
    if (
        config.get("protocol") != "spambase_grouped_split_v2"
        or config.get("split") != SPLIT_POLICY_V2
    ):
        raise ValueError("frozen v2 split policy changed; requires a versioned amendment")
    # All source/inclusion/preprocessing/evaluation fields and their strict schema
    # are deliberately unchanged; v1 validation is read-only on this fresh dictionary.
    validate_v1_config({**config, "protocol": "spambase_grouped_split_v1", "split": SPLIT_POLICY})


def run_spambase_split_v2(config_path: Path, output_dir: Path) -> dict:
    started = time.perf_counter()
    commit, dirty = _git_state()
    root = output_dir / (
        datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f") + "__spambase__grouped-split-v2"
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
            raise ValueError("v2 split execution requires a clean frozen code/config commit")
        config_bytes = config_path.read_bytes()
        hashes[str(config_path.resolve())] = sha256(config_bytes)
        (root / "config.yaml").write_bytes(config_bytes)
        outputs.append("config.yaml")
        config = yaml.safe_load(config_bytes)
        validate_config_v2(config)
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
        row_ids, allocations = allocate_groups_v2(members, seed=config["split"]["seed"])
        write_json(root / "split_row_ids.json", row_ids)
        write_json(root / "group_allocations.json", allocations)
        outputs.extend(["split_row_ids.json", "group_allocations.json"])
        result = verify_partitions_v2(members, row_ids)
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
        "decision": "await_independent_scientific_review"
        if accepted
        else "hold_split_policy_review",
        "scientific_acceptance": False,
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
