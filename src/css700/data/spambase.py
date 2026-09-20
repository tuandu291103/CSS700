"""Load only training rows after verifying the exact scientifically reviewed v2 split."""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler

from css700.data.grouped_split_v2 import verify_partitions_v2
from css700.data.spambase_audit import FEATURE_NAMES, inspect_data, parse_data
from css700.experiments.spambase_audit import load_pinned_source, sha256
from css700.experiments.spambase_split import verified_read
from css700.experiments.spambase_split_v2 import validate_config_v2

SPLIT_FILES = {
    "config.yaml",
    "manifest.json",
    "summary.json",
    "split_row_ids.json",
    "group_allocations.json",
    "row_assignments.json",
}
TARGET_NAMES = ("non-spam", "spam")


@dataclass(frozen=True)
class SpambaseTrainingSource:
    x_train_raw: np.ndarray
    y_train: np.ndarray
    train_indices: np.ndarray
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    split_fingerprint: str
    partition_counts: dict
    source_file_sha256: dict


def select_training_rows(values: np.ndarray, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Copy the specified training rows; no full-cohort statistics or preprocessing."""
    index_array = np.asarray(indices)
    if (
        values.ndim != 2
        or values.shape[1] < 2
        or index_array.ndim != 1
        or not len(index_array)
        or index_array.dtype.kind not in "iu"
        or len(np.unique(index_array)) != len(index_array)
        or np.any(index_array < 0)
        or np.any(index_array >= len(values))
    ):
        raise ValueError("valid unique integer training row IDs required")
    selected = values[index_array.astype(np.int64)]
    x, y = selected[:, :-1].copy(), selected[:, -1].astype(np.int64)
    x.setflags(write=False)
    y.setflags(write=False)
    return x, y


def load_reviewed_spambase_training(
    review_path: Path, expected_review_sha256: str, *, hashes: dict | None = None
) -> SpambaseTrainingSource:
    """Fail closed on unreviewed, rejected, mismatched or altered artifacts; never resplit."""
    hashes = {} if hashes is None else hashes
    review = yaml.safe_load(verified_read(review_path, expected_review_sha256, hashes))
    if (
        review.get("review_version") != "spambase_split_v2_scientific_review_1"
        or review.get("dataset") != "spambase"
        or review.get("scientific_acceptance") is not True
        or review.get("technical_status") != "ok"
        or review.get("acceptance_scope") != "split_structure_for_training_only_preflight"
        or review.get("decision") != "go_to_training_only_preflight"
        or review.get("v1_remains_diagnostic_only") is not True
        or review.get("independent_verification", {}).get("status") != "passed"
        or set(review.get("file_sha256", {})) != SPLIT_FILES
    ):
        raise ValueError("an affirmative, pinned v2 split review is required")
    root = Path(review["run_dir"])
    saved = {
        name: verified_read(root / name, expected, hashes)
        for name, expected in review["file_sha256"].items()
    }
    manifest, summary = (json.loads(saved[n]) for n in ("manifest.json", "summary.json"))
    if (
        manifest["status"] != "ok"
        or manifest["worktree_dirty_at_start"]
        or manifest["commit_sha"] != review["execution_commit"]
        or summary["status"] != "ok"
        or summary["errors"]
        or summary["decision"] != "await_independent_scientific_review"
        or summary["scientific_acceptance"] is not False
        or any(
            summary[k] != 0
            for k in ("rows_removed", "labels_changed", "preprocessors_fitted", "models_trained")
        )
    ):
        raise ValueError("review does not identify an accepted untouched v2 execution")
    expected_outputs = {n: h for n, h in review["file_sha256"].items() if n != "manifest.json"}
    if manifest["output_file_sha256"] != expected_outputs:
        raise ValueError("split manifest/output pins disagree")

    upstream = {}
    config_bytes = verified_read(
        Path(review["split_config"]), review["split_config_sha256"], upstream
    )
    if config_bytes != saved["config.yaml"]:
        raise ValueError("split config copy differs from reviewed config")
    config = yaml.safe_load(config_bytes)
    validate_config_v2(config)
    lock_bytes = verified_read(Path(config["source_lock"]), config["source_lock_sha256"], upstream)
    payloads, raw_hashes = load_pinned_source(
        Path(config["source_dir"]), yaml.safe_load(lock_bytes)
    )
    upstream.update(raw_hashes)
    audit = config["readiness_audit"]
    audit_bytes = {
        n: verified_read(Path(audit["run_dir"]) / n, h, upstream)
        for n, h in audit["file_sha256"].items()
    }
    hashes.update(upstream)
    if upstream != manifest["source_file_sha256"] or audit_bytes["source_lock.yaml"] != lock_bytes:
        raise ValueError("upstream source identities differ from the reviewed split")
    prior = json.loads(audit_bytes["summary.json"])
    if prior["status"] != "ok" or prior["errors"] or not all(prior["checks"].values()):
        raise ValueError("source readiness audit is not accepted")
    report = inspect_data(payloads["spambase.data"], payloads["spambase.names"])
    if not all(report["schema_checks"].values()) or report != json.loads(
        audit_bytes["integrity_report.json"]
    ):
        raise ValueError("raw source no longer reproduces the integrity audit")
    for name in ("row_memberships", "duplicate_groups", "feature_profiles"):
        if report[name] != json.loads(audit_bytes[f"{name}.json"]):
            raise ValueError(f"inconsistent readiness export: {name}")

    ids = json.loads(saved["split_row_ids.json"])
    members = report["row_memberships"]
    verification = verify_partitions_v2(members, ids)
    expected_checks = {
        "clean_worktree": True,
        "pinned_source_and_readiness_audit_reproduced": True,
        **verification["checks"],
        "all_inputs_unchanged": True,
    }
    if not all(expected_checks.values()) or summary["checks"] != expected_checks:
        raise ValueError("split gates no longer reproduce")
    for key in ("partitions", "split_fingerprint", "source_group_structure"):
        if summary[key] != verification[key]:
            raise ValueError(f"split summary mismatch: {key}")
    if review["split_fingerprint"] != verification["split_fingerprint"]:
        raise ValueError("review/split fingerprint mismatch")

    partition_by_id = {i: p for p, indices in ids.items() for i in indices}
    sizes = Counter(m["feature_vector_sha256"] for m in members)
    records = json.loads(saved["row_assignments.json"])
    expected_records = [
        {
            **m,
            "partition": partition_by_id[m["row_id"]],
            "primary_weight": 1.0,
            "group_equal_sensitivity_weight": 1.0 / sizes[m["feature_vector_sha256"]],
        }
        for m in members
    ]
    if records != expected_records:
        raise ValueError("row identity, label, group or evaluation weight mismatch")
    groups = {}
    for m in members:
        groups.setdefault(m["feature_vector_sha256"], []).append(m["row_id"])
    allocations = json.loads(saved["group_allocations.json"])
    if len(allocations) != len(groups) or {a["feature_vector_sha256"] for a in allocations} != set(
        groups
    ):
        raise ValueError("allocation groups are incomplete or repeated")
    for j, a in enumerate(allocations):
        group_ids = groups[a["feature_vector_sha256"]]
        counts = [sum(members[i]["target_class"] == c for i in group_ids) for c in (0, 1)]
        if a != {
            "allocation_order": j,
            "feature_vector_sha256": a["feature_vector_sha256"],
            "partition": partition_by_id[group_ids[0]],
            "row_ids": group_ids,
            "class_counts": counts,
            "conflicting_labels": min(counts) > 0,
        }:
            raise ValueError("allocation members/counts disagree with source")
    if summary["conflicting_groups"] != [a for a in allocations if a["conflicting_labels"]]:
        raise ValueError("conflicting-label group records disagree")
    values, issues = parse_data(payloads["spambase.data"])
    if issues:
        raise ValueError("invalid raw source values")
    x_train, y_train = select_training_rows(values, ids["train"])
    train_indices = np.asarray(ids["train"], dtype=np.int64)
    train_indices.setflags(write=False)
    if not all(sha256(Path(p).read_bytes()) == h for p, h in hashes.items()):
        raise ValueError("an input changed during loading")
    return SpambaseTrainingSource(
        x_train,
        y_train,
        train_indices,
        FEATURE_NAMES,
        TARGET_NAMES,
        verification["split_fingerprint"],
        verification["partitions"],
        dict(hashes),
    )


def fit_training_standardizer(values: np.ndarray) -> tuple[StandardScaler, np.ndarray, dict]:
    """Fit population-variance standardization on exactly the supplied training matrix."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError("nonempty finite training matrix required")
    scaler = StandardScaler().fit(x)
    transformed = scaler.transform(x)
    constant = np.ptp(x, axis=0) == 0
    numerical_guard = (~constant) & (scaler.scale_ == 1) & (scaler.var_ != 1)
    info = {
        "fit_rows": int(scaler.n_samples_seen_),
        "features": x.shape[1],
        "variance_ddof": 0,
        "constant_feature_indices": np.flatnonzero(constant).tolist(),
        "numerical_scale_guard_indices": np.flatnonzero(numerical_guard).tolist(),
    }
    return scaler, transformed, info
