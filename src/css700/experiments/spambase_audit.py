"""Append-only official-source acquisition and offline pre-split integrity audit."""

import hashlib
import io
import json
import platform
import time
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from css700.data.spambase_audit import inspect_data
from css700.experiments.som_mapping_audit import _git_state

ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/94/spambase.zip"
MEMBERS = ("spambase.data", "spambase.names", "spambase.DOCUMENTATION")
RAW_FILES = ("spambase.zip", *MEMBERS)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def archive_members(payload: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if sorted(archive.namelist()) != sorted(MEMBERS):
            raise ValueError("archive members changed, duplicated or contain unexpected paths")
        if any(r.file_size > 5_000_000 for r in archive.infolist()):
            raise ValueError("source member exceeds bounded download size")
        return {name: archive.read(name) for name in MEMBERS}


def fetch_spambase(output_dir: Path) -> dict:
    commit, dirty = _git_state()
    root = output_dir / (datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f") + "__spambase__uci-94")
    root.mkdir(parents=True, exist_ok=False)
    receipt = {
        "source_url": ARCHIVE_URL,
        "commit_sha": commit,
        "worktree_dirty_at_start": dirty,
        "accessed_at_utc": datetime.now(UTC).isoformat(),
        "status": "running",
        "files": {},
        "dataset_url": "https://archive.ics.uci.edu/dataset/94/spambase",
        "license_reference": "CC BY 4.0, as listed by UCI; see dataset_url",
        "citation": "Hopkins, Reeber, Forman & Suermondt (1999), Spambase, doi:10.24432/C53G6X",
        "scope": "acquisition_only_no_data_analysis",
    }
    errors = []
    try:
        if dirty:
            raise ValueError("official source acquisition requires a clean code commit")
        with urllib.request.urlopen(ARCHIVE_URL, timeout=30) as response:
            if response.status != 200 or response.geturl() != ARCHIVE_URL:
                raise ValueError("unexpected source response or redirect")
            payload = response.read(5_000_001)
            receipt.update(
                final_url=response.geturl(),
                content_type=response.headers.get("Content-Type"),
                etag=response.headers.get("ETag"),
                last_modified=response.headers.get("Last-Modified"),
            )
        # Preserve returned bytes even if validation fails; never execute archive content.
        (root / "spambase.zip").write_bytes(payload)
        receipt["files"]["spambase.zip"] = {"sha256": sha256(payload), "bytes": len(payload)}
        if len(payload) > 5_000_000:
            raise ValueError("archive exceeds download limit")
        for name, content in archive_members(payload).items():
            (root / name).write_bytes(content)
            receipt["files"][name] = {"sha256": sha256(content), "bytes": len(content)}
        receipt["status"] = "ok"
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
        receipt["status"] = "acquisition_failed"
    receipt.update(errors=errors, completed_at_utc=datetime.now(UTC).isoformat())
    write_json(root / "receipt.json", receipt)
    return {
        "status": receipt["status"],
        "output": str(root),
        "files": receipt["files"],
        "errors": errors,
    }


def load_pinned_source(root: Path, lock: dict) -> tuple[dict, dict]:
    if set(lock) != {"dataset", "source_url", "receipt_sha256", "file_sha256"} or (
        lock["dataset"] != "spambase"
        or lock["source_url"] != ARCHIVE_URL
        or set(lock["file_sha256"]) != set(RAW_FILES)
    ):
        raise ValueError("unexpected Spambase source-lock schema")
    hashes, payloads = {}, {}
    for name, expected in {**lock["file_sha256"], "receipt.json": lock["receipt_sha256"]}.items():
        path = root / name
        if path.is_symlink():
            raise ValueError("source symlinks are not accepted")
        content = path.read_bytes()
        if sha256(content) != expected:
            raise ValueError(f"pinned source hash mismatch: {name}")
        hashes[str(path.resolve())] = expected
        payloads[name] = content
    receipt = json.loads(payloads["receipt.json"])
    if (
        receipt["status"] != "ok"
        or receipt["errors"]
        or receipt["worktree_dirty_at_start"]
        or receipt["source_url"] != ARCHIVE_URL
        or receipt["final_url"] != ARCHIVE_URL
        or set(receipt["files"]) != set(RAW_FILES)
    ):
        raise ValueError("source acquisition was not accepted")
    for name in RAW_FILES:
        if receipt["files"][name] != {
            "sha256": lock["file_sha256"][name],
            "bytes": len(payloads[name]),
        }:
            raise ValueError("receipt/file identity mismatch")
    extracted = archive_members(payloads["spambase.zip"])
    if any(extracted[name] != payloads[name] for name in MEMBERS):
        raise ValueError("retained source bytes differ from pinned archive members")
    return payloads, hashes


def run_spambase_audit(source_dir: Path, lock_path: Path, output_dir: Path) -> dict:
    started = time.perf_counter()
    commit, dirty = _git_state()
    root = output_dir / (
        datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f") + "__spambase__readiness-audit"
    )
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "commit_sha": commit,
        "worktree_dirty_at_start": dirty,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "running",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scope": "whole_release_pre_split_integrity_only",
        "source_dir": str(source_dir.resolve()),
        "committee_approval_claimed": False,
    }
    write_json(root / "manifest.json", manifest)
    checks, hashes, errors = {"clean_worktree": not dirty}, {}, []
    report = {}
    try:
        if dirty:
            raise ValueError("readiness audit requires a clean source commit")
        payload = lock_path.read_bytes()
        hashes[str(lock_path.resolve())] = sha256(payload)
        (root / "source_lock.yaml").write_bytes(payload)
        payloads, source_hashes = load_pinned_source(source_dir, yaml.safe_load(payload))
        hashes.update(source_hashes)
        checks["pinned_source_and_receipt_verified"] = True
        report = inspect_data(payloads["spambase.data"], payloads["spambase.names"])
        checks.update(report["schema_checks"])
        write_json(root / "integrity_report.json", report)
        write_json(root / "duplicate_groups.json", report["duplicate_groups"])
        write_json(root / "row_memberships.json", report["row_memberships"])
        write_json(root / "feature_profiles.json", report["feature_profiles"])
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    try:
        checks["source_files_unchanged"] = all(
            sha256(Path(p).read_bytes()) == h for p, h in hashes.items()
        )
    except OSError as error:
        checks["source_files_unchanged"] = False
        errors.append(str(error))
    accepted = all(checks.values()) and not errors
    summary = {
        "status": "ok" if accepted else "quality_gate_failed",
        "output": str(root),
        "checks": checks,
        "errors": errors,
        "rows": report.get("rows"),
        "features": report.get("features"),
        "class_counts": report.get("class_counts"),
        "parse_issue_counts": report.get("parse_issue_counts"),
        "duplicates": report.get("duplicates"),
        "decision": "go_to_split_policy_review" if accepted else "hold_source_integrity_review",
        "training_ready": False,
        "rows_removed": 0,
        "splits_created": 0,
        "models_trained": 0,
        "runtime_seconds": time.perf_counter() - started,
        "interpretation": (
            "Integrity audit only; no source identity, label repair or committee approval inferred."
        ),
    }
    manifest.update(
        status=summary["status"],
        source_file_sha256=hashes,
        completed_at_utc=datetime.now(UTC).isoformat(),
    )
    write_json(root / "manifest.json", manifest)
    write_json(root / "summary.json", summary)
    return summary
