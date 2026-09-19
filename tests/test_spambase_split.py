"""Synthetic-only split tests. No real Spambase allocation occurs in the test suite."""

import copy
import io
import json
import zipfile
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import yaml

from css700.data.grouped_split import PARTITIONS, SPLIT_POLICY, allocate_groups, verify_partitions
from css700.data.spambase_audit import FEATURE_NAMES, duplicate_groups
from css700.experiments.spambase_audit import ARCHIVE_URL, run_spambase_audit, sha256
from css700.experiments.spambase_split import (
    AUDIT_FILES,
    EVALUATION_POLICY,
    PREPROCESSING_POLICY,
    run_spambase_split,
    validate_config,
)


def synthetic_values(n=200):
    values = np.zeros((n, 58))
    values[:, 0] = np.arange(n) / 100
    values[:, 54:57] = 1
    values[n * 3 // 5 :, -1] = 1
    values[1:5, :-1] = values[0, :-1]
    values[-1, :-1] = values[0, :-1]  # Mixed-label group, six rows.
    return values


def memberships():
    return duplicate_groups(synthetic_values())[2]


def test_determinism_order_invariance_and_unchanged_memberships():
    members = memberships()
    before = copy.deepcopy(members)
    first = allocate_groups(members)
    assert first == allocate_groups(list(reversed(members))) == allocate_groups(members)
    assert members == before
    report = verify_partitions(members, first[0])
    assert all(report["checks"].values())
    assert len([a for a in first[1] if a["conflicting_labels"]]) == 1
    assert first[1][0]["row_ids"] == [0, 1, 2, 3, 4, 199]
    # Seed sensitivity is tested only on synthetic data, never by selecting a real split.
    assert first[0] != allocate_groups(members, seed=47)[0]


def test_exact_objective_against_independent_fraction_arithmetic():
    members = memberships()
    _, allocations = allocate_groups(members)
    totals = [sum(m["target_class"] == c for m in members) for c in (0, 1)]
    counts = [[0, 0] for _ in PARTITIONS]
    fractions = [Fraction(7, 10), Fraction(3, 20), Fraction(3, 20)]

    def objective(candidate):
        score = Fraction(0)
        for j, fraction in enumerate(fractions):
            for c in (0, 1):
                target = fraction * totals[c]
                score += ((candidate[j][c] - target) / target) ** 2
            target = fraction * len(members)
            score += ((sum(candidate[j]) - target) / target) ** 2
        return score

    for allocation in allocations:
        candidates = []
        for j in range(3):
            candidate = copy.deepcopy(counts)
            for c in (0, 1):
                candidate[j][c] += allocation["class_counts"][c]
            candidates.append(candidate)
        chosen = min(range(3), key=lambda j: objective(candidates[j]))
        assert allocation["partition"] == PARTITIONS[chosen]
        counts = candidates[chosen]


@pytest.mark.parametrize("kind", ["duplicate", "omission", "fractional", "label"])
def test_bad_membership_rejected(kind):
    members = memberships()
    if kind == "duplicate":
        members[1]["row_id"] = 0
    elif kind == "omission":
        members.pop(1)
    elif kind == "fractional":
        members[1]["row_id"] = 1.5
    else:
        members[1]["target_class"] = 2
    with pytest.raises(ValueError):
        allocate_groups(members)


@pytest.mark.parametrize("kind", ["cross_group", "duplicate", "omission", "order"])
def test_partition_corruption_rejected(kind):
    members = memberships()
    ids, allocations = allocate_groups(members)
    home = allocations[0]["partition"]
    other = next(p for p in PARTITIONS if p != home)
    if kind == "cross_group":
        ids[home].remove(0)
        ids[other] = sorted(ids[other] + [0])
    elif kind == "duplicate":
        ids[other] = sorted(ids[other] + [0])
    elif kind == "omission":
        ids[home].remove(0)
    else:
        ids[home].reverse()
    with pytest.raises(ValueError):
        verify_partitions(members, ids)


def test_infeasible_groups_fail_balance_gates_without_splitting_them():
    members = memberships()
    for m in members:
        m["feature_vector_sha256"] = "one-group"
    ids, _ = allocate_groups(members)
    checks = verify_partitions(members, ids)["checks"]
    assert checks["zero_cross_partition_feature_group_overlap"]
    assert not checks["both_classes_in_every_partition"]
    assert not checks["row_fraction_tolerance"] and not checks["class_proportion_tolerance"]


def fixture_config(tmp_path, monkeypatch):
    monkeypatch.setattr("css700.experiments.spambase_audit._git_state", lambda: ("fixture", False))
    monkeypatch.setattr("css700.experiments.spambase_split._git_state", lambda: ("fixture", False))
    raw = tmp_path / "raw"
    raw.mkdir()
    values = synthetic_values(4601)
    values[:, -1] = 0
    values[2788:, -1] = 1
    payloads = {
        "spambase.data": (
            "\n".join(",".join(str(v) for v in row) for row in values) + "\n"
        ).encode(),
        "spambase.names": (
            "1, 0.\n" + "\n".join(f"{n}: continuous." for n in FEATURE_NAMES)
        ).encode(),
        "spambase.DOCUMENTATION": b"Synthetic only.",
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)
    payloads["spambase.zip"] = buffer.getvalue()
    for name, payload in payloads.items():
        (raw / name).write_bytes(payload)
    receipt = {
        "status": "ok",
        "errors": [],
        "worktree_dirty_at_start": False,
        "source_url": ARCHIVE_URL,
        "final_url": ARCHIVE_URL,
        "files": {name: {"sha256": sha256(p), "bytes": len(p)} for name, p in payloads.items()},
    }
    (raw / "receipt.json").write_text(json.dumps(receipt))
    lock = tmp_path / "source.yaml"
    lock.write_text(
        yaml.safe_dump(
            {
                "dataset": "spambase",
                "source_url": ARCHIVE_URL,
                "file_sha256": {name: sha256(p) for name, p in payloads.items()},
                "receipt_sha256": sha256((raw / "receipt.json").read_bytes()),
            }
        )
    )
    audit = run_spambase_audit(raw, lock, tmp_path / "audits")
    assert audit["status"] == "ok"
    audit_root = Path(audit["output"])
    config = {
        "protocol": "spambase_grouped_split_v1",
        "dataset": "spambase",
        "source_dir": str(raw),
        "source_lock": str(lock),
        "source_lock_sha256": sha256(lock.read_bytes()),
        "readiness_audit": {
            "run_dir": str(audit_root),
            "file_sha256": {name: sha256((audit_root / name).read_bytes()) for name in AUDIT_FILES},
        },
        "split": copy.deepcopy(SPLIT_POLICY),
        "preprocessing": PREPROCESSING_POLICY,
        "evaluation": EVALUATION_POLICY,
    }
    path = tmp_path / "split.yaml"
    path.write_text(yaml.safe_dump(config))
    return config, path


def test_frozen_config_rejects_seed_or_policy_changes():
    # Read only the tracked configuration, never its data/artifact paths.
    config = yaml.safe_load(Path("configs/spambase_split_v1.yaml").read_text())
    validate_config(config)
    config["split"]["seed"] = 7
    with pytest.raises(ValueError, match="policy"):
        validate_config(config)


@pytest.mark.parametrize("mode", ["ok", "tampered_audit", "mutated_source", "dirty"])
def test_runner_acceptance_and_retained_failures(tmp_path, monkeypatch, mode):
    import css700.experiments.spambase_split as runner

    config, path = fixture_config(tmp_path, monkeypatch)
    if mode == "tampered_audit":
        (Path(config["readiness_audit"]["run_dir"]) / "row_memberships.json").write_text("[]")
    if mode == "dirty":
        monkeypatch.setattr(runner, "_git_state", lambda: ("fixture", True))
    if mode == "mutated_source":
        original = runner.allocate_groups

        def mutate(*args, **kwargs):
            result = original(*args, **kwargs)
            (Path(config["source_dir"]) / "spambase.data").write_bytes(b"changed")
            return result

        monkeypatch.setattr(runner, "allocate_groups", mutate)
    result = run_spambase_split(path, tmp_path / "splits")
    assert result["status"] == ("ok" if mode == "ok" else "quality_gate_failed")
    assert not result["training_ready"]
    assert result["preprocessors_fitted"] == result["models_trained"] == result["rows_removed"] == 0
    assert result["labels_changed"] == 0
    root = Path(result["output"])
    assert json.loads((root / "summary.json").read_text()) == result
    if mode == "ok":
        rows = json.loads((root / "row_assignments.json").read_text())
        assert len(rows) == 4601
        assert [r["target_class"] for r in rows] == [0] * 2788 + [1] * 1813
        sums = {}
        for r in rows:
            assert r["primary_weight"] == 1
            key = r["feature_vector_sha256"]
            sums[key] = sums.get(key, 0) + r["group_equal_sensitivity_weight"]
        assert all(abs(value - 1) < 1e-12 for value in sums.values())
        manifest = json.loads((root / "manifest.json").read_text())
        assert all(
            sha256((root / n).read_bytes()) == h for n, h in manifest["output_file_sha256"].items()
        )
    if mode == "mutated_source":
        assert (root / "split_row_ids.json").exists() and not result["checks"][
            "all_inputs_unchanged"
        ]
