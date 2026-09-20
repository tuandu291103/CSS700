"""Synthetic-only prospective v2 validation. Never allocate the actual Spambase cohort."""

import copy
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import test_spambase_split as v1_tests
import yaml

from css700.data.grouped_split import PARTITIONS, allocate_groups
from css700.data.grouped_split_v2 import SPLIT_POLICY_V2, allocate_groups_v2, verify_partitions_v2
from css700.data.spambase_audit import duplicate_groups
from css700.experiments.spambase_audit import sha256
from css700.experiments.spambase_split_v2 import run_spambase_split_v2, validate_config_v2


def mixed_values(n=2000, sizes=None, correlated=False):
    sizes = sizes if sizes is not None else [25, 15, 10, 7, 5, 3, 2] * 8
    if sum(sizes) > n:
        raise ValueError("synthetic group sizes exceed corpus")
    sizes = [*sizes, *([1] * (n - sum(sizes)))]
    values = np.zeros((n, 58))
    values[:, 54:57] = 1
    cursor = 0
    for j, size in enumerate(sizes):
        values[cursor : cursor + size, 0] = j / 100
        values[cursor : cursor + size, -1] = j % 2
        if size > 1 and j % 11 == 0:
            values[cursor, -1] = 1 - values[cursor, -1]
        cursor += size
    if correlated:
        values[:, -1] = 0
        values[n * 3 // 5 :, -1] = 1
    return values


@pytest.mark.parametrize("correlated", [False, True])
def test_mixed_sizes_balance_reproducibility_and_preservation(correlated):
    members = duplicate_groups(mixed_values(correlated=correlated))[2]
    original = copy.deepcopy(members)
    ids, allocations = allocate_groups_v2(members)
    assert (ids, allocations) == allocate_groups_v2(list(reversed(members)))
    assert members == original
    report = verify_partitions_v2(members, ids)
    assert all(report["checks"].values()), report
    assert sum(d["rows"] for d in report["partitions"].values()) == len(members)
    for details in report["partitions"].values():
        assert details["duplicate_groups"] > 0 and details["singleton_groups"] > 0
        assert sum(b["rows"] for b in details["group_size_bins"].values()) == details["rows"]
        assert sum(b["groups"] for b in details["group_size_bins"].values()) == details["groups"]
    # V1's structural defect is caught by new gates on the same synthetic corpus.
    old_ids, _ = allocate_groups(members)
    assert not all(verify_partitions_v2(members, old_ids)["checks"].values())


def test_corrects_v1_first_group_preference_without_a_new_seed():
    values = mixed_values(n=1000, sizes=[])
    values[:, -1] = 0
    values[600:, -1] = 1
    repeated = [0, 1, 2, 3, 4, 5, 600, 601, 602, 603]
    values[repeated, :-1] = values[0, :-1]
    members = duplicate_groups(values)[2]
    assert allocate_groups(members)[1][0]["partition"] == "validation"
    assert allocate_groups_v2(members)[1][0]["partition"] == "train"


def test_all_choices_match_independent_fraction_objective():
    members = duplicate_groups(mixed_values(n=900, sizes=[20, 12, 8, 4, 2] * 4))[2]
    _, allocations = allocate_groups_v2(members)
    sizes = Counter(m["feature_vector_sha256"] for m in members)
    totals = [sum(m["target_class"] == c for m in members) for c in (0, 1)] + [
        len(members),
        len(sizes),
        sum(n for n in sizes.values() if n > 1),
    ]
    counts = [[0] * 5 for _ in PARTITIONS]
    fractions = [Fraction(7, 10), Fraction(3, 20), Fraction(3, 20)]

    def score(candidate):
        value = Fraction(0)
        for p, fraction in enumerate(fractions):
            for j, total in enumerate(totals):
                if total:
                    target = fraction * total
                    value += (candidate[p][j] - target) ** 2 / target
        return value

    for allocation in allocations:
        size = len(allocation["row_ids"])
        added = [*allocation["class_counts"], size, 1, size if size > 1 else 0]
        candidates = []
        for p in range(3):
            candidate = copy.deepcopy(counts)
            candidate[p] = [a + b for a, b in zip(candidate[p], added, strict=True)]
            candidates.append(candidate)
        chosen = min(range(3), key=lambda p: score(candidates[p]))
        assert allocation["partition"] == PARTITIONS[chosen]
        counts = candidates[chosen]


@pytest.mark.parametrize("sizes", [[], [450, *([2] * 20)]])
def test_infeasible_source_structure_is_retained_not_silently_accepted(sizes):
    members = duplicate_groups(mixed_values(n=1000, sizes=sizes))[2]
    ids, allocations = allocate_groups_v2(members)
    report = verify_partitions_v2(members, ids)
    assert report["checks"]["complete_disjoint_row_coverage"]
    assert report["checks"]["zero_cross_partition_feature_group_overlap"]
    assert not all(report["checks"].values())
    assert sum(len(a["row_ids"]) for a in allocations) == 1000


@pytest.mark.parametrize("kind", ["duplicate_id", "missing_id", "invalid_label"])
def test_bad_memberships_rejected(kind):
    members = duplicate_groups(mixed_values())[2]
    if kind == "duplicate_id":
        members[1]["row_id"] = 0
    elif kind == "missing_id":
        members.pop(1)
    else:
        members[0]["target_class"] = 2
    with pytest.raises(ValueError):
        allocate_groups_v2(members)


def test_split_group_rejected_independently_of_allocation():
    members = duplicate_groups(mixed_values())[2]
    ids, allocations = allocate_groups_v2(members)
    home = allocations[0]["partition"]
    other = next(p for p in PARTITIONS if p != home)
    row = allocations[0]["row_ids"][0]
    ids[home].remove(row)
    ids[other] = sorted([*ids[other], row])
    with pytest.raises(ValueError, match="crosses"):
        verify_partitions_v2(members, ids)


@pytest.mark.parametrize("kind", ["seed", "tolerance", "fractions", "preprocessing"])
def test_frozen_config_rejects_scientific_changes(kind):
    # The config is read; none of its actual data paths are accessed in tests.
    config = yaml.safe_load(Path("configs/spambase_split_v2.yaml").read_text())
    validate_config_v2(config)
    if kind == "seed":
        config["split"]["seed"] = 7
    elif kind == "tolerance":
        config["split"]["max_duplicate_row_proportion_error"] = 0.1
    elif kind == "fractions":
        config["split"]["fractions"] = [0.6, 0.2, 0.2]
    else:
        config["preprocessing"]["imputation"] = "mean"
    with pytest.raises(ValueError):
        validate_config_v2(config)


@pytest.mark.parametrize("mode", ["ok", "infeasible", "tamper", "mutated_source", "dirty"])
def test_runner_acceptance_review_gate_and_preserved_failures(tmp_path, monkeypatch, mode):
    import css700.experiments.spambase_split_v2 as runner

    # Reuse the v1 synthetic source/audit helper, not the real-data source paths.
    def fixture_values(n):
        return mixed_values(n=n, sizes=[] if mode == "infeasible" else None)

    monkeypatch.setattr(v1_tests, "synthetic_values", fixture_values)
    config, path = v1_tests.fixture_config(tmp_path, monkeypatch)
    config.update(protocol="spambase_grouped_split_v2", split=copy.deepcopy(SPLIT_POLICY_V2))
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(runner, "_git_state", lambda: ("fixture", mode == "dirty"))
    if mode == "tamper":
        (Path(config["readiness_audit"]["run_dir"]) / "row_memberships.json").write_text("[]")
    if mode == "mutated_source":
        original = runner.allocate_groups_v2

        def mutate(*args, **kwargs):
            result = original(*args, **kwargs)
            (Path(config["source_dir"]) / "spambase.data").write_bytes(b"changed")
            return result

        monkeypatch.setattr(runner, "allocate_groups_v2", mutate)
    result = run_spambase_split_v2(path, tmp_path / "runs")
    assert result["status"] == ("ok" if mode == "ok" else "quality_gate_failed")
    assert not result["training_ready"] and not result["scientific_acceptance"]
    assert result["rows_removed"] == result["labels_changed"] == 0
    assert result["models_trained"] == result["preprocessors_fitted"] == 0
    root = Path(result["output"])
    assert json.loads((root / "summary.json").read_text()) == result
    if mode == "ok":
        assert result["decision"] == "await_independent_scientific_review"
        rows = json.loads((root / "row_assignments.json").read_text())
        assert [r["target_class"] for r in rows] == [0] * 2788 + [1] * 1813
        assert [r["row_id"] for r in rows] == list(range(4601))
        weights = Counter()
        for row in rows:
            assert row["primary_weight"] == 1
            weights[row["feature_vector_sha256"]] += row["group_equal_sensitivity_weight"]
        assert all(abs(total - 1) < 1e-12 for total in weights.values())
        manifest = json.loads((root / "manifest.json").read_text())
        assert all(
            sha256((root / n).read_bytes()) == h for n, h in manifest["output_file_sha256"].items()
        )
    if mode in ("infeasible", "mutated_source"):
        assert (root / "split_row_ids.json").exists()
    if mode == "mutated_source":
        assert not result["checks"]["all_inputs_unchanged"]
