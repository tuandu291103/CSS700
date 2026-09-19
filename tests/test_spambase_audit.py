"""Synthetic source/schema/duplicate checks; never download data during tests."""

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from css700.data.spambase_audit import (
    FEATURE_NAMES,
    duplicate_groups,
    inspect_data,
    parse_data,
    parse_names,
)
from css700.experiments.spambase_audit import (
    ARCHIVE_URL,
    MEMBERS,
    archive_members,
    fetch_spambase,
    load_pinned_source,
    run_spambase_audit,
    sha256,
)


def names_bytes():
    return (
        "| synthetic dictionary\n1, 0. | classes\n"
        + "\n".join(f"{n}: continuous." for n in FEATURE_NAMES)
    ).encode()


def numeric_rows(n=6):
    x = np.zeros((n, 58))
    x[:, 54:57] = 1
    return x


def data_bytes(values):
    return ("\n".join(",".join(str(v) for v in row) for row in values) + "\n").encode()


def synthetic_source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    values = numeric_rows(4601)
    values[2788:, -1] = 1
    payloads = {
        "spambase.data": data_bytes(values),
        "spambase.names": names_bytes(),
        "spambase.DOCUMENTATION": b"Synthetic fixture, not research data.",
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)
    payloads["spambase.zip"] = buffer.getvalue()
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)
    receipt = {
        "status": "ok",
        "errors": [],
        "worktree_dirty_at_start": False,
        "source_url": ARCHIVE_URL,
        "final_url": ARCHIVE_URL,
        "files": {n: {"sha256": sha256(p), "bytes": len(p)} for n, p in payloads.items()},
    }
    (root / "receipt.json").write_text(json.dumps(receipt))
    lock = {
        "dataset": "spambase",
        "source_url": ARCHIVE_URL,
        "receipt_sha256": sha256((root / "receipt.json").read_bytes()),
        "file_sha256": {n: sha256(p) for n, p in payloads.items()},
    }
    lock_path = tmp_path / "source-lock.yaml"
    lock_path.write_text(yaml.safe_dump(lock))
    return root, lock, lock_path


def test_dictionary_preserves_punctuation_and_rejects_wrong_order():
    assert len(parse_names(names_bytes())) == 57
    assert FEATURE_NAMES[52] == "char_freq_$"
    bad = names_bytes().replace(b"word_freq_make", b"unknown")
    with pytest.raises(ValueError, match="order"):
        parse_names(bad)


@pytest.mark.parametrize(
    "token,kind",
    [
        ("?", "missing_token"),
        ("", "missing_token"),
        ("NaN", "missing_token"),
        ("inf", "nonfinite_token"),
        ("bad", "nonnumeric_token"),
    ],
)
def test_parser_retains_invalid_cells_and_source_ids(token, kind):
    fields = data_bytes(numeric_rows(1)).decode().strip().split(",")
    fields[4] = token
    parsed, issues = parse_data((",".join(fields) + "\n").encode())
    assert parsed.shape == (1, 58) and np.isnan(parsed[0, 4])
    assert issues == [
        {"row_id": 0, "line_number": 1, "column_index": 4, "issue": kind, "token": token}
    ]


def test_blank_or_short_rows_not_silently_discarded():
    data = data_bytes(numeric_rows(1)) + b"\n1,2,3\n"
    parsed, issues = parse_data(data)
    assert len(parsed) == 3
    assert [r["row_id"] for r in issues] == [1, 2]
    assert all(r["issue"] == "wrong_field_count" for r in issues)


def test_duplicate_and_conflict_counts_have_distinct_meanings():
    values = numeric_rows()
    values[2, -1] = 1
    values[3:5, 0] = 2
    values[5, 0] = 3
    values[1, 1] = -0.0
    stats, groups, members = duplicate_groups(values)
    assert stats["unique_feature_vectors"] == 3
    assert stats["duplicate_feature_groups"] == 2
    assert stats["rows_in_duplicate_feature_groups"] == 5
    assert stats["redundant_feature_rows_beyond_first"] == 3
    assert stats["redundant_labeled_rows_beyond_first"] == 2
    assert stats["conflicting_feature_groups"] == 1
    assert stats["rows_in_conflicting_feature_groups"] == 3
    assert groups[0]["row_ids"] == [0, 1, 2] and groups[0]["class_counts"] == {"0": 2, "1": 1}
    assert len(members) == 6 and [r["row_id"] for r in members] == list(range(6))
    assert members[0]["feature_vector_sha256"] == members[1]["feature_vector_sha256"]
    np.testing.assert_array_equal(values[:, -1], [0, 0, 1, 0, 0, 0])


def test_float_formatting_groups_but_nearby_values_do_not():
    values = numeric_rows(3)
    values[2, 0] = 1e-10
    payload = data_bytes(values).replace(b"0.0", b"0", 1)
    parsed, issues = parse_data(payload)
    assert not issues and duplicate_groups(parsed)[0]["unique_feature_vectors"] == 2


def test_ranges_integers_and_labels_are_reported_without_repair():
    values = numeric_rows(3)
    values[0, 0] = 101
    values[1, 54] = 0
    values[1, 55] = 1.5
    values[2, -1] = 2
    report = inspect_data(data_bytes(values), names_bytes())
    assert report["invalid_label_row_ids"] == [2]
    assert report["feature_profiles"][0]["out_of_range_row_ids"] == [0]
    assert report["feature_profiles"][54]["out_of_range_row_ids"] == [1]
    assert report["feature_profiles"][55]["noninteger_row_ids"] == [1]
    assert not report["duplicates"]["performed"]


def test_archive_rejects_unexpected_paths():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in MEMBERS:
            archive.writestr(name, b"fixture")
        archive.writestr("../escape", b"untrusted")
    with pytest.raises(ValueError, match="members"):
        archive_members(buffer.getvalue())


def test_pinned_source_checks_receipt_and_member_bytes(tmp_path):
    root, lock, _ = synthetic_source(tmp_path)
    payloads, hashes = load_pinned_source(root, lock)
    assert len(hashes) == 5 and len(payloads) == 5
    (root / "spambase.data").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        load_pinned_source(root, lock)


def test_changed_archive_member_cannot_be_legitimized_by_file_receipt_only(tmp_path):
    root, lock, _ = synthetic_source(tmp_path)
    changed = b"changed"
    (root / "spambase.data").write_bytes(changed)
    lock["file_sha256"]["spambase.data"] = sha256(changed)
    receipt = json.loads((root / "receipt.json").read_text())
    receipt["files"]["spambase.data"] = {"sha256": sha256(changed), "bytes": len(changed)}
    (root / "receipt.json").write_text(json.dumps(receipt))
    lock["receipt_sha256"] = sha256((root / "receipt.json").read_bytes())
    with pytest.raises(ValueError, match="archive members"):
        load_pinned_source(root, lock)


def test_dirty_fetch_and_audit_preserve_failure_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr("css700.experiments.spambase_audit._git_state", lambda: ("fixture", True))

    def never(*args, **kwargs):
        pytest.fail("must not reach network")

    monkeypatch.setattr("css700.experiments.spambase_audit.urllib.request.urlopen", never)
    first = fetch_spambase(tmp_path)
    second = run_spambase_audit(tmp_path, tmp_path / "absent", tmp_path)
    assert first["status"] == "acquisition_failed"
    assert second["status"] == "quality_gate_failed" and not second["training_ready"]
    assert (Path(first["output"]) / "receipt.json").exists()


@pytest.mark.parametrize("mutate", [False, True])
def test_full_synthetic_audit_and_final_source_recheck(tmp_path, monkeypatch, mutate):
    import css700.experiments.spambase_audit as runner

    root, _, lock_path = synthetic_source(tmp_path)
    monkeypatch.setattr(runner, "_git_state", lambda: ("fixture", False))

    def inspect(*args):
        report = inspect_data(*args)
        if mutate:
            (root / "spambase.names").write_bytes(b"changed")
        return report

    monkeypatch.setattr(runner, "inspect_data", inspect)
    result = run_spambase_audit(root, lock_path, tmp_path)
    assert result["status"] == ("quality_gate_failed" if mutate else "ok")
    assert result["rows"] == 4601 and result["duplicates"]["conflicting_feature_groups"] == 1
    assert result["splits_created"] == result["models_trained"] == result["rows_removed"] == 0
    assert not result["training_ready"] and result["checks"]["source_files_unchanged"] is not mutate
