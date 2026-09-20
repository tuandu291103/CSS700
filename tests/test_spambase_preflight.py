"""Synthetic-only preflight tests; never preprocess the actual Spambase source here."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import test_spambase_split as v1_tests
import yaml
from test_spambase_split_v2 import mixed_values

from css700.data.grouped_split_v2 import SPLIT_POLICY_V2
from css700.data.spambase import (
    SPLIT_FILES,
    TARGET_NAMES,
    SpambaseTrainingSource,
    fit_training_standardizer,
    load_reviewed_spambase_training,
    select_training_rows,
)
from css700.data.spambase_audit import FEATURE_NAMES
from css700.experiments.spambase_audit import sha256
from css700.experiments.spambase_preflight import (
    PREFLIGHT_POLICY,
    cpu_compatibility_check,
    prepare_training_grid,
    run_spambase_preflight,
)
from css700.experiments.spambase_split_v2 import run_spambase_split_v2


@pytest.fixture
def reviewed(tmp_path, monkeypatch):
    monkeypatch.setattr(v1_tests, "synthetic_values", lambda n: mixed_values(n=n))
    config, path = v1_tests.fixture_config(tmp_path, monkeypatch)
    config.update(protocol="spambase_grouped_split_v2", split=copy.deepcopy(SPLIT_POLICY_V2))
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(
        "css700.experiments.spambase_split_v2._git_state", lambda: ("fixture", False)
    )
    result = run_spambase_split_v2(path, tmp_path / "splits")
    assert result["status"] == "ok"
    root = Path(result["output"])
    review = {
        "review_version": "spambase_split_v2_scientific_review_1",
        "dataset": "spambase",
        "scientific_acceptance": True,
        "technical_status": "ok",
        "acceptance_scope": "split_structure_for_training_only_preflight",
        "decision": "go_to_training_only_preflight",
        "v1_remains_diagnostic_only": True,
        "independent_verification": {"status": "passed"},
        "file_sha256": {n: sha256((root / n).read_bytes()) for n in SPLIT_FILES},
        "run_dir": str(root),
        "execution_commit": "fixture",
        "split_config": str(path),
        "split_config_sha256": sha256(path.read_bytes()),
        "split_fingerprint": result["split_fingerprint"],
    }
    review_path = tmp_path / "review.yaml"
    review_path.write_text(yaml.safe_dump(review))
    preflight = {
        "protocol": "spambase_training_preflight_v1",
        "review_path": str(review_path),
        "review_sha256": sha256(review_path.read_bytes()),
        "policy": copy.deepcopy(PREFLIGHT_POLICY),
    }
    preflight_path = tmp_path / "preflight.yaml"
    preflight_path.write_text(yaml.safe_dump(preflight))
    return review, review_path, config, preflight_path


def test_loader_exact_ids_labels_features_and_21_pins(reviewed):
    review, path, config, _ = reviewed
    source = load_reviewed_spambase_training(path, sha256(path.read_bytes()))
    values = np.loadtxt(Path(config["source_dir"]) / "spambase.data", delimiter=",")
    ids = json.loads((Path(review["run_dir"]) / "split_row_ids.json").read_text())
    np.testing.assert_array_equal(source.train_indices, ids["train"])
    np.testing.assert_array_equal(source.x_train_raw, values[ids["train"], :-1])
    np.testing.assert_array_equal(source.y_train, values[ids["train"], -1])
    assert source.feature_names == FEATURE_NAMES and source.target_names == TARGET_NAMES
    assert source.split_fingerprint == review["split_fingerprint"]
    assert len(source.source_file_sha256) == 21
    assert all(sha256(Path(p).read_bytes()) == h for p, h in source.source_file_sha256.items())
    assert not source.x_train_raw.flags.writeable and not source.y_train.flags.writeable
    assert not source.train_indices.flags.writeable
    assert not hasattr(source, "x_test") and not hasattr(source, "x_validation")


@pytest.mark.parametrize(
    "field,value",
    [
        ("scientific_acceptance", False),
        ("review_version", "spambase_split_v1_scientific_review_1"),
        ("decision", "hold"),
        ("split_fingerprint", "incorrect"),
    ],
)
def test_semantically_invalid_review_rejected_even_if_rehashed(reviewed, field, value):
    review, path, _, _ = reviewed
    review[field] = value
    path.write_text(yaml.safe_dump(review))
    with pytest.raises(ValueError):
        load_reviewed_spambase_training(path, sha256(path.read_bytes()))


@pytest.mark.parametrize("target", ["review", "source", "assignments"])
def test_changed_pinned_bytes_rejected(reviewed, target):
    review, path, config, _ = reviewed
    expected = sha256(path.read_bytes())
    changed = {
        "review": path,
        "source": Path(config["source_dir"]) / "spambase.data",
        "assignments": Path(review["run_dir"]) / "row_assignments.json",
    }[target]
    changed.write_bytes(changed.read_bytes() + b" ")
    with pytest.raises(ValueError):
        load_reviewed_spambase_training(path, expected)


@pytest.mark.parametrize("field,value", [("target_class", 99), ("primary_weight", 0.5)])
def test_rehashed_but_inconsistent_row_records_rejected(reviewed, field, value):
    review, path, _, _ = reviewed
    root = Path(review["run_dir"])
    rows_path = root / "row_assignments.json"
    rows = json.loads(rows_path.read_text())
    rows[0][field] = value
    rows_path.write_text(json.dumps(rows))
    review["file_sha256"]["row_assignments.json"] = sha256(rows_path.read_bytes())
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output_file_sha256"]["row_assignments.json"] = sha256(rows_path.read_bytes())
    manifest_path.write_text(json.dumps(manifest))
    review["file_sha256"]["manifest.json"] = sha256(manifest_path.read_bytes())
    path.write_text(yaml.safe_dump(review))
    with pytest.raises(ValueError, match="row identity, label, group or evaluation weight"):
        load_reviewed_spambase_training(path, sha256(path.read_bytes()))


def test_scaler_training_only_spy_heldout_invariance_and_constants(monkeypatch):
    from sklearn.preprocessing import StandardScaler

    values = np.random.default_rng(5).normal(size=(30, 58))
    values[:, -1] = np.arange(30) % 2
    values[:, 4] = 7  # Retained nonzero constant.
    values[:, 5] = 0  # Retained all-zero constant.
    indices = list(range(0, 30, 2))
    calls = []
    original_fit, original_transform = StandardScaler.fit, StandardScaler.transform

    def fit(self, x, *args, **kwargs):
        calls.append(("fit", x.copy()))
        return original_fit(self, x, *args, **kwargs)

    def transform(self, x, *args, **kwargs):
        calls.append(("transform", x.copy()))
        return original_transform(self, x, *args, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", fit)
    monkeypatch.setattr(StandardScaler, "transform", transform)
    train, labels = select_training_rows(values, indices)
    first, transformed, info = fit_training_standardizer(train)
    values[1::2, :-1] = 1e12
    values[1::2, -1] = 1 - values[1::2, -1]
    second_x, second_y = select_training_rows(values, indices)
    second, second_transformed, _ = fit_training_standardizer(second_x)
    for name in ("mean_", "var_", "scale_"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    np.testing.assert_array_equal(transformed, second_transformed)
    np.testing.assert_array_equal(labels, second_y)
    assert [kind for kind, _ in calls] == ["fit", "transform", "fit", "transform"]
    for _, supplied in calls:
        np.testing.assert_array_equal(supplied, train)
    assert info["fit_rows"] == 15 and info["constant_feature_indices"] == [4, 5]
    assert np.all(transformed[:, [4, 5]] == 0) and np.all(first.scale_[[4, 5]] == 1)


@pytest.mark.parametrize("indices", [[1, 1], [-1], [30], [0.5], []])
def test_training_selector_rejects_invalid_ids(indices):
    with pytest.raises(ValueError):
        select_training_rows(np.zeros((30, 58)), indices)


def test_grid_and_cpu_checks_preserve_features_and_torch_state(tmp_path):
    torch = pytest.importorskip("torch")
    values = np.random.default_rng(8).normal(size=(40, 58))
    values[:, -1] = np.arange(40) % 2
    x, y = select_training_rows(values, list(range(40)))
    source = SpambaseTrainingSource(
        x, y, np.arange(40), FEATURE_NAMES, TARGET_NAMES, "test", {}, {}
    )
    rng = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    summary, outputs = prepare_training_grid(source, tmp_path)
    assert all(summary["checks"].values()), summary
    assert summary["training_image_shape"] == [40, 1, 8, 8]
    assert summary["cpu_smoke"]["optimizer_steps"] == 0
    assert summary["cpu_smoke"]["batch_rows"] == 32
    assert summary["cpu_smoke"]["parameters"] == 4866
    assert len(outputs) == 4
    assert torch.equal(rng, torch.get_rng_state())
    assert threads == torch.get_num_threads()
    assert deterministic == torch.are_deterministic_algorithms_enabled()
    assert warn_only == torch.is_deterministic_algorithms_warn_only_enabled()
    with np.load(tmp_path / "training_arrays.npz", allow_pickle=False) as arrays:
        assert set(arrays.files) == {
            "train_indices",
            "y_train",
            "x_standardized",
            "images",
            "occupied_mask",
            "scaler_mean",
            "scaler_var",
            "scaler_scale",
        }
        np.testing.assert_array_equal(
            arrays["images"].reshape(40, 64)[:, :57], arrays["x_standardized"].astype(np.float32)
        )
        assert np.all(arrays["images"].reshape(40, 64)[:, 57:] == 0)
    # A deliberately invalid input must not leak altered global runtime settings.
    with pytest.raises((RuntimeError, IndexError)):
        cpu_compatibility_check(np.zeros((32, 1, 8, 8), np.float32), np.full(32, 9))
    assert torch.equal(rng, torch.get_rng_state()) and threads == torch.get_num_threads()
    assert deterministic == torch.are_deterministic_algorithms_enabled()
    assert warn_only == torch.is_deterministic_algorithms_warn_only_enabled()


@pytest.mark.parametrize("mode", ["ok", "dirty", "policy", "mutated_source"])
def test_runner_retains_failures_and_never_trains_or_transforms_heldout(
    reviewed, monkeypatch, tmp_path, mode
):
    pytest.importorskip("torch")
    import css700.experiments.spambase_preflight as runner

    _, _, config, preflight_path = reviewed
    monkeypatch.setattr(runner, "_git_state", lambda: ("fixture", mode == "dirty"))
    if mode == "policy":
        changed = yaml.safe_load(preflight_path.read_text())
        changed["policy"]["heldout_transform"] = True
        preflight_path.write_text(yaml.safe_dump(changed))
    if mode == "mutated_source":
        original = runner.prepare_training_grid

        def mutate(*args, **kwargs):
            result = original(*args, **kwargs)
            (Path(config["source_dir"]) / "spambase.data").write_bytes(b"changed")
            return result

        monkeypatch.setattr(runner, "prepare_training_grid", mutate)
    result = run_spambase_preflight(preflight_path, tmp_path / "preflights")
    assert result["status"] == ("ok" if mode == "ok" else "quality_gate_failed"), result
    assert result["heldout_rows_transformed_or_encoded"] == result["optimizer_steps"] == 0
    assert result["models_trained"] == result["explanations_generated"] == 0
    assert not result["test_evaluation_ready"]
    root = Path(result["output"])
    assert json.loads((root / "summary.json").read_text()) == result
    manifest = json.loads((root / "manifest.json").read_text())
    assert all(
        sha256((root / n).read_bytes()) == h for n, h in manifest["output_file_sha256"].items()
    )
    if mode == "ok":
        assert len(manifest["source_file_sha256"]) == 22
        assert result["decision"] == "go_to_frozen_grid_training_protocol"
        assert result["scaling"]["constant_feature_indices"] == list(range(1, 57))
    if mode == "mutated_source":
        assert not result["checks"]["all_inputs_unchanged"]
        assert (root / "training_arrays.npz").exists()  # Retain failed-run evidence.


def test_tracked_preflight_config_matches_prospective_policy():
    config = yaml.safe_load(Path("configs/spambase_preflight_v1.yaml").read_text())
    assert config["policy"] == PREFLIGHT_POLICY
    assert config["review_sha256"] == sha256(Path(config["review_path"]).read_bytes())
