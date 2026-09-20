"""Synthetic-only first-training tests. Never train or transform actual Spambase here."""

# ruff: noqa: E402
import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import test_spambase_preflight as preflight_tests
import yaml

torch = pytest.importorskip("torch")

from css700.data.spambase_grid import PREFLIGHT_FILES, load_spambase_grid_inputs
from css700.encoding import GridEncoder
from css700.experiments.spambase_audit import sha256
from css700.experiments.spambase_grid import (
    TRAINING_POLICY,
    run_spambase_grid,
    validate_training_config,
    validation_scores,
)
from css700.experiments.spambase_preflight import run_spambase_preflight
from css700.training.grid_cnn import PredictionResult
from css700.training.validation_cnn import train_validation_cnn


@pytest.fixture
def accepted(tmp_path, monkeypatch):
    split, split_path, source, preflight_path = preflight_tests.reviewed.__wrapped__(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(
        "css700.experiments.spambase_preflight._git_state", lambda: ("fixture", False)
    )
    result = run_spambase_preflight(preflight_path, tmp_path / "preflight-runs")
    assert result["status"] == "ok"
    root = Path(result["output"])
    review = {
        "review_version": "spambase_training_preflight_v1_review_1",
        "dataset": "spambase",
        "technical_status": "ok",
        "acceptance": True,
        "acceptance_scope": "training_preprocessing_grid_cpu_compatibility_only",
        "decision": "go_to_frozen_grid_training_protocol",
        "training_protocol_ready": True,
        "test_evaluation_ready": False,
        "independent_calculation": {"status": "passed"},
        "file_sha256": {n: sha256((root / n).read_bytes()) for n in PREFLIGHT_FILES},
        "run_dir": str(root),
        "execution_commit": "fixture",
        "preflight_config": str(preflight_path),
        "preflight_config_sha256": sha256(preflight_path.read_bytes()),
        "split_review": str(split_path),
        "split_review_sha256": sha256(split_path.read_bytes()),
        "split_fingerprint": split["split_fingerprint"],
    }
    path = tmp_path / "preflight-review.yaml"
    path.write_text(yaml.safe_dump(review))
    config = {
        "protocol": "spambase_grid_train_validation_v1",
        "preflight_review": str(path),
        "preflight_review_sha256": sha256(path.read_bytes()),
        "training": copy.deepcopy(TRAINING_POLICY),
    }
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return review, path, split, source, config_path


def test_loader_uses_frozen_scaler_and_never_creates_test_tensors(accepted, monkeypatch):
    from sklearn.preprocessing import StandardScaler

    review, path, split, source, _ = accepted
    calls = []
    original = GridEncoder.transform

    def encode(self, values):
        calls.append(values.copy())
        return original(self, values)

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("refitting the reviewed scaler is forbidden")

    monkeypatch.setattr(StandardScaler, "fit", forbidden_fit)
    monkeypatch.setattr(GridEncoder, "transform", encode)
    hashes = {}
    inputs = load_spambase_grid_inputs(path, sha256(path.read_bytes()), hashes)
    ids = json.loads((Path(split["run_dir"]) / "split_row_ids.json").read_text())
    raw = np.loadtxt(Path(source["source_dir"]) / "spambase.data", delimiter=",")
    a = inputs.arrays
    assert len(hashes) == 30
    assert [len(x) for x in calls] == [len(ids["train"]), len(ids["validation"])]
    expected = (raw[ids["validation"], :-1] - a["scaler_mean"]) / a["scaler_scale"]
    np.testing.assert_array_equal(calls[1], expected)
    np.testing.assert_array_equal(
        a["validation_images"].reshape(len(expected), 64)[:, :57], expected.astype(np.float32)
    )
    assert np.all(a["validation_images"].reshape(len(expected), 64)[:, 57:] == 0)
    np.testing.assert_array_equal(a["validation_indices"], ids["validation"])
    np.testing.assert_array_equal(a["y_validation"], raw[ids["validation"], -1])
    assert not any("test" in k for k in a)
    with np.load(Path(review["run_dir"]) / "training_arrays.npz", allow_pickle=False) as stored:
        np.testing.assert_array_equal(a["train_images"], stored["images"])
        for name in ("scaler_mean", "scaler_var", "scaler_scale"):
            np.testing.assert_array_equal(a[name], stored[name])
    assert all(sha256(Path(p).read_bytes()) == h for p, h in hashes.items())


@pytest.mark.parametrize("kind", ["review", "artifact", "source", "rehashed_scaler"])
def test_loader_rejects_tampering(accepted, kind):
    review, path, _, source, _ = accepted
    root = Path(review["run_dir"])
    if kind == "review":
        review["acceptance"] = False
        path.write_text(yaml.safe_dump(review))
    elif kind == "source":
        (Path(source["source_dir"]) / "spambase.data").write_bytes(b"changed")
    elif kind == "artifact":
        (root / "training_arrays.npz").write_bytes(b"changed")
    else:
        with np.load(root / "training_arrays.npz", allow_pickle=False) as stored:
            arrays = {k: stored[k].copy() for k in stored.files}
        arrays["scaler_mean"][0] += 1
        np.savez_compressed(root / "training_arrays.npz", **arrays)
        review["file_sha256"]["training_arrays.npz"] = sha256(
            (root / "training_arrays.npz").read_bytes()
        )
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["output_file_sha256"]["training_arrays.npz"] = review["file_sha256"][
            "training_arrays.npz"
        ]
        (root / "manifest.json").write_text(json.dumps(manifest))
        review["file_sha256"]["manifest.json"] = sha256((root / "manifest.json").read_bytes())
        path.write_text(yaml.safe_dump(review))
    with pytest.raises(ValueError):
        load_spambase_grid_inputs(path, sha256(path.read_bytes()), {})


def easy_arrays():
    y = np.repeat(np.array([0, 1], dtype=np.int64), 16)
    x = np.broadcast_to((y * 2 - 1)[:, None, None, None], (32, 1, 8, 8)).astype(np.float32).copy()
    return x, y, x[::2].copy(), y[::2].copy()


def arguments():
    return dict(
        seed=42,
        epochs=8,
        batch_size=16,
        learning_rate=0.01,
        weight_decay=0.0001,
        patience=3,
        max_seconds=30,
    )


def test_training_is_reproducible_has_no_test_argument_and_restores_state():
    assert not any("test" in p for p in inspect.signature(train_validation_cnn).parameters)
    rng = torch.get_rng_state().clone()
    threads, deterministic, warn = (
        torch.get_num_threads(),
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    first = train_validation_cnn(*easy_arrays(), **arguments())
    second = train_validation_cnn(*easy_arrays(), **arguments())
    assert first.history == second.history
    np.testing.assert_array_equal(first.validation.logits, second.validation.logits)
    assert first.validation.metrics["accuracy"] == 1
    assert first.parameters_changed and first.optimizer_steps == len(first.history) * 2
    assert first.best_epoch == next(
        row["epoch"]
        for row in first.history
        if row["validation_macro_f1"] == max(r["validation_macro_f1"] for r in first.history)
    )
    assert torch.equal(rng, torch.get_rng_state()) and threads == torch.get_num_threads()
    assert deterministic == torch.are_deterministic_algorithms_enabled()
    assert warn == torch.is_deterministic_algorithms_warn_only_enabled()
    with pytest.raises(TimeoutError):
        train_validation_cnn(*easy_arrays(), **{**arguments(), "max_seconds": 1e-12})
    assert torch.equal(rng, torch.get_rng_state()) and threads == torch.get_num_threads()
    assert deterministic == torch.are_deterministic_algorithms_enabled()
    assert warn == torch.is_deterministic_algorithms_warn_only_enabled()


def test_ties_keep_earliest_and_stop_at_exact_patience(monkeypatch):
    import css700.training.validation_cnn as trainer

    original = trainer.evaluate_classifier
    callbacks = []

    def fixed_score(*args, **kwargs):
        result = original(*args, **kwargs)
        return PredictionResult(
            result.loss,
            result.logits,
            result.probabilities,
            result.predictions,
            {**result.metrics, "macro_f1": 0.4},
        )

    monkeypatch.setattr(trainer, "evaluate_classifier", fixed_score)
    outcome = train_validation_cnn(
        *easy_arrays(),
        **{**arguments(), "epochs": 20, "patience": 2},
        on_epoch=lambda history, model, improved: callbacks.append(improved),
    )
    assert outcome.best_epoch == 1 and len(outcome.history) == 3
    assert outcome.stop_reason == "early_stopping_patience"
    assert callbacks == [True, False, False]


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_seed", 7),
        ("epochs", 101),
        ("test_evaluation", True),
        ("refit_scaler", True),
        ("selection", "validation_loss"),
    ],
)
def test_frozen_config_rejects_policy_changes(field, value):
    config = yaml.safe_load(Path("configs/spambase_grid_seed_42_v1.yaml").read_text())
    validate_training_config(config)
    config["training"][field] = value
    with pytest.raises(ValueError):
        validate_training_config(config)


def test_group_equal_metrics_match_hand_calculation():
    labels, predictions = np.array([0, 0, 0, 1, 1]), np.array([0, 0, 0, 0, 1])
    ordinary = validation_scores(labels, predictions)
    weighted = validation_scores(labels, predictions, np.array([1 / 3, 1 / 3, 1 / 3, 1, 1]))
    assert ordinary["accuracy"] == 0.8
    assert weighted["accuracy"] == pytest.approx(2 / 3)
    assert weighted["macro_f1"] == pytest.approx(2 / 3)
    assert weighted["confusion_matrix"] == [[1, 0], [1, 1]]


@pytest.mark.parametrize("mode", ["ok", "dirty", "timeout", "mutated_source"])
def test_runner_preserves_outputs_and_failures_without_test_use(
    accepted, monkeypatch, tmp_path, mode
):
    import css700.experiments.spambase_grid as runner

    _, _, _, source, config_path = accepted
    monkeypatch.setattr(runner, "_git_state", lambda: ("fixture", mode == "dirty"))
    # A one-epoch policy exists only in this synthetic test; real config stays at 100.
    synthetic_policy = {**TRAINING_POLICY, "epochs": 1}
    monkeypatch.setattr(runner, "TRAINING_POLICY", synthetic_policy)
    config = yaml.safe_load(config_path.read_text())
    config["training"] = synthetic_policy
    config_path.write_text(yaml.safe_dump(config))
    original = runner.train_validation_cnn
    calls = []

    def bounded(*args, **kwargs):
        assert len(args) == 4  # Train and validation only, no test arguments.
        calls.append(kwargs["epochs"])
        result = original(*args, **kwargs)
        if mode == "timeout":
            raise TimeoutError("synthetic timeout after one saved epoch")
        if mode == "mutated_source":
            (Path(source["source_dir"]) / "spambase.data").write_bytes(b"changed")
        return result

    monkeypatch.setattr(runner, "train_validation_cnn", bounded)
    result = run_spambase_grid(config_path, tmp_path / "training-runs")
    assert result["status"] == ("ok" if mode == "ok" else "quality_gate_failed"), result
    root = Path(result["output"])
    assert json.loads((root / "summary.json").read_text()) == result
    manifest = json.loads((root / "manifest.json").read_text())
    assert all(
        sha256((root / n).read_bytes()) == h for n, h in manifest["output_file_sha256"].items()
    )
    assert (
        result["test_rows_transformed_or_scored"]
        == result["scalers_refitted"]
        == result["explanations_generated"]
        == 0
    )
    assert not result["test_evaluation_ready"]
    if mode == "dirty":
        assert not calls and not (root / "best_checkpoint.pt").exists()
    else:
        assert calls == [1]
        assert (root / "best_checkpoint.pt").is_file()
        with np.load(root / "inputs.npz", allow_pickle=False) as a:
            assert not any("test" in n for n in a.files)
    if mode == "ok":
        assert len(manifest["source_file_sha256"]) == 31
        assert result["checks"]["checkpoint_reload_exact"]
        assert result["decision"] == "await_independent_training_review"
    if mode == "timeout":
        assert result["partial_completed_epochs"] == 1
        assert not (root / "validation_predictions.npz").exists()
    if mode == "mutated_source":
        assert not result["checks"]["all_inputs_unchanged"]


def test_tracked_config_pins_exact_preflight_review():
    config = yaml.safe_load(Path("configs/spambase_grid_seed_42_v1.yaml").read_text())
    validate_training_config(config)
    assert config["training"]["epochs"] == 100
    assert config["preflight_review_sha256"] == sha256(
        Path(config["preflight_review"]).read_bytes()
    )
