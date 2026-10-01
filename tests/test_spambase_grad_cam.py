"""Prospective synthetic tests only; no real checkpoint or attribution execution."""

# ruff: noqa: E402
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
import test_spambase_grid as training_tests

from css700.data.spambase_audit import FEATURE_NAMES
from css700.encoding import GridEncoder
from css700.experiments.spambase_audit import sha256
from css700.experiments.spambase_checkpoint import (
    TRAINING_FILES,
    FrozenValidationCheckpoint,
    load_validation_checkpoint,
)
from css700.experiments.spambase_grad_cam import (
    GRAD_CAM_POLICY,
    run_spambase_grad_cam,
    validate_grad_cam_config,
)
from css700.interpretability.spambase_grad_cam import (
    explain_validation,
    map_diagnostics,
    selected_examples,
)
from css700.models import SmallTabularImageCNN


@pytest.fixture
def reviewed_training(tmp_path, monkeypatch):
    import css700.experiments.spambase_grid as trainer

    _, preflight_path, _, _, config_path = training_tests.accepted.__wrapped__(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(trainer, "_git_state", lambda: ("fixture", False))
    policy = {**trainer.TRAINING_POLICY, "epochs": 1}
    monkeypatch.setattr(trainer, "TRAINING_POLICY", policy)
    config = yaml.safe_load(config_path.read_text())
    config["training"] = policy
    config_path.write_text(yaml.safe_dump(config))
    result = trainer.run_spambase_grid(config_path, tmp_path / "training-runs")
    assert result["status"] == "ok", result
    root = Path(result["output"])
    review = {
        "review_version": "spambase_grid_train_validation_v1_review_1",
        "dataset": "spambase",
        "technical_status": "ok",
        "acceptance": True,
        "acceptance_scope": "seed_42_training_and_validation_checkpoint_only",
        "decision": "go_to_validation_grad_cam_protocol",
        "test_evaluation_ready": False,
        "independent_calculation": {"status": "passed"},
        "run_dir": str(root),
        "file_sha256": {n: sha256((root / n).read_bytes()) for n in TRAINING_FILES},
        "execution_commit": "fixture",
        "training_config": str(config_path),
        "training_config_sha256": sha256(config_path.read_bytes()),
        "preflight_review": str(preflight_path),
        "preflight_review_sha256": sha256(preflight_path.read_bytes()),
        "validation_accuracy": result["validation_primary"]["accuracy"],
        "validation_macro_f1": result["validation_primary"]["macro_f1"],
        **{
            k: result[k]
            for k in (
                "model_seed",
                "epochs_completed",
                "best_epoch",
                "optimizer_steps",
                "stop_reason",
                "split_fingerprint",
            )
        },
    }
    path = tmp_path / "training-review.yaml"
    path.write_text(yaml.safe_dump(review))
    cam_config = {
        "protocol": "spambase_validation_grad_cam_v1",
        "training_review": str(path),
        "training_review_sha256": sha256(path.read_bytes()),
        "policy": copy.deepcopy(GRAD_CAM_POLICY),
    }
    cam_path = tmp_path / "cam.yaml"
    cam_path.write_text(yaml.safe_dump(cam_config))
    return review, path, cam_path


def tiny_source(zero=False):
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(7)
            model = SmallTabularImageCNN()
            if zero:
                with torch.no_grad():
                    model.classifier.weight.zero_()
                    model.classifier.bias.copy_(torch.tensor([1.0, 0.0]))
            encoder = GridEncoder(rows=8, columns=8).fit(FEATURE_NAMES)
            images = encoder.transform(np.random.default_rng(9).normal(size=(4, 57)))
            with torch.no_grad():
                logits = model(torch.from_numpy(images)).numpy()
            p = {
                "sample_ids": np.arange(4),
                "labels": np.arange(4) % 2,
                "predictions": logits.argmax(axis=1),
                "logits": logits,
                "group_equal_weights": np.ones(4),
            }
            source = FrozenValidationCheckpoint(
                {"validation_images": images, "occupied_mask": encoder.padding_mask()},
                p,
                encoder.mapping_records(),
                {"model_state_dict": model.state_dict()},
                "fixture",
                "fixture",
            )
            return source, model
    finally:
        torch.set_num_threads(old_threads)


def test_analytic_reference_exact_mapping_targets_and_runtime_state():
    source, model = tiny_source()
    rng = torch.get_rng_state().clone()
    threads, deterministic, warn = (
        torch.get_num_threads(),
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    outputs, checks = explain_validation(source)
    assert all(checks.values()), checks
    with torch.no_grad():
        activation = (
            model.features[:3](torch.from_numpy(source.arrays["validation_images"]))
            .numpy()
            .astype(np.float64)
        )
        head = model.classifier.weight.numpy().astype(np.float64)[source.predictions["predictions"]]
    derivative = head[:, :, None, None] * (activation > 0) / 64
    reference = np.maximum(
        (derivative.mean(axis=(2, 3), keepdims=True) * activation).sum(axis=1, keepdims=True), 0
    )
    np.testing.assert_allclose(outputs["raw_maps"], reference, rtol=1e-5, atol=1e-7)
    np.testing.assert_array_equal(
        outputs["feature_scores"], outputs["raw_maps"].reshape(4, 64)[:, :57]
    )
    assert torch.equal(rng, torch.get_rng_state()) and threads == torch.get_num_threads()
    assert deterministic == torch.are_deterministic_algorithms_enabled()
    assert warn == torch.is_deterministic_algorithms_warn_only_enabled()
    with pytest.raises(TimeoutError):
        explain_validation(source, max_seconds=1e-12)
    assert torch.equal(rng, torch.get_rng_state()) and threads == torch.get_num_threads()
    assert deterministic == torch.are_deterministic_algorithms_enabled()
    assert warn == torch.is_deterministic_algorithms_warn_only_enabled()


def test_zero_maps_are_retained_not_a_gate_failure():
    source, _ = tiny_source(zero=True)
    outputs, checks = explain_validation(source)
    assert all(checks.values())
    records, aggregate, summary = map_diagnostics(outputs, source)
    assert summary["zero_raw_maps"] == summary["zero_feature_maps"] == 4
    assert summary["padding_fraction_nonzero_maps"]["defined_rows"] == 0
    assert all(r["padding_fraction"] is None and not r["positive_top_5"] for r in records)
    assert aggregate["predicted_class_1"]["rows"] == 0


def test_gradient_logit_tolerance_does_not_relax_exact_inference(monkeypatch):
    import css700.interpretability.spambase_grad_cam as implementation
    from css700.interpretability.grad_cam import GradCAMResult

    source, _ = tiny_source()
    original = implementation.grad_cam

    def perturbed(*args, **kwargs):
        result = original(*args, **kwargs)
        return GradCAMResult(
            result.raw, result.normalized, result.target_classes, result.logits + 0.001
        )

    monkeypatch.setattr(implementation, "grad_cam", perturbed)
    _, checks = explain_validation(source)
    assert checks["frozen_inference_logits_exact"]
    assert not checks["gradient_logits_numerically_match"]
    monkeypatch.setattr(implementation, "grad_cam", original)
    source.predictions["logits"][0, 0] = np.nextafter(
        source.predictions["logits"][0, 0], np.float32(np.inf)
    )
    _, checks = explain_validation(source)
    assert not checks["frozen_inference_logits_exact"]
    assert checks["gradient_logits_numerically_match"]


def test_padding_only_positive_lists_ties_and_group_weighted_means():
    source, _ = tiny_source()
    raw = np.zeros((4, 1, 8, 8), np.float32)
    flat = raw.reshape(4, 64)
    flat[1, :57], flat[1, 57:] = 1, 100
    flat[2, 0] = 0.5
    flat[3, 57:] = 2  # Padding-only: no named-feature evidence.
    source.predictions["group_equal_weights"][:] = [1, 0.5, 0.5, 1]
    outputs = {
        "raw_maps": raw,
        "feature_scores": flat[:, :57].astype(np.float64),
        "target_classes": source.predictions["predictions"],
    }
    records, aggregate, summary = map_diagnostics(outputs, source)
    assert records[1]["padding_fraction"] == pytest.approx(700 / 757)
    assert records[3]["padding_fraction"] == 1 and records[3]["zero_feature_map"]
    assert summary["zero_raw_maps"] == 1 and summary["zero_feature_maps"] == 2
    assert [r["feature_index"] for r in records[1]["positive_top_5"]] == list(range(5))
    assert records[1]["positive_tie_at_5"] and records[1]["positive_tie_at_10"]
    assert len(records[2]["positive_top_5"]) == 1
    assert not records[3]["positive_top_5"]
    assert aggregate["all_predicted_classes"]["row_primary"][0]["mean_raw_score"] == pytest.approx(
        1.5 / 4
    )
    assert aggregate["all_predicted_classes"]["group_equal"][0]["mean_raw_score"] == pytest.approx(
        0.75 / 3
    )


def test_example_selection_is_fixed_and_does_not_choose_by_attribution():
    p = {"labels": np.array([1, 0, 0, 1, 0]), "predictions": np.array([0, 1, 0, 1, 0])}
    assert selected_examples(p) == [2, 3, 1, 0]


def test_loader_verifies_40_inputs_and_returns_validation_only(reviewed_training, monkeypatch):
    from sklearn.preprocessing import StandardScaler

    _, path, _ = reviewed_training

    def forbidden(*args, **kwargs):
        raise AssertionError("no fit or encoding allowed during explanation loading")

    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    monkeypatch.setattr(GridEncoder, "transform", forbidden)
    hashes = {}
    source = load_validation_checkpoint(path, sha256(path.read_bytes()), hashes)
    assert len(hashes) == 40
    assert set(source.arrays) == {
        "validation_indices",
        "y_validation",
        "validation_images",
        "validation_group_weights",
        "occupied_mask",
    }
    assert all(sha256(Path(p).read_bytes()) == h for p, h in hashes.items())
    assert not source.arrays["validation_images"].flags.writeable
    assert len(source.feature_map) == 57


@pytest.mark.parametrize("kind", ["review", "artifact", "rehashed_targets", "rehashed_ids"])
def test_loader_rejects_altered_or_inconsistent_source(reviewed_training, kind):
    review, path, _ = reviewed_training
    root = Path(review["run_dir"])
    if kind == "review":
        review["acceptance"] = False
    elif kind == "artifact":
        (root / "feature_map.json").write_text("[]")
    else:
        name = "validation_predictions.npz"
        with np.load(root / name, allow_pickle=False) as stored:
            arrays = {k: stored[k].copy() for k in stored.files}
        if kind == "rehashed_targets":
            arrays["predictions"][0] = 1 - arrays["predictions"][0]
        else:
            arrays["sample_ids"][0] += 1
        np.savez_compressed(root / name, **arrays)
        review["file_sha256"][name] = sha256((root / name).read_bytes())
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["output_file_sha256"][name] = review["file_sha256"][name]
        (root / "manifest.json").write_text(json.dumps(manifest))
        review["file_sha256"]["manifest.json"] = sha256((root / "manifest.json").read_bytes())
    path.write_text(yaml.safe_dump(review))
    with pytest.raises(ValueError):
        load_validation_checkpoint(path, sha256(path.read_bytes()), {})


@pytest.mark.parametrize(
    "key,value",
    [
        ("split", "test"),
        ("targets", "true_class"),
        ("layer", "features.3"),
        ("batch_size", 16),
        ("refit_or_retrain", True),
    ],
)
def test_frozen_policy_rejects_changes(key, value):
    config = yaml.safe_load(Path("configs/spambase_grad_cam_validation_v1.yaml").read_text())
    validate_grad_cam_config(config)
    config["policy"][key] = value
    with pytest.raises(ValueError):
        validate_grad_cam_config(config)


@pytest.mark.parametrize("mode", ["ok", "dirty", "timeout", "mutated_source"])
def test_runner_preserves_failures_and_keeps_test_gate_closed(
    reviewed_training, monkeypatch, tmp_path, mode
):
    pytest.importorskip("matplotlib")
    import css700.experiments.spambase_grad_cam as runner

    _, path, config_path = reviewed_training
    monkeypatch.setattr(runner, "_git_state", lambda: ("fixture", mode == "dirty"))
    original = runner.explain_validation

    def explain(*args, **kwargs):
        if mode == "timeout":
            raise TimeoutError("synthetic timeout")
        result = original(*args, **kwargs)
        if mode == "mutated_source":
            path.write_text(path.read_text() + "\n")
        return result

    monkeypatch.setattr(runner, "explain_validation", explain)
    result = run_spambase_grad_cam(config_path, tmp_path / "explanations")
    assert result["status"] == ("ok" if mode == "ok" else "quality_gate_failed"), result
    assert not result["test_evaluation_ready"]
    assert (
        result["models_trained"]
        == result["scalers_refitted"]
        == result["optimizer_steps"]
        == result["test_rows_transformed_or_scored"]
        == 0
    )
    root = Path(result["output"])
    manifest = json.loads((root / "manifest.json").read_text())
    assert all(
        sha256((root / n).read_bytes()) == h for n, h in manifest["output_file_sha256"].items()
    )
    if mode == "ok":
        assert len(manifest["source_file_sha256"]) == 41
        assert (root / "validation_examples.png").stat().st_size > 1000
        with np.load(root / "grad_cam_arrays.npz", allow_pickle=False) as a:
            assert a["feature_scores"].shape == (result["explained_rows"], 57)
            assert not any("test" in n for n in a.files)
    if mode == "mutated_source":
        assert not result["checks"]["all_inputs_unchanged"]
        assert (root / "grad_cam_arrays.npz").exists()


def test_tracked_config_pins_training_review():
    config = yaml.safe_load(Path("configs/spambase_grad_cam_validation_v1.yaml").read_text())
    validate_grad_cam_config(config)
    assert config["training_review_sha256"] == sha256(Path(config["training_review"]).read_bytes())
