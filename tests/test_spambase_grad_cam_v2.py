"""Synthetic activation-boundary, frozen-policy, provenance and failure-path checks."""

# ruff: noqa: E402
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
import test_spambase_grad_cam as v1_tests

from css700.experiments.spambase_audit import sha256
from css700.experiments.spambase_checkpoint import load_validation_checkpoint
from css700.experiments.spambase_grad_cam_v2 import (
    HELD_FILES,
    V2_POLICY,
    load_held_v1,
    run_spambase_grad_cam_v2,
    validate_v2_config,
)
from css700.interpretability.activation_boundary import (
    bitwise_equal,
    boundary_grad_cam,
    explain_boundary_validation,
)
from css700.interpretability.grad_cam import GradCAMResult
from css700.interpretability.spambase_grad_cam import explain_validation


def runtime_state():
    return (
        torch.get_num_threads(),
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch._C._get_nnpack_enabled(),
        torch.get_rng_state().numpy().tobytes(),
        torch.is_grad_enabled(),
    )


def test_bitwise_identity_includes_dtype_and_signed_zero():
    a = np.zeros((2, 2), dtype=np.float32)
    assert bitwise_equal(a, a.copy())
    b = a.copy()
    b[0, 0] = -0.0
    assert np.array_equal(a, b) and not bitwise_equal(a, b)
    assert not bitwise_equal(a, a.astype(np.float64))


@pytest.mark.parametrize("zero", [False, True])
def test_boundary_preserves_inference_matches_analytic_and_excludes_padding(zero):
    source, model = v1_tests.tiny_source(zero=zero)
    previous, _ = explain_validation(source)
    state = runtime_state()
    outputs, diagnostic, checks, numerical = explain_boundary_validation(source, previous)
    assert all(checks.values()), checks
    assert state == runtime_state()
    assert numerical["comparisons"]["candidate_vs_frozen_inference"]["exact"]
    np.testing.assert_array_equal(
        outputs["feature_scores"], outputs["raw_maps"].reshape(4, 64)[:, :57]
    )
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.backends.nnpack.flags(enabled=True), torch.no_grad():
            activation = model.features[:3](torch.from_numpy(source.arrays["validation_images"]))
        a = activation.numpy().astype(np.float64)
        weights = (
            model.classifier.weight.detach()
            .numpy()
            .astype(np.float64)[source.predictions["predictions"]]
        )
        derivative = weights[:, :, None, None] * (a > 0) / 64
        reference = np.maximum(
            (derivative.mean((2, 3), keepdims=True) * a).sum(1, keepdims=True), 0
        )
        np.testing.assert_allclose(outputs["raw_maps"], reference, rtol=1e-5, atol=1e-7)
    finally:
        torch.set_num_threads(threads)
    assert len(diagnostic) == 4
    if zero:
        assert not outputs["raw_maps"].any() and not outputs["normalized_maps"].any()


@pytest.mark.parametrize("invalid", ["training", "shape", "targets", "class"])
def test_boundary_rejects_invalid_model_inputs(invalid):
    source, model = v1_tests.tiny_source()
    model.eval()
    x = torch.from_numpy(source.arrays["validation_images"])
    targets = torch.from_numpy(source.predictions["predictions"])
    if invalid == "training":
        model.train()
    elif invalid == "shape":
        x = x[:, :, :7, :]
    elif invalid == "targets":
        targets = targets.float()
    else:
        targets[:] = 2
    with pytest.raises(ValueError):
        boundary_grad_cam(model, x, targets)


@pytest.mark.parametrize("mode", ["timeout", "exception", "outer_no_grad"])
def test_runtime_state_restored_on_success_and_failure(monkeypatch, mode):
    import css700.interpretability.activation_boundary as impl

    source, _ = v1_tests.tiny_source()
    previous, _ = explain_validation(source)
    if mode == "exception":

        def fail(*args, **kwargs):
            raise RuntimeError("synthetic exception")

        monkeypatch.setattr(impl, "boundary_grad_cam", fail)
    with torch.backends.nnpack.flags(enabled=False), torch.no_grad():
        before = runtime_state()
        if mode == "outer_no_grad":
            _, _, checks, _ = explain_boundary_validation(source, previous)
            assert all(checks.values()), checks
        else:
            with pytest.raises((TimeoutError, RuntimeError)):
                explain_boundary_validation(
                    source, previous, max_seconds=1e-12 if mode == "timeout" else 120
                )
        assert runtime_state() == before


def test_exact_candidate_gate_rejects_even_one_ulp(monkeypatch):
    import css700.interpretability.activation_boundary as impl

    source, _ = v1_tests.tiny_source()
    previous, _ = explain_validation(source)
    original = impl.boundary_grad_cam

    def perturb(*args, **kwargs):
        result = original(*args, **kwargs)
        logits = result.logits.clone()
        logits[0, 0] = torch.nextafter(logits[0, 0], torch.tensor(float("inf")))
        return GradCAMResult(result.raw, result.normalized, result.target_classes, logits)

    monkeypatch.setattr(impl, "boundary_grad_cam", perturb)
    _, _, checks, _ = explain_boundary_validation(source, previous)
    assert checks["frozen_inference_logits_exact"]
    assert checks["original_gradient_logit_tolerance_passed"]
    assert not checks["candidate_logits_exact_frozen_inference"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("gradient_logit_atol", 1e-3),
        ("candidate_nnpack_enabled", False),
        ("candidate_logits", "approximately_equal"),
        ("test_evaluation", True),
        ("diagnostic_modes", []),
    ],
)
def test_frozen_v2_rejects_amendments(key, value):
    config = yaml.safe_load(Path("configs/spambase_grad_cam_validation_v2.yaml").read_text())
    validate_v2_config(config)
    config["policy"][key] = value
    with pytest.raises(ValueError):
        validate_v2_config(config)


@pytest.fixture
def held_fixture(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import css700.experiments.spambase_grad_cam as runner

    _, training_review, cam_path = v1_tests.reviewed_training.__wrapped__(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "_git_state", lambda: ("fixture", False))
    original = runner.explain_validation

    def synthetic_hold(*args, **kwargs):
        outputs, checks = original(*args, **kwargs)
        checks["gradient_logits_numerically_match"] = False
        return outputs, checks

    monkeypatch.setattr(runner, "explain_validation", synthetic_hold)
    result = runner.run_spambase_grad_cam(cam_path, tmp_path / "held-runs")
    assert result["status"] == "quality_gate_failed" and not result["errors"]
    root = Path(result["output"])
    review = {
        "review_version": "spambase_validation_grad_cam_v1_review_1",
        "acceptance": False,
        "technical_status": "quality_gate_failed",
        "test_evaluation_ready": False,
        "failed_checks": ["gradient_logits_numerically_match"],
        "execution_commit": "fixture",
        "run_dir": str(root),
        "file_sha256": {n: sha256((root / n).read_bytes()) for n in HELD_FILES},
        "training_review": str(training_review),
        "training_review_sha256": sha256(training_review.read_bytes()),
    }
    path = tmp_path / "held-review.yaml"
    path.write_text(yaml.safe_dump(review))
    config = {
        "protocol": "spambase_validation_grad_cam_v2",
        "training_review": str(training_review),
        "training_review_sha256": sha256(training_review.read_bytes()),
        "held_review": str(path),
        "held_review_sha256": sha256(path.read_bytes()),
        "policy": copy.deepcopy(V2_POLICY),
    }
    config_path = tmp_path / "v2.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path, config, path, review


@pytest.mark.parametrize("mode", ["ok", "dirty", "timeout", "mutated_input", "failed_gate"])
def test_v2_runner_preserves_failures_and_never_opens_test(
    held_fixture, monkeypatch, tmp_path, mode
):
    import css700.experiments.spambase_grad_cam_v2 as runner

    config_path, _, review_path, _ = held_fixture
    monkeypatch.setattr(runner, "_git_state", lambda: ("fixture", mode == "dirty"))
    original = runner.explain_boundary_validation

    def controlled(*args, **kwargs):
        if mode == "timeout":
            raise TimeoutError("synthetic budget")
        result = original(*args, **kwargs)
        if mode == "mutated_input":
            review_path.write_text(review_path.read_text() + "\n")
        if mode == "failed_gate":
            result[2]["candidate_logits_exact_frozen_inference"] = False
        return result

    monkeypatch.setattr(runner, "explain_boundary_validation", controlled)
    result = run_spambase_grad_cam_v2(config_path, tmp_path / "v2-runs")
    assert result["status"] == ("ok" if mode == "ok" else "quality_gate_failed"), result
    assert not result["test_evaluation_ready"] and not result["v1_acceptance_changed"]
    assert all(
        result[k] == 0
        for k in (
            "models_trained",
            "optimizer_steps",
            "scalers_refitted",
            "test_rows_transformed_or_scored",
        )
    )
    root = Path(result["output"])
    manifest = json.loads((root / "manifest.json").read_text())
    assert all(
        sha256((root / n).read_bytes()) == h for n, h in manifest["output_file_sha256"].items()
    )
    if mode in ("ok", "failed_gate", "mutated_input"):
        assert len(manifest["source_file_sha256"]) == 51
        assert (root / "grad_cam_arrays.npz").is_file() and (
            root / "numerical_logits.npz"
        ).is_file()
    if mode == "mutated_input":
        assert not result["checks"]["all_inputs_unchanged"]
    if mode == "ok":
        audit = subprocess.run(
            [sys.executable, "scripts/verify_spambase_grad_cam_v2.py", str(root)],
            check=True,
            capture_output=True,
            text=True,
        )
        verified = json.loads(audit.stdout)
        assert verified["candidate_logits_bitwise_equal_frozen_inference"]
        assert verified["analytic_reference_failed_cells"] == 0


@pytest.mark.parametrize("mode", ["acceptance", "checkpoint", "artifact"])
def test_held_loader_rejects_tampering(held_fixture, mode):
    _, config, path, review = held_fixture
    source = load_validation_checkpoint(
        Path(config["training_review"]), config["training_review_sha256"], {}
    )
    if mode == "acceptance":
        review["acceptance"] = True
    elif mode == "checkpoint":
        review["training_review_sha256"] = "wrong"
    else:
        (Path(review["run_dir"]) / "feature_map.json").write_text("[]")
    path.write_text(yaml.safe_dump(review))
    config["held_review_sha256"] = sha256(path.read_bytes())
    with pytest.raises(ValueError):
        load_held_v1(config, source, {})


def test_v2_config_pins_and_preserves_v1_tolerances():
    config = yaml.safe_load(Path("configs/spambase_grad_cam_validation_v2.yaml").read_text())
    validate_v2_config(config)
    for key in ("training_review", "held_review"):
        assert config[key + "_sha256"] == sha256(Path(config[key]).read_bytes())
    v1 = yaml.safe_load(Path("configs/spambase_grad_cam_validation_v1.yaml").read_text())
    assert all(config["policy"][k] == v for k, v in v1["policy"].items())
