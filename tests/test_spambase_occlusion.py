"""Synthetic-only prospective Occlusion, source-pin and independent-audit tests."""

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
import test_spambase_faithfulness as deletion_tests
import test_spambase_grad_cam as cam_tests
import test_spambase_grad_cam_v2 as v2_tests

from css700.evaluation.grouped_occlusion import grouped_rank_summary, plot_agreement, rank_records
from css700.experiments.spambase_audit import sha256
from css700.experiments.spambase_occlusion import (
    DELETION_FILES,
    POLICY,
    check_deletion_k1,
    run_spambase_occlusion,
    validate_config,
)
from css700.interpretability.spambase_occlusion import (
    evaluate_occlusion,
    replace_one,
    signed_effects,
)


def test_single_feature_replacement_and_unchanged_values():
    images = np.arange(128, dtype=np.float32).reshape(2, 1, 8, 8)
    ref = np.zeros((1, 8, 8), np.float32)
    for coordinate in (56, 0, 7):
        values = replace_one(images, coordinate, ref)
        for i in range(2):
            expected = images[i].copy()
            expected.flat[coordinate] = 0
            np.testing.assert_array_equal(values[i], expected)
    assert images[1, 0, 0, 0] == 64


def test_signed_effects_keep_negative_and_use_frozen_targets():
    original = np.array([[2, 1], [0, 3]], np.float32)
    logits = np.array([[[[4, 0], [1, 5]], [[4, 2], [5, 4]]]] * 2, np.float32)
    effects = signed_effects(logits, original, np.array([0, 1]))
    np.testing.assert_array_equal(effects["logit"][0], [[-2, 1], [1, -1]])
    assert effects["probability"][0, 0, 0] < 0
    assert effects["probability"][0, 0, 1] > 0


def test_absolute_rank_ties_and_constant_rows_are_explicit():
    x = np.tile(np.arange(12, dtype=float), (3, 1))
    x[1] = 0
    x[2] = 1
    x[2, :2] = 5
    scores = np.stack([-x, x])
    records = rank_records(scores, x, np.array([0, 1, 0]), np.array([0, 0, 0]), np.arange(3))
    assert len(records) == 9
    assert records[0]["spearman"] == pytest.approx(1)
    assert records[0]["top_5_overlap_fraction"] == 1
    assert records[1]["constant_vector"] and records[1]["spearman"] is None
    assert records[1]["top_5_overlap_fraction"] is None
    assert records[2]["left_top_5_cutoff_tie"]


def test_bootstrap_complete_groups_and_undefined_eligibility():
    x = np.tile(np.arange(12, dtype=float), (4, 1))
    y = x.copy()
    y[2] = y[2, ::-1]
    y[3] = 0
    labels, targets, groups = (
        np.array([0, 0, 1, 1]),
        np.zeros(4, int),
        np.array(["a", "a", "b", "c"]),
    )
    records = rank_records(np.stack([x, x]), y, targets, labels, np.arange(4))
    result = grouped_rank_summary(records, groups, labels, targets, resamples=20, minimum_groups=2)
    all_rows = [
        r for r in result if r["group"] == "all" and r["comparison"] == "occlusion_zero_vs_grad_cam"
    ]
    assert all_rows[0]["eligible_rows"] == 3 and all_rows[0]["eligible_feature_groups"] == 2
    assert all_rows[0]["spearman"]["mean"] == pytest.approx(1 / 3)
    assert all_rows[1]["spearman"]["mean"] == pytest.approx(0)
    draws = np.random.default_rng(20261006).integers(3, size=(20, 3))
    den = np.array([2, 1, 0])[draws].sum(1)
    valid = den > 0
    values = np.array([2, -1, 0])[draws[valid]].sum(1) / den[valid]
    assert all_rows[0]["spearman"]["valid_bootstrap_draws"] == int(valid.sum())
    if valid.all():
        np.testing.assert_allclose(
            all_rows[0]["spearman"]["ci95"], np.quantile(values, [0.025, 0.975])
        )
    else:
        assert all_rows[0]["spearman"]["ci95"] is None
    assert all(
        r["spearman"]["ci95"] is None
        for r in result
        if r["group"] == "true_1" and r["comparison"] != "occlusion_zero_vs_median"
    )


def test_all_constant_rows_have_no_undefined_statistics_disguised_as_zero():
    x = np.zeros((2, 12))
    records = rank_records(np.stack([x, x]), x, np.array([0, 0]), np.array([0, 1]), np.arange(2))
    summaries = grouped_rank_summary(
        records, np.array(["a", "b"]), np.array([0, 1]), np.array([0, 0]), resamples=10
    )
    assert all(
        r["spearman"] == {"mean": None, "ci95": None, "valid_bootstrap_draws": 0} for r in summaries
    )


@pytest.mark.parametrize(
    "mode", ["ok", "timeout", "bad_original", "padding", "bad_reference", "zero_model"]
)
def test_execution_guards_and_state_restoration(tmp_path, mode):
    source, _ = cam_tests.tiny_source(zero=mode == "zero_model")
    coords = np.arange(57)
    references = np.zeros((2, 1, 8, 8), np.float32)
    references[1, 0].flat[:57] = 0.5
    if mode == "bad_original":
        source.predictions["logits"][0, 0] += 1
    if mode == "padding":
        coords[-1] = 63
    if mode == "bad_reference":
        references[1, 0, 7, 7] = 1
    before = v2_tests.runtime_state()
    partial = tmp_path / "partial.npy"
    if mode in ("ok", "zero_model"):
        logits, original, checks, counts = evaluate_occlusion(
            source, coords, references, partial_path=partial
        )
        assert all(checks.values()), checks
        assert logits.shape == (2, 4, 57, 2) and counts["forward_calls"] == 115
        assert counts["conceptual_model_evaluations"] == 460
        if mode == "zero_model":
            assert not signed_effects(logits, original, source.predictions["predictions"])[
                "logit"
            ].any()
    else:
        with pytest.raises((ValueError, TimeoutError)):
            evaluate_occlusion(
                source,
                coords,
                references,
                partial_path=partial,
                max_seconds=1e-12 if mode == "timeout" else 120,
            )
        if mode in ("timeout", "bad_original"):
            assert np.isnan(np.load(partial)).any()
    assert v2_tests.runtime_state() == before


def test_k1_consistency_detects_one_changed_value():
    logits = np.arange(2 * 3 * 4 * 2, dtype=np.float32).reshape(2, 3, 4, 2)
    ranks = np.array([[[0, 1, 2, 3]] * 3, [[3, 0, 1, 2]] * 3])
    previous = np.zeros((2, 2, 3, 3, 2), np.float32)
    for r in range(2):
        for i, order in enumerate(ranks):
            previous[r, i, 1] = logits[r, np.arange(3), order[:, 0]]
    design = {"steps": np.array([0, 1, 4]), "rankings": ranks}
    assert check_deletion_k1(logits, design, previous)
    previous[1, 1, 1, 2, 0] += 1
    assert not check_deletion_k1(logits, design, previous)


@pytest.fixture
def accepted_sources(tmp_path, monkeypatch):
    import css700.experiments.spambase_faithfulness as runner

    cam_path, cam_review = deletion_tests.accepted_cam.__wrapped__(tmp_path, monkeypatch)
    policy = {
        **copy.deepcopy(runner.POLICY),
        "random_repeats": 2,
        "bootstrap_resamples": 10,
        "deletion_steps": [0, 1, 57],
    }
    monkeypatch.setattr(runner, "POLICY", policy)
    monkeypatch.setattr(runner, "_git_state", lambda: ("fixture", False))
    cfg = {
        "protocol": "spambase_validation_grad_cam_deletion_v1",
        "grad_cam_review": str(cam_path),
        "grad_cam_review_sha256": sha256(cam_path.read_bytes()),
        "policy": policy,
    }
    path = tmp_path / "deletion.yaml"
    path.write_text(yaml.safe_dump(cfg))
    result = runner.run_spambase_faithfulness(path, tmp_path / "deletion")
    assert result["status"] == "ok", result
    root = Path(result["output"])
    review = {
        "review_version": "spambase_validation_grad_cam_deletion_v1_review_1",
        "acceptance": True,
        "acceptance_scope": "seed_42_validation_grad_cam_conditional_deletion_evidence_only",
        "technical_status": "ok",
        "decision": "go_to_prospective_input_level_explanation_protocol",
        "test_evaluation_ready": False,
        "independent_calculation": {"status": "passed"},
        "execution_commit": "fixture",
        "run_dir": str(root),
        "file_sha256": {n: sha256((root / n).read_bytes()) for n in DELETION_FILES},
        "grad_cam_review": str(cam_path),
        "grad_cam_review_sha256": sha256(cam_path.read_bytes()),
        "faithfulness_config": str(path),
        "faithfulness_config_sha256": sha256(path.read_bytes()),
    }
    review_path = tmp_path / "deletion-review.yaml"
    review_path.write_text(yaml.safe_dump(review))
    config = {
        "protocol": "spambase_validation_occlusion_v1",
        "grad_cam_review": str(cam_path),
        "grad_cam_review_sha256": sha256(cam_path.read_bytes()),
        "faithfulness_review": str(review_path),
        "faithfulness_review_sha256": sha256(review_path.read_bytes()),
        "policy": copy.deepcopy(POLICY),
    }
    config_path = tmp_path / "occlusion.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path, config, review_path, review


@pytest.mark.parametrize("mode", ["ok", "dirty", "timeout", "mutation", "rejected", "corrupt"])
def test_runner_provenance_failure_retention_and_independent_replay(
    accepted_sources, tmp_path, monkeypatch, mode
):
    import css700.experiments.spambase_occlusion as runner

    config_path, config, review_path, review = accepted_sources
    monkeypatch.setattr(runner, "_git_state", lambda: ("fixture", mode == "dirty"))
    if mode == "rejected":
        review["acceptance"] = False
        review_path.write_text(yaml.safe_dump(review))
        config["faithfulness_review_sha256"] = sha256(review_path.read_bytes())
        config_path.write_text(yaml.safe_dump(config))
    if mode == "corrupt":
        (Path(review["run_dir"]) / "summary.json").write_text("{}")
    original = runner.evaluate_occlusion

    def evaluate(*args, **kwargs):
        if mode == "timeout":
            kwargs["max_seconds"] = 1e-12
        values = original(*args, **kwargs)
        if mode == "mutation":
            review_path.write_text(review_path.read_text() + "\n")
        return values

    monkeypatch.setattr(runner, "evaluate_occlusion", evaluate)
    result = run_spambase_occlusion(config_path, tmp_path / "outputs")
    assert result["status"] == ("ok" if mode == "ok" else "quality_gate_failed"), result
    assert (
        not result["test_evaluation_ready"] and not result["agreement_magnitude_is_acceptance_gate"]
    )
    assert all(
        result[k] == 0
        for k in (
            "models_trained",
            "scalers_refitted",
            "optimizer_steps",
            "test_rows_transformed_or_scored",
            "integrated_gradients_rows",
        )
    )
    root = Path(result["output"])
    manifest = json.loads((root / "manifest.json").read_text())
    assert all(
        sha256((root / n).read_bytes()) == h for n, h in manifest["output_file_sha256"].items()
    )
    if mode in ("timeout", "mutation"):
        assert (root / "perturbed_logits.npy").is_file()
    if mode == "ok":
        audit = subprocess.run(
            [sys.executable, "scripts/verify_spambase_occlusion.py", str(root)],
            check=True,
            capture_output=True,
            text=True,
        )
        assert json.loads(audit.stdout)["all_perturbation_and_k1_logits_bitwise_exact"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("split", "test"),
        ("ranking", "positive_only"),
        ("batch_size", 512),
        ("replacements", ["training_median"]),
        ("refit_or_retrain", True),
    ],
)
def test_policy_rejects_unfrozen_changes(key, value):
    config = yaml.safe_load(Path("configs/spambase_occlusion_v1.yaml").read_text())
    validate_config(config)
    config["policy"][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_tracked_config_pins_both_accepted_reviews():
    config = yaml.safe_load(Path("configs/spambase_occlusion_v1.yaml").read_text())
    validate_config(config)
    for key in ("grad_cam_review", "faithfulness_review"):
        assert sha256(Path(config[key]).read_bytes()) == config[key + "_sha256"]


def test_distribution_plot_retains_counts(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from matplotlib.axes import Axes

    titles = []
    original = Axes.set_title

    def capture(self, text, **kwargs):
        titles.append(text)
        return original(self, text, **kwargs)

    monkeypatch.setattr(Axes, "set_title", capture)
    x = np.tile(np.arange(12), (2, 1))
    records = rank_records(np.stack([x, x]), x, np.array([0, 0]), np.array([0, 1]), np.arange(2))
    plot_agreement(tmp_path / "plot.png", records)
    assert sum("2/2 eligible rows" in t for t in titles) == 3
