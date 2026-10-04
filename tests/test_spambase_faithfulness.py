"""Synthetic tests only: no real validation perturbation or test evaluation."""

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
import test_spambase_grad_cam as cam_tests
import test_spambase_grad_cam_v2 as v2_tests

from css700.evaluation.grouped_deletion import deletion_effects, grouped_summaries
from css700.experiments.spambase_audit import sha256
from css700.experiments.spambase_explanation_source import CAM_FILES, load_deletion_source
from css700.experiments.spambase_faithfulness import (
    POLICY,
    run_spambase_faithfulness,
    validate_config,
)
from css700.interpretability.spambase_deletion import (
    deletion_rankings,
    evaluate_deletions,
    replace_ranked,
)


def test_rankings_stable_complete_group_shared_and_reproducible():
    scores = np.array([[1, 2, 2, 0], [1, 2, 2, 0], [0, 0, 0, 0]], dtype=float)
    groups = np.array(["b", "b", "a"])
    ranks = deletion_rankings(scores, groups, repeats=10)
    assert ranks.shape == (12, 3, 4)
    np.testing.assert_array_equal(ranks[0, 0], [1, 2, 0, 3])
    np.testing.assert_array_equal(ranks[1, 0], [3, 0, 1, 2])
    assert np.all(np.sort(ranks, axis=-1) == np.arange(4))
    np.testing.assert_array_equal(ranks[2:, 0], ranks[2:, 1])
    np.testing.assert_array_equal(ranks, deletion_rankings(scores, groups, repeats=10))
    assert not np.array_equal(ranks[2:, 0], ranks[2:, 2])


def test_exact_cumulative_masks_preserve_padding_under_permuted_coordinates():
    x = np.arange(18, dtype=np.float32).reshape(3, 1, 2, 3)
    coordinates = np.array([4, 0, 3, 1])
    order = np.array([[3, 0, 1, 2], [0, 1, 2, 3], [2, 0, 3, 1]])
    ref = np.full((1, 2, 3), -1, np.float32)
    for k in range(5):
        result = replace_ranked(x, order, coordinates, ref, k).reshape(3, 6)
        for i in range(3):
            expected = x[i].flatten().copy()
            expected[coordinates[order[i, :k]]] = -1
            np.testing.assert_array_equal(result[i], expected)
            assert np.sum(result[i] == -1) == k
        np.testing.assert_array_equal(result[:, [2, 5]], x.reshape(3, 6)[:, [2, 5]])


def test_signed_aopc_uses_irregular_steps_and_does_not_clip_negative_effects():
    original = np.array([[3.0, 0.0], [0.0, 2.0]], np.float32)
    logits = np.broadcast_to(original, (2, 4, 3, 2, 2)).copy()
    logits[:, :, 1, 0, 0] = 4.0
    logits[:, :, 2, 0, 0] = 5.0
    logits[:, :, 1, 1, 1] = 1.0
    logits[:, :, 2, 1, 1] = 0.0
    drops, aopc = deletion_effects(logits, original, np.array([0, 1]), [0, 1, 4])
    np.testing.assert_array_equal(drops["logit"][0, 0], [[0, 0], [-1, 1], [-2, 2]])
    np.testing.assert_allclose(aopc["logit"][0, 0], [-1.25, 1.25])
    assert (aopc["probability"][:, :, 0] < 0).all()


def test_group_bootstrap_keeps_duplicates_and_paired_contrasts():
    groups = np.array(["a", "a", "b", "c"])
    original = np.zeros((4, 2), np.float32)
    targets, labels = np.zeros(4, np.int64), np.array([0, 0, 0, 1])
    logits = np.zeros((2, 4, 2, 4, 2), np.float32)
    logits[:, 0, -1, :, 0] = [-2, -2, -8, -14]
    curves, summaries, _ = grouped_summaries(
        logits, original, targets, labels, groups, [0, 57], resamples=100, minimum_groups=2
    )
    rows = [
        r
        for r in summaries
        if r["group"] == "all"
        and r["replacement"] == "zero"
        and r["metric"] == "logit"
        and r["order_or_contrast"] == "top_minus_random"
    ]
    assert rows[0]["mean"] == pytest.approx(3.25)
    assert rows[1]["mean"] == pytest.approx(4.0)
    rng = np.random.default_rng(20261005)
    positions = rng.integers(3, size=(100, 3))
    row_distribution = np.array(
        [np.array([2, 4, 7])[p].sum() / np.array([2, 1, 1])[p].sum() for p in positions]
    )
    group_distribution = np.array([np.array([1, 4, 7])[p].mean() for p in positions])
    np.testing.assert_allclose(rows[0]["ci95"], np.quantile(row_distribution, [0.025, 0.975]))
    np.testing.assert_allclose(rows[1]["ci95"], np.quantile(group_distribution, [0.025, 0.975]))
    assert all(r["ci95"] is None for r in summaries if r["group"] == "incorrect")
    for r in summaries:
        if r["group"] == "incorrect":
            assert r["rows"] == r["feature_groups"] == 1
    assert len(curves) > 0


def test_no_effect_is_a_valid_scientific_result():
    original = np.array([[1.0, 0.0], [0.0, 1.0]], np.float32)
    logits = np.broadcast_to(original, (2, 4, 2, 2, 2)).copy()
    _, summaries, _ = grouped_summaries(
        logits,
        original,
        np.array([0, 1]),
        np.array([0, 1]),
        np.array(["a", "b"]),
        [0, 57],
        resamples=10,
    )
    assert all(r["mean"] == 0 for r in summaries)


def test_figure_title_retains_sample_count(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    from css700.evaluation.grouped_deletion import plot_deletion_curves

    titles = []
    original = Figure.suptitle

    def capture(self, text, **kwargs):
        titles.append(text)
        return original(self, text, **kwargs)

    monkeypatch.setattr(Figure, "suptitle", capture)
    curves = [
        {
            "group": "all",
            "weighting": "row_primary",
            "replacement": ref,
            "order": order,
            "deleted_features": 0,
            "mean_logit_drop": 0.0,
        }
        for ref in ("zero", "training_median")
        for order in ("top", "random", "bottom")
    ]
    plot_deletion_curves(tmp_path / "figure.png", curves, rows=4, random_repeats=2)
    assert "4 rows;" in titles[0] and "2 group-shared orders" in titles[0]


@pytest.mark.parametrize(
    "mode", ["ok", "timeout", "bad_original", "bad_ranking", "padding_coordinate"]
)
def test_execution_guards_partial_files_and_runtime_restoration(tmp_path, mode):
    source, _ = cam_tests.tiny_source()
    scores = np.ones((4, 57))
    ranks = deletion_rankings(scores, np.array(["a", "b", "c", "d"]), repeats=2)
    coords = np.arange(57)
    references = np.zeros((2, 1, 8, 8), np.float32)
    references[1, 0].flat[:57] = 0.5
    if mode == "bad_original":
        source.predictions["logits"][0, 0] += 1
    if mode == "bad_ranking":
        ranks[0, 0, 0] = ranks[0, 0, 1]
    if mode == "padding_coordinate":
        coords[-1] = 63
    before = v2_tests.runtime_state()
    path = tmp_path / "logits.npy"
    if mode == "ok":
        outputs, original, checks, counts = evaluate_deletions(
            source, ranks, coords, references, [0, 5, 57], partial_path=path
        )
        assert all(checks.values()), checks
        assert outputs.shape == (2, 4, 3, 4, 2)
        np.testing.assert_array_equal(original, source.predictions["logits"])
        assert counts["conceptual_model_evaluations"] == 4 * (1 + 2 * 4 * 2)
    else:
        with pytest.raises((ValueError, TimeoutError)):
            evaluate_deletions(
                source,
                ranks,
                coords,
                references,
                [0, 5, 57],
                partial_path=path,
                max_seconds=1e-12 if mode == "timeout" else 300,
            )
        if mode in ("timeout", "bad_original"):
            assert path.is_file()
            assert np.isnan(np.load(path)).any()
    assert v2_tests.runtime_state() == before


@pytest.mark.parametrize(
    "key,value",
    [
        ("split", "test"),
        ("random_repeats", 3),
        ("targets", "true_class"),
        ("padding", "mask"),
        ("refit_or_retrain", True),
    ],
)
def test_frozen_config_rejects_changes(key, value):
    config = yaml.safe_load(Path("configs/spambase_faithfulness_v1.yaml").read_text())
    validate_config(config)
    config["policy"][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.fixture
def accepted_cam(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import css700.experiments.spambase_grad_cam_v2 as runner

    cam_config, config, _, _ = v2_tests.held_fixture.__wrapped__(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "_git_state", lambda: ("fixture", False))
    result = runner.run_spambase_grad_cam_v2(cam_config, tmp_path / "cam")
    assert result["status"] == "ok", result
    root = Path(result["output"])
    review = {
        "review_version": "spambase_validation_grad_cam_v2_review_1",
        "acceptance": True,
        "acceptance_scope": "seed_42_validation_grad_cam_tracing_only",
        "technical_status": "ok",
        "decision": "go_to_validation_faithfulness_protocol",
        "test_evaluation_ready": False,
        "independent_calculation": {"status": "passed"},
        "execution_commit": "fixture",
        "run_dir": str(root),
        "file_sha256": {n: sha256((root / n).read_bytes()) for n in CAM_FILES},
        "training_review": config["training_review"],
        "training_review_sha256": config["training_review_sha256"],
    }
    path = tmp_path / "cam-review.yaml"
    path.write_text(yaml.safe_dump(review))
    return path, review


def test_source_uses_training_median_groups_and_no_refit(accepted_cam, monkeypatch):
    from sklearn.preprocessing import StandardScaler

    from css700.encoding import GridEncoder

    path, _ = accepted_cam

    def forbidden(*args, **kwargs):
        raise AssertionError("no fitting or new encoding")

    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    monkeypatch.setattr(GridEncoder, "transform", forbidden)
    hashes = {}
    source, scores, coordinates, references, groups = load_deletion_source(
        path, sha256(path.read_bytes()), hashes
    )
    with np.load(Path(source.source_run) / "inputs.npz") as a:
        np.testing.assert_array_equal(references[1], np.median(a["train_images"], axis=0))
    assert scores.shape == (len(groups), 57)
    assert len(np.unique(coordinates)) == 57 and not references[0].any()
    assert len(hashes) == 62


@pytest.mark.parametrize("mode", ["rejected", "corrupt_file"])
def test_source_rejects_unaccepted_or_corrupt_explanations(accepted_cam, mode):
    path, review = accepted_cam
    if mode == "rejected":
        review["acceptance"] = False
        path.write_text(yaml.safe_dump(review))
    else:
        (Path(review["run_dir"]) / "feature_map.json").write_text("[]")
    with pytest.raises(ValueError):
        load_deletion_source(path, sha256(path.read_bytes()), {})


@pytest.mark.parametrize("mode", ["ok", "dirty", "timeout", "mutation"])
def test_runner_keeps_test_closed_and_failures_append_only(
    accepted_cam, tmp_path, monkeypatch, mode
):
    import css700.experiments.spambase_faithfulness as runner

    path, _ = accepted_cam
    policy = {
        **copy.deepcopy(POLICY),
        "random_repeats": 2,
        "bootstrap_resamples": 10,
        "deletion_steps": [0, 5, 57],
    }
    monkeypatch.setattr(runner, "POLICY", policy)
    monkeypatch.setattr(runner, "_git_state", lambda: ("fixture", mode == "dirty"))
    config = {
        "protocol": "spambase_validation_grad_cam_deletion_v1",
        "grad_cam_review": str(path),
        "grad_cam_review_sha256": sha256(path.read_bytes()),
        "policy": policy,
    }
    config_path = tmp_path / "deletion.yaml"
    config_path.write_text(yaml.safe_dump(config))
    original = runner.evaluate_deletions

    def execute(*args, **kwargs):
        if mode == "timeout":
            kwargs["max_seconds"] = 1e-12
        result = original(*args, **kwargs)
        if mode == "mutation":
            path.write_text(path.read_text() + "\n")
        return result

    monkeypatch.setattr(runner, "evaluate_deletions", execute)
    result = run_spambase_faithfulness(config_path, tmp_path / "outputs")
    assert result["status"] == ("ok" if mode == "ok" else "quality_gate_failed"), result
    assert not result["effect_direction_is_acceptance_gate"] and not result["test_evaluation_ready"]
    assert all(
        result[k] == 0
        for k in (
            "models_trained",
            "scalers_refitted",
            "test_rows_transformed_or_scored",
            "optimizer_steps",
            "new_explanations_generated",
        )
    )
    root = Path(result["output"])
    manifest = json.loads((root / "manifest.json").read_text())
    assert all(
        sha256((root / n).read_bytes()) == h for n, h in manifest["output_file_sha256"].items()
    )
    if mode != "dirty":
        assert (root / "design.npz").is_file()
        assert (root / "perturbed_logits.npy").is_file()
    if mode == "mutation":
        assert not result["checks"]["all_inputs_unchanged"]
    if mode == "ok":
        audit = subprocess.run(
            [sys.executable, "scripts/verify_spambase_faithfulness.py", str(root)],
            check=True,
            text=True,
            capture_output=True,
        )
        assert json.loads(audit.stdout)["all_perturbation_logits_bitwise_exact"]


def test_tracked_config_pins_accepted_grad_cam_review():
    c = yaml.safe_load(Path("configs/spambase_faithfulness_v1.yaml").read_text())
    validate_config(c)
    assert sha256(Path(c["grad_cam_review"]).read_bytes()) == c["grad_cam_review_sha256"]
