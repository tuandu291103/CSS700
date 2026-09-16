"""Mechanism controls, source integrity and no evaluation-to-fit leakage."""

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from sklearn.linear_model import Ridge

from css700.evaluation.lime_locality import (
    VARIANTS,
    anchor_leverage,
    descriptive_summaries,
    fit_locality_variant,
    paired_contrasts,
    variant_inputs,
    verify_reference,
)
from css700.experiments.lime_locality import (
    inventory_digest,
    require_accepted_preflight,
    run_lime_locality,
    validate_config,
)
from css700.interpretability.lime_tabular import (
    LimeDesign,
    Neighborhood,
    fit_probability_lime,
)


def synthetic_design():
    rng = np.random.default_rng(51)
    fit_z = rng.integers(0, 2, (120, 3)).astype(float)
    fit_z[0] = 1
    eval_z = rng.integers(0, 2, (60, 3)).astype(float)
    neighborhoods = []
    for z in (fit_z, eval_z):
        inverse = rng.normal(size=z.shape)
        d = np.linalg.norm(z - 1, axis=1)
        w = np.exp(-(d**2) / (2 * 0.75**2 * 3))
        neighborhoods.append(Neighborhood(z, inverse, d, w))
    design = LimeDesign(*neighborhoods, metadata={"kernel_width": 0.75 * np.sqrt(3)})
    inputs = np.concatenate([n.inverse for n in neighborhoods])
    positive = 1 / (1 + np.exp(-inputs[:, 1]))
    return design, np.column_stack((1 - positive, positive)), np.array([0.8, 1.1, 1.4])


def test_config_cannot_silently_add_a_sweep_or_select_a_winner():
    config = yaml.safe_load(Path("configs/lime_locality_diagnostic_v1.yaml").read_text())
    validate_config(config)
    for changes in (
        {"selection_rule": "minimum_error"},
        {"narrow_width_multiplier": 0.1},
        {"new_model_calls": False},
        {"unknown": 5},
    ):
        with pytest.raises(ValueError, match="frozen locality"):
            validate_config({**config, **changes})


def test_reference_matches_existing_lime_adapter_and_rejects_corruption():
    pytest.importorskip("lime")
    design, p, scale = synthetic_design()
    model = SimpleNamespace(classes_=[0, 1], predict_proba=lambda _: p)
    original = fit_probability_lime(model, design)
    result = fit_locality_variant(design, p, scale, "reference")
    verify_reference(result, original, original["diagnostics"])
    corrupted = {**original, "coefficients": original["coefficients"] + 0.1}
    with pytest.raises(ValueError, match="reference"):
        verify_reference(result, corrupted, original["diagnostics"])
    with pytest.raises(ValueError, match="target"):
        verify_reference(
            result,
            original,
            {
                **original["diagnostics"],
                "target_class": 1 - original["diagnostics"]["target_class"],
            },
        )


def test_matched_mass_and_weight_penalty_scaling_equivalence():
    design, p, scale = synthetic_design()
    raw = variant_inputs(design, scale, "narrow_raw")
    matched = variant_inputs(design, scale, "narrow_matched_mass")
    np.testing.assert_allclose(matched["fit_weights"].sum(), design.fit.weights.sum(), atol=1e-12)
    multiplier = matched["weight_multiplier"]
    np.testing.assert_allclose(matched["fit_weights"], raw["fit_weights"] * multiplier)
    target = int(p[0].argmax())
    alternative_alpha = Ridge(alpha=1 / multiplier, solver="cholesky").fit(
        raw["fit_design"], p[:120, target], sample_weight=raw["fit_weights"]
    )
    result = fit_locality_variant(design, p, scale, "narrow_matched_mass")
    np.testing.assert_allclose(result["coefficients"], alternative_alpha.coef_, atol=1e-12)
    assert result["diagnostics"]["nonanchor_effective_sample_size"] == pytest.approx(
        fit_locality_variant(design, p, scale, "narrow_raw")["diagnostics"][
            "nonanchor_effective_sample_size"
        ]
    )


def test_continuous_control_keeps_neighbors_and_weights_but_centers_anchor():
    design, _, scale = synthetic_design()
    inputs = variant_inputs(design, scale, "continuous_same_neighbors")
    np.testing.assert_array_equal(inputs["fit_design"][0], np.zeros(3))
    np.testing.assert_array_equal(inputs["fit_weights"], design.fit.weights)
    np.testing.assert_allclose(
        inputs["evaluation_design"] * scale + design.fit.inverse[0], design.evaluation.inverse
    )
    with pytest.raises(ValueError, match="unregistered"):
        variant_inputs(design, scale, "best")
    with pytest.raises(ValueError, match="scale"):
        variant_inputs(design, np.zeros(3), "reference")


def test_common_reference_evaluation_and_no_response_leakage():
    design, p, scale = synthetic_design()
    result = fit_locality_variant(design, p, scale, "narrow_raw")
    target = int(p[0].argmax())
    residual = result["evaluation_surrogate_predictions"] - p[120:, target]
    expected = np.sqrt(np.average(residual**2, weights=design.evaluation.weights))
    assert result["diagnostics"]["reference_evaluation"]["weighted_rmse"] == pytest.approx(expected)
    altered = p.copy()
    altered[120:] = altered[120:, ::-1]
    other = fit_locality_variant(design, altered, scale, "narrow_raw")
    np.testing.assert_array_equal(other["coefficients"], result["coefficients"])
    assert other["diagnostics"]["intercept"] == result["diagnostics"]["intercept"]
    with pytest.raises(ValueError, match="probabilities"):
        fit_locality_variant(design, p * 2, scale, "reference")


def test_anchor_leverage_matches_finite_difference_and_normal_equations():
    design, p, scale = synthetic_design()
    for variant in VARIANTS:
        inputs = variant_inputs(design, scale, variant)
        x, w = inputs["fit_design"], inputs["fit_weights"]
        y = p[:120, 1]
        first = Ridge(alpha=1, solver="cholesky").fit(x, y, sample_weight=w)
        perturbed = y.copy()
        perturbed[0] += 1e-4
        second = Ridge(alpha=1, solver="cholesky").fit(x, perturbed, sample_weight=w)
        influence = (second.predict(x[:1])[0] - first.predict(x[:1])[0]) / 1e-4
        assert anchor_leverage(x, w) == pytest.approx(influence, abs=1e-9)
        xc = x - np.average(x, axis=0, weights=w)
        yc = y - np.average(y, weights=w)
        beta = np.linalg.solve(xc.T @ (w[:, None] * xc) + np.eye(3), xc.T @ (w * yc))
        np.testing.assert_allclose(first.coef_, beta, atol=1e-12)


def test_paired_identity_complete_conditions_and_descriptive_only():
    design, p, scale = synthetic_design()
    rows = [
        dict(
            sample_id=4,
            explainer_seed=11,
            family="random_forest",
            variant=v,
            **fit_locality_variant(design, p, scale, v)["diagnostics"],
        )
        for v in VARIANTS
    ]
    pairs = paired_contrasts(rows)
    assert len(pairs) == 3
    summary = descriptive_summaries(rows, pairs)
    assert len(summary["conditions"]) == 8  # All-descriptive and separate seed, not pooling CIs.
    assert "interval" not in str(summary["conditions"])
    with pytest.raises(ValueError, match="duplicate"):
        paired_contrasts(rows + [rows[0]])
    with pytest.raises(ValueError, match="incomplete"):
        paired_contrasts(rows[:-1])
    with pytest.raises(ValueError, match="target"):
        paired_contrasts(
            [rows[0], {**rows[1], "target_class": 1 - rows[0]["target_class"]}, *rows[2:]]
        )


def test_inventory_detects_changed_added_and_removed_files(tmp_path):
    file = tmp_path / "source.json"
    file.write_text('{"accepted":true}')
    original, _ = inventory_digest(tmp_path)
    file.write_text('{"accepted":false}')
    assert inventory_digest(tmp_path)[0] != original
    file.write_text('{"accepted":true}')
    extra = tmp_path / "extra.json"
    extra.write_text("{}")
    assert inventory_digest(tmp_path)[0] != original
    extra.unlink()
    assert inventory_digest(tmp_path)[0] == original


def test_acceptance_reader_rejects_failed_incomplete_or_test_sources():
    config = yaml.safe_load(Path("configs/lime_locality_diagnostic_v1.yaml").read_text())
    manifest = dict(
        status="ok",
        worktree_dirty_at_start=False,
        commit_sha=config["preflight_commit"],
        config_sha256=config["preflight_config_sha256"],
        anchor_training_ids=config["anchor_training_ids"],
        scope="training_anchors_and_synthetic_neighbors_only",
    )
    checks = dict.fromkeys(
        (
            "clean_worktree",
            "no_cnn_runtime",
            "all_48_training_fits_completed",
            "six_same_seed_repeats_passed",
            "only_registered_training_anchors",
            "source_files_unchanged",
        ),
        True,
    )
    summary = dict(
        status="ok",
        errors=[],
        completed_training_fits=48,
        test_explanations_generated=0,
        checks=checks,
    )
    require_accepted_preflight(manifest, summary, config)
    for change in (
        {"status": "failed"},
        {"checks": {}},
        {"completed_training_fits": 47},
        {"test_explanations_generated": 1},
    ):
        with pytest.raises(ValueError, match="registered accepted"):
            require_accepted_preflight(manifest, {**summary, **change}, config)
    bad = copy.deepcopy(manifest)
    bad["anchor_training_ids"][0] = 999
    with pytest.raises(ValueError, match="registered accepted"):
        require_accepted_preflight(bad, summary, config)


def test_dirty_run_retains_failure_without_source_reads(tmp_path, monkeypatch):
    import css700.experiments.lime_locality as module

    monkeypatch.setattr(module, "_git_state", lambda: ("dirty-fixture", True))
    monkeypatch.setattr(
        module, "load_preflight", lambda *args: pytest.fail("read source from dirty run")
    )
    result = run_lime_locality(Path("configs/lime_locality_diagnostic_v1.yaml"), tmp_path)
    assert result["status"] == "quality_gate_failed"
    assert result["new_model_calls"] == result["test_explanations_generated"] == 0
    assert result["selected_variant"] is None
    assert (Path(result["output"]) / "summary.json").exists()
