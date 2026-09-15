"""LIME protocol and attribution-identity checks without importing the CNN runtime."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from sklearn.linear_model import Ridge

from css700.experiments.lime_preflight import run_lime_preflight, validate_lime_config
from css700.interpretability.lime_tabular import (
    build_lime_design,
    derived_seed,
    fit_diagnostics,
    fit_probability_lime,
    make_explainer,
)


def fixture_design(seed=11):
    pytest.importorskip("lime")
    training = np.random.default_rng(5).normal(size=(80, 3))
    kwargs = dict(
        training=training,
        training_ids=np.arange(80),
        heldout_ids=np.array([80, 81]),
        row=training[0],
        names=("a", "b", "c"),
        sample_id=0,
        explainer_seed=seed,
        fit_samples=500,
        evaluation_samples=100,
    )
    return build_lime_design(**kwargs), kwargs


def test_registered_lime_config_rejects_ignored_settings():
    config = yaml.safe_load(Path("configs/lime_seed_42_v1.yaml").read_text())
    validate_lime_config(config)
    for change in (
        {"feature_selection": "auto"},
        {"fit_samples": 1000},
        {"unknown": 1},
        {"ridge_alpha": True},
        {"model_seed": 7},
    ):
        with pytest.raises(ValueError, match="configuration"):
            validate_lime_config({**config, **change})


def test_stream_identity_and_role_separation():
    seeds = {
        derived_seed(seed, sample, role)
        for seed in (11, 29, 47)
        for sample in (0, 1)
        for role in ("fit", "evaluation")
    }
    assert len(seeds) == 12
    assert derived_seed(11, 0, "fit") == derived_seed(11, 0, "fit")
    with pytest.raises(ValueError):
        derived_seed(True, 0, "fit")


def test_native_neighborhood_repeats_and_uses_training_only():
    design, kwargs = fixture_design()
    repeat = build_lime_design(**kwargs)
    other, _ = fixture_design(29)
    assert design.fit.design.shape == (500, 3)
    assert design.evaluation.design.shape == (100, 3)
    assert design.metadata == repeat.metadata
    np.testing.assert_array_equal(design.fit.inverse, repeat.fit.inverse)
    np.testing.assert_array_equal(design.evaluation.inverse, repeat.evaluation.inverse)
    assert not np.array_equal(design.fit.inverse, other.fit.inverse)
    assert not np.array_equal(design.fit.inverse[1:101], design.evaluation.inverse)
    assert np.all(np.isin(design.fit.design, [0, 1]))
    np.testing.assert_allclose(
        design.fit.weights, np.exp(-(design.fit.distances**2) / (2 * 0.75**2 * 3))
    )
    with pytest.raises(ValueError, match="overlap"):
        build_lime_design(**{**kwargs, "heldout_ids": np.array([0])})
    with pytest.raises(ValueError, match="invalid LIME"):
        build_lime_design(**{**kwargs, "names": ("a", "a", "c")})
    with pytest.raises(ValueError, match="constant"):
        build_lime_design(**{**kwargs, "training": np.ones((80, 3))})
    with pytest.raises(ValueError, match="invalid LIME"):
        build_lime_design(**{**kwargs, "row": np.array([np.nan, 0, 1])})


def test_lime_matches_public_api_and_weighted_ridge_for_class_zero():
    design, kwargs = fixture_design()
    model = SimpleNamespace(
        classes_=np.array([0, 1]),
        predict_proba=lambda x: np.column_stack(
            (0.2 + 0.5 * (x[:, 1] > 0), 0.8 - 0.5 * (x[:, 1] > 0))
        ),
    )
    result = fit_probability_lime(model, design, target=0)
    native = make_explainer(kwargs["training"], kwargs["names"], derived_seed(11, 0, "fit"))
    public = native.explain_instance(
        kwargs["row"],
        model.predict_proba,
        labels=(0,),
        top_labels=None,
        num_features=3,
        num_samples=500,
        model_regressor=Ridge(alpha=1, solver="cholesky"),
    )
    expected = np.array([v for _, v in sorted(public.as_map()[0])])
    np.testing.assert_allclose(result["coefficients"], expected, atol=1e-10, rtol=1e-8)
    assert result["diagnostics"]["target_class"] == 0
    assert abs(result["coefficients"][1]) > 0.1
    # Independent weighted, centered normal equations with alpha=1, free intercept.
    x, w = design.fit.design, design.fit.weights
    y = model.predict_proba(design.fit.inverse)[:, 0]
    centered_x = x - np.average(x, weights=w, axis=0)
    centered_y = y - np.average(y, weights=w)
    beta = np.linalg.solve(
        centered_x.T @ (w[:, None] * centered_x) + np.eye(3), centered_x.T @ (w * centered_y)
    )
    np.testing.assert_allclose(result["coefficients"], beta, atol=1e-10, rtol=1e-8)
    opposite = fit_probability_lime(model, design, target=1)
    np.testing.assert_allclose(opposite["coefficients"], -beta, atol=1e-10, rtol=1e-8)
    with pytest.raises(ValueError, match="class order"):
        fit_probability_lime(SimpleNamespace(classes_=[1, 0]), design)


def test_constant_predictor_and_poor_fit_are_not_false_acceptance_claims():
    design, _ = fixture_design()
    model = SimpleNamespace(
        classes_=[0, 1], predict_proba=lambda x: np.tile([0.3, 0.7], (len(x), 1))
    )
    result = fit_probability_lime(model, design)
    np.testing.assert_allclose(result["coefficients"], 0, atol=1e-10)
    assert result["diagnostics"]["evaluation"]["weighted_r2"] is None
    assert result["diagnostics"]["anchor_absolute_error"] < 1e-10
    poor = fit_diagnostics(np.array([0, 1, 0]), np.array([2, -1, 2]), np.ones(3))
    assert poor["weighted_r2"] < 0
    assert poor["surrogate_out_of_probability_bounds_fraction"] == 1
    with pytest.raises(ValueError):
        fit_diagnostics(np.ones(3), np.ones(3), -np.ones(3))


def test_independent_evaluation_budget_does_not_change_fitted_coefficients():
    design, kwargs = fixture_design()
    larger_evaluation = build_lime_design(**{**kwargs, "evaluation_samples": 200})
    np.testing.assert_array_equal(design.fit.inverse, larger_evaluation.fit.inverse)
    model = SimpleNamespace(
        classes_=[0, 1],
        predict_proba=lambda x: np.column_stack(
            (0.2 + 0.5 * (x[:, 1] > 0), 0.8 - 0.5 * (x[:, 1] > 0))
        ),
    )
    a = fit_probability_lime(model, design)
    b = fit_probability_lime(model, larger_evaluation)
    np.testing.assert_array_equal(a["coefficients"], b["coefficients"])
    assert a["diagnostics"]["fit"] == b["diagnostics"]["fit"]


def test_unreviewed_lime_release_fails_before_private_sampler(monkeypatch):
    import css700.interpretability.lime_tabular as module

    monkeypatch.setattr(module, "version", lambda _: "unreviewed")
    with pytest.raises(ValueError, match="inspected release"):
        make_explainer(np.arange(12).reshape(4, 3), ("a", "b", "c"), 11)


def test_preflight_retains_dirty_failure_without_loading_models(tmp_path, monkeypatch):
    import css700.experiments.lime_preflight as module

    monkeypatch.setattr(module, "_git_state", lambda: ("dirty-fixture", True))
    result = run_lime_preflight(Path("configs/lime_seed_42_v1.yaml"), tmp_path)
    assert result["status"] == "quality_gate_failed"
    assert result["test_explanations_generated"] == 0
    assert result["completed_training_fits"] == 0
    assert (Path(result["output"]) / "summary.json").exists()
