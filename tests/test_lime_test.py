"""Synthetic-only test-pilot checks; no real held-out explanations in tests."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from css700.evaluation.lime_comparison import (
    FAMILIES,
    SEEDS,
    comparison_tables,
    feature_rows,
    fit_quality_summary,
)
from css700.evaluation.local_agreement import summarize_rank_rows
from css700.experiments.lime_test import run_lime_test, verify_fit
from css700.experiments.lime_test_sources import load_test_sources, verify_predictions
from css700.experiments.tree_shap_sources import file_sha256
from css700.interpretability.lime_tabular import (
    build_lime_design,
    fit_diagnostics,
    fit_probability_lime,
)

CONFIG = Path("configs/lime_seed_42_v1.yaml")


def comparison_fixture(n=12, d=12):
    ids = np.arange(500, 500 + n)
    names = tuple(f"f{j}" for j in range(d))
    labels = np.ones(n, dtype=int)
    predictions = {f: labels.copy() for f in FAMILIES}
    predictions["xgboost"][0] = 0
    metric = fit_diagnostics(np.array([0.0, 1.0]), np.array([2.0, -1.0]), np.ones(2))
    records, features = [], []
    for family in FAMILIES:
        for seed in SEEDS:
            for i, sample in enumerate(ids):
                record = {
                    "family": family,
                    "explainer_seed": seed,
                    "sample_id": int(sample),
                    "partition": "test",
                    "true_label": int(labels[i]),
                    "target_class": int(predictions[family][i]),
                    "intercept": 0.5,
                    "anchor_model_probability": 0.8,
                    "anchor_surrogate_prediction": 2.0,
                    "anchor_absolute_error": 1.2,
                    "fit": metric,
                    "evaluation": metric,
                }
                records.append(record)
                features.extend(
                    feature_rows(
                        record,
                        np.arange(d, dtype=float),
                        {
                            "feature_names": names,
                            "anchor_bin_ids": [2] * d,
                            "sample_id": int(sample),
                            "explainer_seed": seed,
                        },
                    )
                )
    shap = {
        "sample_ids": ids,
        "labels": labels,
        "feature_names": names,
        "conditions": {
            f"{f}__train_all": {
                "scores": np.tile(np.arange(d, dtype=float), (n, 1)),
                "predictions": predictions[f],
            }
            for f in FAMILIES
        },
    }
    return features, records, shap


def test_comparison_counts_primary_seeds_and_poor_fit_retention():
    features, records, shap = comparison_fixture()
    rows = comparison_tables(features, records, shap)
    assert len(rows) == 15 * 12 and len({r["comparison"] for r in rows}) == 15
    assert sum(r["primary_lime_shap"] for r in rows) == 24
    assert sum(not r["primary_eligible"] for r in rows) == 3
    assert all(r["left_anchor_absolute_error"] == 1.2 for r in rows)
    assert all(r["left_evaluation_weighted_r2"] < 0 for r in rows)
    assert all(r["spearman"] == 1 for r in rows)
    summaries = summarize_rank_rows(rows, resamples=1000, seed=2026)
    assert len(summaries) == 90
    for r in summaries:
        interval = r["spearman"]["mean_95_percentile_interval"]
        assert (interval is None) == (r["eligible_samples"] < 10)
    assert any(r["total_samples"] == 0 for r in summaries)
    assert summaries == summarize_rank_rows(rows, resamples=1000, seed=2026)


@pytest.mark.parametrize(
    "change", ["duplicate_fit", "target", "label", "feature", "missing", "seed"]
)
def test_comparison_rejects_broken_identity(change):
    features, records, shap = comparison_fixture()
    if change == "duplicate_fit":
        records.append(records[0])
    elif change == "target":
        records[0]["target_class"] = 0
    elif change == "label":
        records[0]["true_label"] = 0
    elif change == "feature":
        features[0]["feature_name"] = "wrong"
    elif change == "missing":
        features.pop()
    else:
        features[0]["explainer_seed"] = 99
    with pytest.raises(ValueError):
        comparison_tables(features, records, shap)


def test_constant_vector_excluded_and_cutoff_ties_retained():
    features, records, shap = comparison_fixture()
    for r in features:
        if r["family"] == "random_forest" and r["explainer_seed"] == 11:
            if r["sample_id"] == 500:
                r["signed_coefficient"] = 0.0
            elif r["sample_id"] == 501 and r["feature_index"] in (6, 7):
                r["signed_coefficient"] = 7.0
    rows = comparison_tables(features, records, shap)
    assert sum(r["constant_vector"] for r in rows) == 4
    assert all(r["spearman"] is None for r in rows if r["constant_vector"])
    assert any(r["left_top_5_cutoff_tie"] for r in rows if r["sample_id"] == 501)


def test_named_export_signed_values_bins_and_stable_ties():
    _, records, _ = comparison_fixture()
    metadata = {
        "sample_id": 500,
        "explainer_seed": 11,
        "feature_names": ["a", "b", "c"],
        "anchor_bin_ids": [0, 1, 2],
    }
    rows = feature_rows(records[0], np.array([-2.0, 2.0, 0.0]), metadata)
    assert [r["absolute_rank"] for r in rows] == [1, 2, 3]
    assert [r["signed_coefficient"] for r in rows] == [-2.0, 2.0, 0.0]
    assert [r["anchor_bin_id"] for r in rows] == [0, 1, 2]
    with pytest.raises(ValueError, match="identity"):
        feature_rows(records[0], np.ones(3), {**metadata, "sample_id": 501})


def test_fit_summary_keeps_negative_null_zero_groups_and_separate_seeds():
    _, records, _ = comparison_fixture()
    records[0] = copy.deepcopy(records[0])
    records[0]["evaluation"]["weighted_r2"] = None
    summaries = fit_quality_summary(records)
    assert len(summaries) == 30
    r = next(
        r
        for r in summaries
        if r["family"] == "random_forest" and r["explainer_seed"] == 11 and r["group"] == "all"
    )
    assert r["samples"] == 12 and r["evaluation_undefined_r2_count"] == 1
    assert r["evaluation_weighted_r2"]["defined_samples"] == 11
    assert r["evaluation_weighted_r2"]["median"] < 0
    assert r["anchor_out_of_bounds_count"] == 12
    assert any(
        r["samples"] == 0 and r["anchor_absolute_error"]["median"] is None for r in summaries
    )


def test_prediction_gate_checks_probabilities_and_targets():
    p = np.array([[0.2, 0.8], [0.7, 0.3]])
    expected = {"probabilities": p, "predictions": p.argmax(axis=1)}
    assert verify_predictions(p, expected) == 0
    for bad in (p[::-1], np.full_like(p, np.nan), p[:1], p + 0.01):
        with pytest.raises(ValueError, match="frozen"):
            verify_predictions(bad, expected)


def test_independent_fit_gate_catches_corruption():
    pytest.importorskip("lime")
    training = np.random.default_rng(1).normal(size=(60, 12))
    design = build_lime_design(
        training,
        np.arange(60),
        np.arange(60, 70),
        training[0],
        tuple(f"f{i}" for i in range(12)),
        sample_id=60,
        explainer_seed=11,
        fit_samples=100,
        evaluation_samples=50,
    )
    model = SimpleNamespace(
        classes_=[0, 1],
        predict_proba=lambda x: np.column_stack(
            (0.2 + 0.5 * (x[:, 0] > 0), 0.8 - 0.5 * (x[:, 0] > 0))
        ),
    )
    result = fit_probability_lime(model, design)
    p = model.predict_proba(training[:1])
    expected = {"probabilities": p, "predictions": p.argmax(axis=1)}
    target = int(p.argmax(axis=1)[0])
    verify_fit(result, design, expected, target)
    for change in ("coef", "evaluation", "anchor", "target"):
        bad = copy.deepcopy(result)
        if change == "coef":
            bad["coefficients"][0] += 0.1
        elif change == "evaluation":
            bad["diagnostics"]["evaluation"]["weighted_rmse"] += 0.1
        elif change == "anchor":
            bad["diagnostics"]["anchor_absolute_error"] += 0.1
        else:
            bad["diagnostics"]["target_class"] = 1 - target
        with pytest.raises(ValueError):
            verify_fit(bad, design, expected, target)


def test_failed_preflight_stops_source_loader(monkeypatch):
    import yaml

    def reject(*_):
        raise ValueError("rejected preflight")

    monkeypatch.setattr("css700.experiments.lime_test_sources.load_preflight", reject)
    with pytest.raises(ValueError, match="rejected preflight"):
        load_test_sources(yaml.safe_load(CONFIG.read_text()), {})


def test_dirty_runs_preserve_failure_without_reading_sources(tmp_path, monkeypatch):
    monkeypatch.setattr("css700.experiments.lime_test._git_state", lambda: ("synthetic", True))

    def never(*_):
        pytest.fail("sources must not load in dirty worktree")

    monkeypatch.setattr("css700.experiments.lime_test.load_test_sources", never)
    first, second = [run_lime_test(CONFIG, tmp_path) for _ in range(2)]
    assert first["output"] != second["output"]
    for r in (first, second):
        assert r["status"] == "quality_gate_failed" and r["generated_test_fits"] == 0
        assert (Path(r["output"]) / "summary.json").is_file()


@pytest.mark.parametrize("mutate_source", [False, True])
def test_synthetic_runner_full_export_counts_and_final_hash_gate(
    tmp_path, monkeypatch, mutate_source
):
    pytest.importorskip("lime")
    import css700.experiments.lime_test as runner

    rng = np.random.default_rng(8)
    names = tuple(f"f{i}" for i in range(30))
    training, testing = rng.normal(size=(397, 30)), rng.normal(size=(86, 30))
    bundle = SimpleNamespace(
        x_train=training,
        x_test=testing,
        train_indices=np.arange(397),
        validation_indices=np.arange(397, 483),
        test_indices=np.arange(483, 569),
        feature_names=names,
        target_names=("a", "b"),
        scaler_mean=np.zeros(30),
        scaler_scale=np.ones(30),
        y_test=np.ones(86, dtype=int),
    )
    model = SimpleNamespace(
        classes_=[0, 1],
        predict_proba=lambda x: np.column_stack(
            (0.2 + 0.5 * (x[:, 0] > 0), 0.8 - 0.5 * (x[:, 0] > 0))
        ),
    )
    p = model.predict_proba(testing)
    predictions = {f: {"probabilities": p, "predictions": p.argmax(axis=1)} for f in FAMILIES}
    shap = {
        "sample_ids": bundle.test_indices,
        "labels": bundle.y_test,
        "feature_names": names,
        "manifest": {"commit_sha": "synthetic"},
        "conditions": {
            f"{f}__train_all": {**predictions[f], "scores": np.tile(np.arange(30), (86, 1))}
            for f in FAMILIES
        },
    }
    watched = tmp_path / "input.txt"
    watched.write_text("unchanged")

    def sources(_, hashes):
        hashes[str(watched)] = file_sha256(watched)
        return {
            "bundle": bundle,
            "shap": shap,
            "preflight_manifest": {
                "packages": {},
                "lime_source_sha256": {},
                "commit_sha": "synthetic",
            },
            "tree_manifest": {"commit_sha": "synthetic"},
            "model_paths": dict.fromkeys(FAMILIES, "synthetic"),
            "tree_predictions": predictions,
        }

    def small_design(*args, **kwargs):
        return build_lime_design(*args, **{**kwargs, "fit_samples": 40, "evaluation_samples": 12})

    def tables(*args):
        rows = comparison_tables(*args)
        if mutate_source:
            watched.write_text("changed")
        return rows

    monkeypatch.setattr(runner, "_git_state", lambda: ("synthetic", False))
    monkeypatch.setattr(runner, "sys", SimpleNamespace(modules={}))
    monkeypatch.setattr(runner, "load_test_sources", sources)
    monkeypatch.setattr(runner.joblib, "load", lambda _: model)
    monkeypatch.setattr(runner, "build_lime_design", small_design)
    monkeypatch.setattr(runner, "comparison_tables", tables)
    result = run_lime_test(CONFIG, tmp_path)
    assert result["errors"] == []
    assert result["status"] == ("quality_gate_failed" if mutate_source else "ok")
    assert result["generated_test_fits"] == result["verified_test_fits"] == 516
    assert result["named_feature_rows"] == 15480
    assert result["paired_comparison_rows"] == 1290
    root = Path(result["output"])
    assert len(json.loads((root / "agreement_summary.json").read_text())) == 90
    assert len(json.loads((root / "fit_quality_summary.json").read_text())) == 30
    assert result["checks"]["source_files_unchanged"] is not mutate_source
