import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from css700.evaluation.local_agreement import paired_rank_rows, summarize_rank_rows
from css700.experiments.tree_shap import _validate_config, run_tree_shap_experiment
from css700.experiments.tree_shap_sources import named_feature_matrix, prediction_arrays
from css700.interpretability.forest_adapter import floor_float32_thresholds
from css700.interpretability.tree_routing import random_forest_routing_audit
from css700.interpretability.tree_shap import (
    binary_probabilities,
    normalize_binary_shap,
    reconstruction_checks,
    training_background,
    verified_numeric_booster,
)


def test_background_only_training_and_fixed_subset() -> None:
    data = np.arange(48).reshape(12, 4)
    ids = np.arange(12) + 100
    all_values, all_ids = training_background(data, ids, np.array([0, 1]))
    np.testing.assert_array_equal(all_values, data)
    np.testing.assert_array_equal(all_ids, ids)
    subset, selected = training_background(data, ids, np.array([0, 1]), size=5)
    expected = np.sort(np.random.default_rng(2026).permutation(12)[:5])
    np.testing.assert_array_equal(selected, ids[expected])
    np.testing.assert_array_equal(subset, data[expected])
    with pytest.raises(ValueError, match="overlap"):
        training_background(data, ids, np.array([100]))
    with pytest.raises(ValueError, match="unique"):
        training_background(data, np.ones(12, dtype=int), np.array([100]))
    with pytest.raises(ValueError, match="invalid"):
        training_background(data * np.nan, ids, np.array([0]))


def test_binary_class_zero_and_independent_reconstruction() -> None:
    raw = np.array([[0.1, 0.2, -0.1]])
    values, base, rule = normalize_binary_shap(raw, 0.4, samples=1, features=3, family="xgboost")
    assert rule.startswith("class_0")
    np.testing.assert_array_equal(values[0, :, 0], -raw[0])
    residuals, checks = reconstruction_checks(
        values, base, np.array([[0.4, 0.6]]), base, atol=1e-5, rtol=1e-5
    )
    assert all(checks.values())
    assert np.abs(residuals).max() < 1e-10
    _, bad = reconstruction_checks(values, base, np.array([[0.1, 0.9]]), base, atol=1e-5, rtol=1e-5)
    assert not bad["probability_reconstruction"]
    with pytest.raises(ValueError, match="shapes"):
        normalize_binary_shap(
            values.transpose(0, 2, 1), base, samples=1, features=3, family="random_forest"
        )
    with pytest.raises(ValueError, match="class order"):
        binary_probabilities(SimpleNamespace(classes_=[1, 0]), raw)


def test_numeric_booster_guard_rejects_real_categories() -> None:
    tree = {
        "split_type": [1],
        "categories": [0],
        "categories_nodes": [],
        "categories_segments": [],
        "categories_sizes": [],
    }
    payload = {"learner": {"gradient_booster": {"model": {"trees": [tree]}}}}
    booster = SimpleNamespace(feature_types=None, save_raw=lambda **_: json.dumps(payload))
    with pytest.raises(ValueError, match="categorical"):
        verified_numeric_booster(SimpleNamespace(get_booster=lambda: booster))


def test_routing_guard_distinguishes_background_and_threshold_precision() -> None:
    low = float(np.nextafter(np.float32(16), np.float32(np.inf)))
    high = float(np.nextafter(np.float32(low), np.float32(np.inf)))
    threshold = (low + high) / 2
    tree = SimpleNamespace(
        children_left=np.array([1, -1, -1]),
        feature=np.array([0, -2, -2]),
        threshold=np.array([threshold, -2, -2]),
    )
    model = SimpleNamespace(n_features_in_=1, estimators_=[SimpleNamespace(tree_=tree)])
    # Midpoint rounds UP in SHAP's independent Node: the high input changes side.
    assert float(np.float32(threshold)) == high
    report = random_forest_routing_audit(model, np.array([[low], [high]]))
    assert not report["passed"]
    assert report["mismatch_counts"] == {"background_role": 1, "query_role": 1}
    assert report["checked_comparisons_per_role"] == 2
    assert report["examples"][0]["stored_threshold"] == threshold
    # A representable threshold still needs background input rounding semantics.
    tree.threshold[0] = high
    report = random_forest_routing_audit(model, np.array([[np.nextafter(high, np.inf)]]))
    assert report["mismatch_counts"] == {"background_role": 1, "query_role": 0}
    assert random_forest_routing_audit(model, np.array([[low], [high]]))["passed"]


def test_floor_thresholds_preserve_float32_boundary_routing() -> None:
    lower = np.array([-16.000004, -1.0, -1e-30, 0.0, 1e-30, 1.0, 16.000002], dtype=np.float32)
    upper = np.nextafter(lower, np.float32(np.inf))
    thresholds = (lower.astype(float) + upper.astype(float)) / 2
    original = thresholds.copy()
    floored = floor_float32_thresholds(thresholds)
    np.testing.assert_array_equal(thresholds, original)
    np.testing.assert_array_equal(floored, lower.astype(float))
    for values in (lower.astype(float), upper.astype(float)):
        np.testing.assert_array_equal(values <= floored, values <= thresholds)
    np.testing.assert_array_equal(floor_float32_thresholds(lower), lower)
    for bad in (np.array([np.nan]), np.array([np.inf]), np.array([-1e300]), np.array([1e300])):
        with pytest.raises(ValueError):
            floor_float32_thresholds(bad)


def test_numerical_repair_config_preserves_scientific_settings() -> None:
    configs = Path(__file__).resolve().parents[1] / "configs"
    original = yaml.safe_load((configs / "tree_shap_seed_42.yaml").read_text())
    repaired = yaml.safe_load((configs / "tree_shap_seed_42_v1_1.yaml").read_text())
    _validate_config(original)
    _validate_config(repaired)
    assert repaired.pop("protocol_version") == "tree-shap-probability-v1.1"
    original.pop("protocol_version")
    assert repaired["shap"].pop("random_forest_adapter") == "float32_equivalent_v1"
    assert original == repaired
    original["protocol_version"] = "tree-shap-probability-v1.1"
    with pytest.raises(ValueError, match="adapter"):
        _validate_config(original)


def test_source_matrix_identity_and_duplicate_checks() -> None:
    ids, labels, predictions = np.array([20]), np.array([0]), np.array([1])
    rows = [
        dict(
            sample_id="20",
            feature_index=str(j),
            feature_name=name,
            predicted_label="1",
            target_class="1",
            true_label="0",
            value=str(j),
        )
        for j, name in enumerate(("a", "b"))
    ]
    np.testing.assert_array_equal(
        named_feature_matrix(rows[::-1], ids, ("a", "b"), predictions, labels, value_field="value"),
        [[0, 1]],
    )
    for bad_rows in (
        rows + [rows[0]],
        rows[:1],
        [dict(rows[0], feature_name="wrong"), rows[1]],
        [dict(rows[0], target_class="0"), rows[1]],
    ):
        with pytest.raises(ValueError):
            named_feature_matrix(
                bad_rows, ids, ("a", "b"), predictions, labels, value_field="value"
            )
    with pytest.raises(ValueError, match="sample IDs"):
        prediction_arrays([], ids, labels)


def test_agreement_exclusions_and_bootstrap_are_explicit() -> None:
    scores = np.tile(np.arange(12), (12, 1))
    right = scores.copy()
    right[1] = 0
    lp = np.zeros(12, dtype=int)
    rp = lp.copy()
    rp[0] = 1
    rows = paired_rank_rows(scores, right, lp, rp, lp, np.arange(12), comparison="a vs b")
    assert rows[0]["exclusion_reason"] == "different_target"
    assert rows[1]["spearman"] is None and rows[1]["top_5_overlap_fraction"] is None
    assert rows[2]["top_5_overlap_fraction"] == 1
    first = summarize_rank_rows(rows, resamples=20, seed=42)
    assert first == summarize_rank_rows(rows, resamples=20, seed=42)
    assert first[0]["eligible_samples"] == 10
    assert first[0]["spearman"]["mean_95_percentile_interval"] == [1.0, 1.0]
    ties = paired_rank_rows(scores, scores, lp, lp, lp, np.arange(12), comparison="ties")
    assert not ties[0]["left_top_5_cutoff_tie"]


def test_failure_bundle_is_retained_and_append_only(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("css700.experiments.tree_shap._git_state", lambda: ("test", True))
    root = Path(__file__).resolve().parents[1]
    first = run_tree_shap_experiment(root / "configs/tree_shap_seed_42.yaml", tmp_path)
    second = run_tree_shap_experiment(root / "configs/tree_shap_seed_42.yaml", tmp_path)
    assert first["status"] == "quality_gate_failed"
    assert first["output"] != second["output"]
    assert (Path(first["output"]) / "manifest.json").is_file()
    assert json.loads((Path(first["output"]) / "summary.json").read_text())["errors"]


def test_tree_model_smoke_in_isolated_process(tmp_path: Path) -> None:
    if any(importlib.util.find_spec(name) is None for name in ("shap", "xgboost", "matplotlib")):
        pytest.skip("requires optional tree explanation dependencies")
    root = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "PYTHONPATH": str(root / "src"),
        "MPLCONFIGDIR": str(tmp_path),
        "XDG_CACHE_HOME": str(tmp_path),
    }
    result = subprocess.run(
        [sys.executable, str(root / "tests/tree_shap_worker.py")],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
