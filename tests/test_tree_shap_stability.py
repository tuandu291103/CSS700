import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from css700.experiments.tree_shap import _csv, _json, _validate_config
from css700.experiments.tree_shap_sources import file_sha256
from css700.experiments.tree_shap_stability import (
    run_tree_shap_stability,
    stability_tables,
    validate_analysis_config,
)
from css700.experiments.tree_shap_stability_sources import (
    load_accepted_tree_run,
    scientific_settings,
)

ROOT = Path(__file__).resolve().parents[1]


def test_replica_configs_change_only_registered_seed_and_sources() -> None:
    reference = yaml.safe_load((ROOT / "configs/tree_shap_seed_42_v1_1.yaml").read_text())
    for seed in (7, 2026):
        replica = yaml.safe_load((ROOT / f"configs/tree_shap_seed_{seed}_v1_1.yaml").read_text())
        _validate_config(replica)
        assert replica["model_seed"] == seed
        assert f"model-seed-{seed}__replicate" in replica["grid_run"]
        assert scientific_settings(replica) == scientific_settings(reference)
        assert set(replica["model_sha256"]) == {"random_forest", "xgboost"}
        assert all(len(value) == 64 for value in replica["model_sha256"].values())
    for seed in (True, "../bad", 0, 13):
        with pytest.raises(ValueError, match="registered"):
            _validate_config({**reference, "model_seed": seed})


def fake_accepted_run(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    config = yaml.safe_load((ROOT / "configs/tree_shap_seed_42_v1_1.yaml").read_text())
    root = tmp_path / "accepted"
    root.mkdir()
    expected = tmp_path / "frozen.yaml"
    expected.write_text(yaml.safe_dump(config))
    (root / "config.yaml").write_bytes(expected.read_bytes())
    n, d = 12, 12
    ids, labels = np.arange(200, 200 + n), np.ones(n, dtype=int)
    bundle = SimpleNamespace(
        x_train=np.arange(120 * d).reshape(120, d) / 7,
        train_indices=np.arange(120),
        validation_indices=np.arange(150, 160),
        test_indices=ids,
        y_test=labels,
        feature_names=tuple(f"f{i}" for i in range(d)),
        target_names=("malignant", "benign"),
        scaler_mean=np.zeros(d),
        scaler_scale=np.ones(d),
    )
    probabilities = np.tile([0.2, 0.8], (n, 1))
    original = {
        "bundle": bundle,
        "file_hashes": {},
        "tree_predictions": {
            family: {"predictions": labels, "probabilities": probabilities}
            for family in ("random_forest", "xgboost")
        },
    }
    monkeypatch.setattr(
        "css700.experiments.tree_shap_stability_sources.load_sources", lambda _: original
    )
    manifest = {
        "status": "ok",
        "worktree_dirty_at_start": False,
        "config_sha256": file_sha256(expected),
        "source_file_sha256": {},
        "feature_names": bundle.feature_names,
        "class_names": bundle.target_names,
        "split_fingerprint": config["expected_split_sha256"],
    }
    _json(root / "manifest.json", manifest)
    np.savez(
        root / "splits.npz",
        train=bundle.train_indices,
        validation=bundle.validation_indices,
        test=ids,
    )
    np.savez(root / "preprocessing.npz", mean=bundle.scaler_mean, scale=bundle.scaler_scale)
    summary = {
        "status": "ok",
        "errors": [],
        "checks": {"all_four_conditions_accepted": True},
        "conditions": {},
    }
    weights = np.arange(1, d + 1) / sum(range(1, d + 1))
    values = np.tile(np.stack((-0.3 * weights, 0.3 * weights), axis=-1)[None, :, :], (n, 1, 1))
    base = np.array([0.5, 0.5])
    for bg in ("train_all", "train_100"):
        positions = (
            np.arange(120)
            if bg == "train_all"
            else np.sort(np.random.default_rng(2026).permutation(120)[:100])
        )
        background = bundle.x_train[positions]
        np.savez(
            root / f"background_{bg}.npz",
            values=background,
            sample_ids=bundle.train_indices[positions],
        )
        for family in ("random_forest", "xgboost"):
            name = f"{family}__{bg}"
            condition = root / name
            condition.mkdir()
            checks = {
                k: True
                for k in (
                    "probability_reconstruction",
                    "base_matches_background_probability_mean",
                    "binary_contributions_complementary",
                )
            }
            if family == "random_forest":
                checks["copied_forest_matches_original_probabilities"] = True
                _json(
                    condition / "routing_audit.json",
                    {"passed": True, "adapted_background_routing_mismatches": 0},
                )
                np.savez(condition / "adapter_thresholds.npz", tree_0=np.array([1.0]))
            report = {
                "status": "ok",
                "test_checks": checks,
                "smoke_checks": {
                    **checks,
                    "warmup_passed": True,
                    "repeat_passed": True,
                    "repeatable": True,
                },
            }
            _json(condition / "checks.json", report)
            summary["conditions"][name] = report
            effective = (
                background.astype(np.float32).astype(float)
                if family == "random_forest"
                else background
            )
            np.savez(
                condition / "explainer_background.npz",
                values=effective,
                sample_ids=bundle.train_indices[positions],
            )
            np.savez(
                condition / "test_attributions.npz",
                values=values,
                base=base,
                probabilities=probabilities,
                residuals=values.sum(axis=1) + base - probabilities,
            )
            rows = [
                {
                    "sample_id": int(sample),
                    "target_class": 1,
                    "true_label": 1,
                    "feature_index": j,
                    "feature_name": bundle.feature_names[j],
                    "signed_attribution": float(values[i, j, 1]),
                }
                for i, sample in enumerate(ids)
                for j in range(d)
            ]
            _csv(condition / "target_features.csv", rows)
    _json(root / "summary.json", summary)
    return root, expected


def test_accepted_reader_revalidates_raw_and_named_scores(tmp_path: Path, monkeypatch) -> None:
    root, expected = fake_accepted_run(tmp_path, monkeypatch)
    hashes = {}
    run = load_accepted_tree_run(root, expected, hashes)
    assert run["seed"] == 42 and len(run["conditions"]) == 4
    assert len(hashes) > 20
    assert run["conditions"]["random_forest__train_all"]["scores"].shape == (12, 12)


@pytest.mark.parametrize(
    "corruption", ["status", "config", "effective_background", "raw_values", "named_values"]
)
def test_accepted_reader_rejects_corrupt_bundles(
    tmp_path: Path, monkeypatch, corruption: str
) -> None:
    root, expected = fake_accepted_run(tmp_path, monkeypatch)
    condition = root / "random_forest__train_all"
    if corruption == "status":
        manifest = json.loads((root / "manifest.json").read_text())
        _json(root / "manifest.json", {**manifest, "status": "quality_gate_failed"})
    elif corruption == "config":
        with (root / "config.yaml").open("a") as handle:
            handle.write("# changed after acceptance\n")
    elif corruption in ("effective_background", "raw_values"):
        path = condition / (
            "explainer_background.npz"
            if corruption == "effective_background"
            else "test_attributions.npz"
        )
        with np.load(path) as data:
            arrays = {k: v.copy() for k, v in data.items()}
        arrays["values"].flat[0] += 0.1
        np.savez(path, **arrays)
    else:
        path = condition / "target_features.csv"
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["signed_attribution"] = "0.1234"
        _csv(path, rows)
    with pytest.raises(ValueError):
        load_accepted_tree_run(root, expected, {})


def test_seed_pairs_are_separate_and_identity_checked() -> None:
    config = yaml.safe_load((ROOT / "configs/tree_shap_stability.yaml").read_text())
    validate_analysis_config(config)
    runs = {}
    for seed in (7, 42, 2026):
        scores = np.tile(np.arange(12), (12, 1))
        predictions = np.ones(12, dtype=int)
        if seed == 7:
            predictions[0] = 0
        conditions = {
            f"{family}__{bg}": {
                "scores": scores,
                "predictions": predictions,
                "probabilities": np.eye(2)[predictions],
            }
            for family in config["families"]
            for bg in config["backgrounds"]
        }
        runs[seed] = {
            "sample_ids": np.arange(12),
            "labels": np.ones(12, dtype=int),
            "conditions": conditions,
        }
    rows, predictive, consensus = stability_tables(runs, config)
    assert len(rows) == 12 * 12 and len(predictive) == 6 and len(consensus) == 2
    assert len({r["comparison"] for r in rows}) == 12
    assert sum(not r["primary_eligible"] for r in rows) == 8
    assert all(r["all_three_same_prediction"] == 11 for r in consensus)
    bad = copy.deepcopy(runs)
    bad[2026]["sample_ids"] = bad[2026]["sample_ids"][::-1]
    with pytest.raises(ValueError, match="identity"):
        stability_tables(bad, config)
    with pytest.raises(ValueError, match="exactly"):
        stability_tables({7: runs[7]}, config)


def test_analysis_dirty_failure_is_append_only(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("css700.experiments.tree_shap_stability._git_state", lambda: ("test", True))
    config = ROOT / "configs/tree_shap_stability.yaml"
    first = run_tree_shap_stability([], config, tmp_path)
    second = run_tree_shap_stability([], config, tmp_path)
    assert first["status"] == second["status"] == "quality_gate_failed"
    assert first["output"] != second["output"]
    assert (Path(first["output"]) / "manifest.json").is_file()
