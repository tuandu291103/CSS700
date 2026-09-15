"""Training-only compatibility check; intentionally no held-out LIME execution."""

import importlib
import json
import platform
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import yaml
from threadpoolctl import threadpool_limits

from css700.experiments.som_mapping_audit import _git_state
from css700.experiments.tree_shap import _json
from css700.experiments.tree_shap_sources import file_sha256, load_sources
from css700.interpretability.lime_tabular import build_lime_design, fit_probability_lime


def validate_lime_config(config: dict) -> None:
    fixed = {
        "protocol_version": "lime-tabular-probability-v1",
        "source_config": "configs/tree_shap_seed_42_v1_1.yaml",
        "source_config_sha256": "40a1ea7d1fd8c9594d74256391dfec635ca717798fe41c72a2a28e7376c3e71f",
        "accepted_shap_run": (
            "artifacts/20260912-171418-618975__breast-cancer__tree-shap__seed-42__pilot"
        ),
        "model_seed": 42,
        "data_split_seed": 42,
        "families": ["random_forest", "xgboost"],
        "lime_version": "0.2.0.1",
        "representation": "training_quartile_membership",
        "feature_selection": "none",
        "feature_count": 30,
        "kernel_width_multiplier": 0.75,
        "distance_metric": "euclidean",
        "ridge_alpha": 1.0,
        "ridge_solver": "cholesky",
        "fit_samples": 5000,
        "evaluation_samples": 2000,
        "explainer_seeds": [11, 29, 47],
        "primary_explainer_seed": 11,
        "seed_derivation": "numpy_SeedSequence_explainer_seed_sample_id_role_uint32",
        "preflight_training_rows": 8,
        "primary_target": "predicted_class",
        "ranking": "absolute_coefficient",
        "fit_quality_filter": "none_report_all",
        "bootstrap_resamples": 1000,
        "bootstrap_seed": 2026,
        "minimum_interval_group": 10,
        "repeat_atol": 1e-10,
        "repeat_rtol": 1e-8,
        "r2_variance_floor": 1e-15,
    }
    # Strict schema: no ignored overrides or accidental bool-as-integer settings.
    if json.dumps(config, sort_keys=True) != json.dumps(fixed, sort_keys=True):
        raise ValueError("unsupported or changed frozen LIME v1 configuration")


def _save_design(root: Path, design) -> None:
    root.mkdir(parents=True, exist_ok=False)
    arrays = {}
    for role in ("fit", "evaluation"):
        neighborhood = getattr(design, role)
        for field in ("design", "inverse", "distances", "weights"):
            arrays[f"{role}_{field}"] = getattr(neighborhood, field)
    np.savez_compressed(root / "neighborhoods.npz", **arrays)
    _json(root / "metadata.json", design.metadata)


def run_lime_preflight(config_path: Path, output_dir: Path) -> dict:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text())
    validate_lime_config(config)
    commit, dirty = _git_state()
    root = output_dir / (
        datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
        + "__breast-cancer__lime__model-seed-42__training-preflight"
    )
    root.mkdir(parents=True, exist_ok=False)
    (root / "config.yaml").write_bytes(config_path.read_bytes())
    hashes = {str(config_path.resolve()): file_sha256(config_path)}
    manifest = {
        "run_id": root.name,
        "commit_sha": commit,
        "worktree_dirty_at_start": dirty,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_sha256": file_sha256(config_path),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "scope": "training_anchors_and_synthetic_neighbors_only",
        "committee_approval_claimed": False,
        "status": "running",
    }
    _json(root / "manifest.json", manifest)
    checks = {"clean_worktree": not dirty, "no_cnn_runtime": "torch" not in sys.modules}
    errors, records = [], []
    try:
        if not all(checks.values()):
            raise ValueError("preflight requires a clean commit and no CNN runtime")
        source_path = Path(config["source_config"])
        hashes[str(source_path.resolve())] = file_sha256(source_path)
        if file_sha256(source_path) != config["source_config_sha256"]:
            raise ValueError("source config changed")
        source_config = yaml.safe_load(source_path.read_text())
        sources = load_sources(source_config)  # Integrity reads, not new held-out inference.
        hashes.update(sources["file_hashes"])
        manifest["source_commits"] = {
            name: sources[f"{name}_manifest"]["commit_sha"] for name in ("tree", "grid")
        }
        manifest["packages"] = {
            n: version(n)
            for n in (
                "lime",
                "numpy",
                "scipy",
                "scikit-learn",
                "xgboost",
                "joblib",
                "threadpoolctl",
            )
        }
        if manifest["packages"]["lime"] != config["lime_version"]:
            raise ValueError("unreviewed LIME version")
        manifest["lime_source_sha256"] = {}
        for name in ("lime.lime_tabular", "lime.lime_base", "lime.discretize"):
            path = Path(importlib.import_module(name).__file__)
            hashes[str(path.resolve())] = file_sha256(path)
            manifest["lime_source_sha256"][name] = file_sha256(path)
        bundle = sources["bundle"]
        if bundle.x_train.shape != (397, config["feature_count"]):
            raise ValueError("unexpected training shape")
        models = {
            family: joblib.load(sources["model_paths"][family]) for family in config["families"]
        }
        manifest["anchor_training_ids"] = bundle.train_indices[:8].tolist()
        manifest["threadpool_limit"] = 1
        manifest["same_seed_repeat_anchor_id"] = int(bundle.train_indices[0])
        _json(root / "manifest.json", manifest)
        with threadpool_limits(limits=1):
            for position in range(config["preflight_training_rows"]):
                sample_id = int(bundle.train_indices[position])
                for seed in config["explainer_seeds"]:
                    kwargs = dict(
                        training=bundle.x_train,
                        training_ids=bundle.train_indices,
                        heldout_ids=np.concatenate(
                            (bundle.validation_indices, bundle.test_indices)
                        ),
                        row=bundle.x_train[position],
                        names=bundle.feature_names,
                        sample_id=sample_id,
                        explainer_seed=seed,
                        fit_samples=config["fit_samples"],
                        evaluation_samples=config["evaluation_samples"],
                    )
                    design = build_lime_design(**kwargs)
                    case = root / f"sample-{sample_id}__explainer-seed-{seed}"
                    _save_design(case, design)
                    repeat_design = build_lime_design(**kwargs) if position == 0 else None
                    if repeat_design is not None:
                        for role in ("fit", "evaluation"):
                            for field in ("design", "inverse", "distances", "weights"):
                                if not np.array_equal(
                                    getattr(getattr(design, role), field),
                                    getattr(getattr(repeat_design, role), field),
                                ):
                                    raise ValueError("same-ID/seed neighborhood repeat failed")
                    for family, model in models.items():
                        fit_started = time.perf_counter()
                        result = fit_probability_lime(model, design)
                        repeat_passed = None
                        if repeat_design is not None:
                            repeat = fit_probability_lime(model, repeat_design)
                            repeat_passed = all(
                                np.allclose(
                                    result[k],
                                    repeat[k],
                                    atol=config["repeat_atol"],
                                    rtol=config["repeat_rtol"],
                                )
                                for k in (
                                    "coefficients",
                                    "probabilities",
                                    "fit_surrogate_predictions",
                                    "evaluation_surrogate_predictions",
                                )
                            )
                            if not repeat_passed:
                                raise ValueError("same-seed LIME surrogate repeat failed")
                        np.savez_compressed(
                            case / f"{family}.npz",
                            **{k: v for k, v in result.items() if k != "diagnostics"},
                        )
                        record = dict(
                            sample_id=sample_id,
                            family=family,
                            explainer_seed=seed,
                            partition="training",
                            same_seed_repeat_passed=repeat_passed,
                            runtime_seconds_including_repeat=time.perf_counter() - fit_started,
                            **result["diagnostics"],
                        )
                        records.append(record)
                        _json(case / f"{family}.json", record)
        checks.update(
            all_48_training_fits_completed=len(records) == 48,
            six_same_seed_repeats_passed=sum(r["same_seed_repeat_passed"] is True for r in records)
            == 6,
            only_registered_training_anchors=set(r["sample_id"] for r in records)
            == set(manifest["anchor_training_ids"]),
        )
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    try:
        checks["source_files_unchanged"] = all(file_sha256(Path(p)) == h for p, h in hashes.items())
    except OSError as error:
        checks["source_files_unchanged"] = False
        errors.append(str(error))
    checks["no_cnn_runtime"] = "torch" not in sys.modules
    summary = {
        "status": "ok" if all(checks.values()) and not errors else "quality_gate_failed",
        "output": str(root),
        "checks": checks,
        "errors": errors,
        "completed_training_fits": len(records),
        "test_explanations_generated": 0,
        "runtime_seconds": time.perf_counter() - started,
        "interpretation": "Technical preflight only; local fit quality is reported, not filtered.",
    }
    manifest.update(
        status=summary["status"],
        completed_at_utc=datetime.now(UTC).isoformat(),
        source_file_sha256=hashes,
    )
    _json(root / "training_diagnostics.json", records)
    _json(root / "manifest.json", manifest)
    _json(root / "summary.json", summary)
    return summary
