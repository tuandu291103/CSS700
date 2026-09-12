"""Append-only seed-42 probability TreeSHAP pilot on existing frozen models."""

import csv
import hashlib
import json
import platform
import sys
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from threadpoolctl import threadpool_limits

from css700.evaluation.local_agreement import paired_rank_rows, summarize_rank_rows
from css700.experiments.som_mapping_audit import _git_state
from css700.experiments.tree_shap_sources import file_sha256, load_sources
from css700.interpretability.tree_routing import TreeRoutingMismatch
from css700.interpretability.tree_shap import (
    ProbabilityTreeSHAP,
    binary_probabilities,
    training_background,
)


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_result(path: Path, result: Any) -> None:
    np.savez_compressed(
        path,
        values=result.values,
        base=result.base,
        probabilities=result.probabilities,
        residuals=result.residuals,
    )


def _validate_config(config: dict) -> None:
    if config["protocol_version"] != "tree-shap-probability-v1":
        raise ValueError("unsupported TreeSHAP protocol")
    if config["background"]["sensitivity_size"] != 100:
        raise ValueError("v1 requires the registered 100-row sensitivity background")
    for key, expected in (
        ("model_output", "probability"),
        ("feature_perturbation", "interventional"),
        ("approximate", False),
    ):
        if config["shap"][key] != expected:
            raise ValueError(f"unexpected TreeSHAP {key}")
    if (
        config["background"]["primary"] != "all_training_rows"
        or config["comparison"]["ranking"] != "absolute"
        or config["comparison"]["primary_target"] != "same_predicted_class"
    ):
        raise ValueError("unsupported background/ranking/target contract")
    for section, key in (
        ("shap", "batch_size"),
        ("shap", "smoke_training_rows"),
        ("background", "sensitivity_size"),
        ("comparison", "bootstrap_resamples"),
        ("comparison", "minimum_interval_group"),
    ):
        value = config[section][key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{section}.{key} must be a positive integer")


def _figure(path: Path, summaries: list[dict], *, complete: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    rows = [row for row in summaries if row["group"] == "all" and row["eligible_samples"]]
    figure, axis = plt.subplots(figsize=(12, max(4, len(rows) * 0.5)))
    for i, row in enumerate(rows):
        score = row["spearman"]
        interval = score["mean_95_percentile_interval"]
        if interval:
            axis.plot(interval, [i, i], color="tab:blue")
        axis.plot(score["mean"], i, "o", color="tab:blue")
    axis.set_yticks(
        range(len(rows)), [f"{r['comparison']} (n={r['eligible_samples']})" for r in rows]
    )
    axis.set(
        xlim=(-1.05, 1.05),
        xlabel="Mean absolute-score Spearman; 95% sample-bootstrap interval",
        title=(
            (
                "Seed-42 same-target agreement\n"
                if complete
                else "INCOMPLETE PILOT: diagnostics only\n"
            )
            + "Descriptive cross-model/method comparisons; not faithfulness"
        ),
    )
    axis.axvline(0, color="gray", linewidth=0.7)
    axis.invert_yaxis()
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def run_tree_shap_experiment(config_path: Path, output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    sha, dirty = _git_state()
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    _validate_config(config)
    run_id = (
        datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
        + "__breast-cancer__tree-shap__seed-42__pilot"
    )
    root = output_dir / run_id
    root.mkdir(parents=True, exist_ok=False)
    (root / "config.yaml").write_bytes(config_bytes)
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "commit_sha": sha,
        "worktree_dirty_at_start": dirty,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "status": "running",
        "committee_approval_claimed": False,
    }
    _json(root / "manifest.json", manifest)
    condition_reports, accepted, comparisons, aggregates = {}, {}, [], []
    source_data = None
    checks = {"clean_worktree": not dirty, "no_cnn_runtime": "torch" not in sys.modules}
    errors = []
    try:
        if not all(checks.values()):
            raise RuntimeError("TreeSHAP requires a clean commit and a process without PyTorch")
        manifest["packages"] = {
            name: metadata.version(name)
            for name in (
                "shap",
                "xgboost",
                "scikit-learn",
                "numpy",
                "scipy",
                "joblib",
                "matplotlib",
                "threadpoolctl",
            )
        }
        source_data = load_sources(config)
        bundle = source_data["bundle"]
        manifest["source_file_sha256"] = source_data["file_hashes"]
        manifest["source_tree_commit"] = source_data["tree_manifest"]["commit_sha"]
        manifest["source_grid_commit"] = source_data["grid_manifest"]["commit_sha"]
        manifest["feature_names"] = bundle.feature_names
        manifest["feature_order_sha256"] = hashlib.sha256(
            json.dumps(bundle.feature_names, separators=(",", ":")).encode()
        ).hexdigest()
        manifest["class_names"] = bundle.target_names
        manifest["split_fingerprint"] = config["expected_split_sha256"]
        manifest["thread_limit"] = 1
        _json(root / "manifest.json", manifest)
        np.savez_compressed(
            root / "splits.npz",
            train=bundle.train_indices,
            validation=bundle.validation_indices,
            test=bundle.test_indices,
        )
        np.savez_compressed(
            root / "preprocessing.npz", mean=bundle.scaler_mean, scale=bundle.scaler_scale
        )
        models = {name: joblib.load(path) for name, path in source_data["model_paths"].items()}
        smoke_size = config["shap"]["smoke_training_rows"]
        if smoke_size > len(bundle.x_train):
            raise ValueError("smoke sample count exceeds training partition")
        settings = config["shap"]
        with threadpool_limits(limits=1):
            for background_name, size in (
                ("train_all", None),
                ("train_100", config["background"]["sensitivity_size"]),
            ):
                if background_name == "train_100" and len(accepted) != 2:
                    errors.append(
                        "Sensitivity skipped because both primary conditions did not pass"
                    )
                    break
                background, background_ids = training_background(
                    bundle.x_train,
                    bundle.train_indices,
                    np.concatenate((bundle.validation_indices, bundle.test_indices)),
                    size=size,
                    seed=config["background"]["sensitivity_seed"],
                )
                np.savez_compressed(
                    root / f"background_{background_name}.npz",
                    values=background,
                    sample_ids=background_ids,
                )
                for family, model in models.items():
                    condition = f"{family}__{background_name}"
                    condition_root = root / condition
                    condition_root.mkdir()
                    report: dict[str, Any] = {
                        "status": "running",
                        "background_rows": len(background),
                    }
                    condition_reports[condition] = report
                    try:
                        probabilities = binary_probabilities(model, bundle.x_test)
                        saved = source_data["tree_predictions"][family]
                        if not np.array_equal(
                            probabilities.argmax(axis=1), saved["predictions"]
                        ) or not np.allclose(
                            probabilities, saved["probabilities"], atol=1e-7, rtol=1e-6
                        ):
                            raise ValueError("reloaded predictions do not match frozen baseline")
                        report["reload_max_abs_probability_difference"] = float(
                            np.abs(probabilities - saved["probabilities"]).max()
                        )
                        init_start = time.perf_counter()
                        explainer = ProbabilityTreeSHAP(
                            model,
                            background,
                            bundle.feature_names,
                            family=family,
                            atol=settings["reconstruction_atol"],
                            rtol=settings["reconstruction_rtol"],
                        )
                        report["initialization_seconds"] = time.perf_counter() - init_start
                        report["adapter"] = explainer.adapter
                        if explainer.routing_audit is not None:
                            _json(condition_root / "routing_audit.json", explainer.routing_audit)
                        report["background_retained_exactly"] = True
                        warm = explainer.explain(bundle.x_train[:1])
                        smoke = explainer.explain(
                            bundle.x_train[:smoke_size], batch_size=settings["batch_size"]
                        )
                        repeat = explainer.explain(
                            bundle.x_train[:smoke_size], batch_size=settings["batch_size"]
                        )
                        _save_result(condition_root / "training_smoke.npz", smoke)
                        _save_result(condition_root / "training_smoke_repeat.npz", repeat)
                        report["warmup_seconds"] = warm.elapsed_seconds
                        report["smoke_and_repeat_seconds"] = (
                            smoke.elapsed_seconds + repeat.elapsed_seconds
                        )
                        report["smoke_training_ids"] = bundle.train_indices[:smoke_size].tolist()
                        report["smoke_checks"] = {
                            **smoke.checks,
                            "warmup_passed": all(warm.checks.values()),
                            "repeat_passed": all(repeat.checks.values()),
                            "repeatable": bool(
                                np.allclose(smoke.values, repeat.values, atol=1e-10, rtol=1e-8)
                            ),
                        }
                        if not all(report["smoke_checks"].values()):
                            raise RuntimeError(
                                "training-only smoke checks failed; no test explanations generated"
                            )
                        result = explainer.explain(bundle.x_test, batch_size=settings["batch_size"])
                        _save_result(condition_root / "test_attributions.npz", result)
                        report.update(
                            {
                                "test_checks": result.checks,
                                "test_seconds_including_validation": result.elapsed_seconds,
                                "seconds_per_test_sample": result.elapsed_seconds
                                / len(bundle.x_test),
                                "max_abs_reconstruction_residual": float(
                                    np.abs(result.residuals).max()
                                ),
                                "max_abs_base_residual": float(
                                    np.abs(result.base - explainer.background_mean).max()
                                ),
                                "output_rule": result.output_rule,
                            }
                        )
                        if not all(result.checks.values()):
                            raise RuntimeError(
                                "test reconstruction checks failed; retained raw diagnostic arrays"
                            )
                        targets = saved["predictions"]
                        selected = result.values[np.arange(len(targets)), :, targets]
                        rows = []
                        for i, sample_id in enumerate(bundle.test_indices):
                            ranks = (
                                np.argsort(
                                    np.argsort(-np.abs(selected[i]), kind="stable"), kind="stable"
                                )
                                + 1
                            )
                            for j, name in enumerate(bundle.feature_names):
                                rows.append(
                                    {
                                        "sample_id": int(sample_id),
                                        "true_label": int(bundle.y_test[i]),
                                        "target_class": int(targets[i]),
                                        "feature_index": j,
                                        "feature_name": name,
                                        "signed_attribution": float(selected[i, j]),
                                        "absolute_rank": int(ranks[j]),
                                        "base_value": float(result.base[targets[i]]),
                                        "probability": float(result.probabilities[i, targets[i]]),
                                    }
                                )
                        _csv(condition_root / "target_features.csv", rows)
                        for target in (0, 1):
                            eligible = targets == target
                            for j, name in enumerate(bundle.feature_names):
                                aggregates.append(
                                    {
                                        "condition": condition,
                                        "target_class": target,
                                        "sample_count": int(eligible.sum()),
                                        "feature_index": j,
                                        "feature_name": name,
                                        "mean_absolute_local_shap": float(
                                            np.abs(selected[eligible, j]).mean()
                                        )
                                        if eligible.any()
                                        else None,
                                    }
                                )
                        accepted[condition] = selected
                        report["status"] = "ok"
                    except Exception as error:
                        if isinstance(error, TreeRoutingMismatch):
                            _json(condition_root / "routing_audit.json", error.audit)
                            report["training_routing_check_passed"] = False
                        report.update(
                            status="failed", error_type=type(error).__name__, error=str(error)
                        )
                        errors.append(f"{condition}: {type(error).__name__}: {error}")
                    _json(condition_root / "checks.json", report)

            def compare(
                left_key: str,
                right_key: str,
                left: np.ndarray,
                right: np.ndarray,
                lp: np.ndarray,
                rp: np.ndarray,
            ) -> None:
                comparisons.extend(
                    paired_rank_rows(
                        left,
                        right,
                        lp,
                        rp,
                        bundle.y_test,
                        bundle.test_indices,
                        comparison=f"{left_key} vs {right_key}",
                    )
                )

            for background_name in ("train_all", "train_100"):
                rf, xgb = f"random_forest__{background_name}", f"xgboost__{background_name}"
                if rf in accepted and xgb in accepted:
                    compare(
                        rf,
                        xgb,
                        accepted[rf],
                        accepted[xgb],
                        source_data["tree_predictions"]["random_forest"]["predictions"],
                        source_data["tree_predictions"]["xgboost"]["predictions"],
                    )
            for family in models:
                primary, sensitivity = f"{family}__train_all", f"{family}__train_100"
                predictions = source_data["tree_predictions"][family]["predictions"]
                if primary in accepted and sensitivity in accepted:
                    compare(
                        primary,
                        sensitivity,
                        accepted[primary],
                        accepted[sensitivity],
                        predictions,
                        predictions,
                    )
                if primary in accepted:
                    for method, scores in source_data["grid_scores"].items():
                        compare(
                            primary,
                            f"grid_{method}",
                            accepted[primary],
                            scores,
                            predictions,
                            source_data["grid_predictions"]["predictions"],
                        )
        summaries = summarize_rank_rows(
            comparisons,
            resamples=config["comparison"]["bootstrap_resamples"],
            seed=config["comparison"]["bootstrap_seed"],
            min_group=config["comparison"]["minimum_interval_group"],
        )
        _csv(root / "paired_agreement.csv", comparisons)
        _csv(root / "class_aggregate_features.csv", aggregates)
        _json(root / "agreement_summary.json", summaries)
        if summaries:
            _figure(root / "agreement.png", summaries, complete=len(accepted) == 4 and not errors)
        checks["all_four_conditions_accepted"] = len(accepted) == 4
        checks["source_artifacts_unchanged"] = all(
            file_sha256(Path(path)) == value for path, value in source_data["file_hashes"].items()
        )
        checks["no_cnn_runtime"] = "torch" not in sys.modules
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    summary = {
        "status": "ok" if all(checks.values()) and not errors else "quality_gate_failed",
        "output": str(root),
        "checks": checks,
        "conditions": condition_reports,
        "errors": errors,
        "runtime_seconds": time.perf_counter() - started,
        "interpretation": (
            "Descriptive local agreement; different models and CNN-logit/tree-probability "
            "outputs. Not faithfulness or clinical relevance."
        ),
    }
    manifest.update(status=summary["status"], completed_at_utc=datetime.now(UTC).isoformat())
    _json(root / "manifest.json", manifest)
    _json(root / "summary.json", summary)
    return summary
