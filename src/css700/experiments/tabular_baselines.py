"""Append-only Random Forest and XGBoost baseline experiment."""

import csv
import hashlib
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from importlib import metadata
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import yaml

from css700.data import load_breast_cancer_splits
from css700.evaluation import spearman_agreement, top_k_overlap
from css700.training import evaluate_tabular_estimator, train_tabular_baseline

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "uncommitted"


def _git_worktree_dirty() -> bool:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return True


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in (
        "numpy",
        "PyYAML",
        "scikit-learn",
        "scipy",
        "joblib",
        "xgboost",
        "matplotlib",
        "css700",
    ):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = "missing"
    return versions


def split_fingerprint(
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
) -> str:
    """Hash ordered split indices using the versioned split-fingerprint contract."""

    digest = hashlib.sha256()
    digest.update(b"css700-split-v1\0")
    for name, values in (("train", train), ("validation", validation), ("test", test)):
        ordered = np.ascontiguousarray(values, dtype="<i8")
        digest.update(name.encode() + b"\0")
        digest.update(len(ordered).to_bytes(8, "little"))
        digest.update(ordered.tobytes())
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _predictive_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for model_name in sorted({str(row["model_name"]) for row in rows}):
        selected = [row for row in rows if row["model_name"] == model_name]
        record: dict[str, Any] = {"model_name": model_name, "seed_count": len(selected)}
        for metric in ("test_accuracy", "test_macro_f1", "test_roc_auc", "test_log_loss"):
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_sample_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            record[f"{metric}_minimum"] = float(values.min())
            record[f"{metric}_maximum"] = float(values.max())
        summary.append(record)
    return summary


def _prediction_agreement_rows(
    predictions: dict[tuple[str, int], np.ndarray],
    model_names: list[str],
    model_seeds: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name in model_names:
        for left_seed, right_seed in combinations(model_seeds, 2):
            left = predictions[(model_name, left_seed)]
            right = predictions[(model_name, right_seed)]
            rows.append(
                {
                    "model_name": model_name,
                    "left_model_seed": left_seed,
                    "right_model_seed": right_seed,
                    "sample_count": len(left),
                    "agreement_fraction": float(np.mean(left == right)),
                }
            )
    return rows


def _importance_stability_rows(
    importances: dict[tuple[str, int], np.ndarray],
    model_names: list[str],
    model_seeds: list[int],
    top_k_values: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name in model_names:
        for left_seed, right_seed in combinations(model_seeds, 2):
            left = importances[(model_name, left_seed)]
            right = importances[(model_name, right_seed)]
            row: dict[str, Any] = {
                "model_name": model_name,
                "left_model_seed": left_seed,
                "right_model_seed": right_seed,
                "spearman": spearman_agreement(left, right),
            }
            for top_k in top_k_values:
                overlap = top_k_overlap(left, right, k=top_k)
                row[f"top_{top_k}_overlap_fraction"] = overlap["overlap_fraction"]
                row[f"top_{top_k}_jaccard"] = overlap["jaccard"]
            rows.append(row)
    return rows


def _write_figure(
    path: Path,
    predictive_rows: list[dict[str, Any]],
    importance_stability: list[dict[str, Any]],
    model_names: list[str],
    model_seeds: list[int],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    colors = {"random_forest": "tab:blue", "xgboost": "tab:orange"}
    labels = {"random_forest": "Random Forest", "xgboost": "XGBoost"}
    seed_positions = np.arange(len(model_seeds))
    for model_name in model_names:
        rows = sorted(
            (row for row in predictive_rows if row["model_name"] == model_name),
            key=lambda row: int(row["model_seed"]),
        )
        axes[0].plot(
            seed_positions,
            [float(row["test_accuracy"]) for row in rows],
            marker="o",
            color=colors[model_name],
            label=f"{labels[model_name]} accuracy",
        )
        axes[0].plot(
            seed_positions,
            [float(row["test_macro_f1"]) for row in rows],
            marker="s",
            linestyle="--",
            color=colors[model_name],
            label=f"{labels[model_name]} macro F1",
        )
    axes[0].set(
        title="Tabular predictive stability",
        xlabel="Model seed",
        ylabel="Test metric",
        xticks=seed_positions,
        ylim=(0.8, 1.0),
    )
    axes[0].set_xticklabels([str(seed) for seed in model_seeds])
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    pair_labels = [f"{left}-{right}" for left, right in combinations(model_seeds, 2)]
    for model_name in model_names:
        rows = [row for row in importance_stability if row["model_name"] == model_name]
        axes[1].plot(
            pair_labels,
            [float(row["spearman"]) for row in rows],
            marker="o",
            color=colors[model_name],
            label=labels[model_name],
        )
    axes[1].set(
        title="Model-native global-importance stability",
        xlabel="Model-seed pair",
        ylabel="Spearman",
        ylim=(-1.0, 1.0),
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_tabular_baselines_experiment(config_path: Path, output_dir: Path) -> dict[str, Any]:
    """Train validation-selected RF/XGBoost baselines and write one immutable bundle."""

    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    started_at = datetime.now(UTC)
    started_counter = time.perf_counter()
    worktree_dirty_at_start = _git_worktree_dirty()
    experiment = config["experiment"]
    data_split_seed = int(experiment["data_split_seed"])
    model_seeds = [int(seed) for seed in experiment["model_seeds"]]
    if len(model_seeds) != len(set(model_seeds)):
        raise ValueError("model seeds must be unique")
    dataset_config = config["dataset"]
    bundle = load_breast_cancer_splits(
        seed=data_split_seed,
        test_size=float(dataset_config["test_size"]),
        validation_size=float(dataset_config["validation_size"]),
        stratify=bool(dataset_config["stratify"]),
    )
    fingerprint = split_fingerprint(
        bundle.train_indices,
        bundle.validation_indices,
        bundle.test_indices,
    )
    expected_fingerprint = str(experiment["expected_split_sha256"])
    model_names = list(config["models"])
    if model_names != ["random_forest", "xgboost"]:
        raise ValueError("the registered baseline models must be random_forest then xgboost")
    top_k_values = [int(value) for value in config["global_importance"]["top_k"]]

    seed_label = "-".join(str(seed) for seed in model_seeds)
    run_id = (
        f"{started_at.strftime('%Y%m%d-%H%M%S')}__breast-cancer__tabular__rf-xgboost"
        f"__split-seed-{data_split_seed}__model-seeds-{seed_label}__baseline"
    )
    run_dir = output_dir / run_id
    directories = {
        name: run_dir / name for name in ("figures", "metrics", "models", "preprocessing", "splits")
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=False)

    predictive_rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, int], np.ndarray] = {}
    importances: dict[tuple[str, int], np.ndarray] = {}
    reload_differences: dict[str, float] = {}

    for model_name in model_names:
        model_config = config["models"][model_name]
        for model_seed in model_seeds:
            fit_started = time.perf_counter()
            outcome = train_tabular_baseline(
                model_name,
                bundle.x_train,
                bundle.y_train,
                bundle.x_validation,
                bundle.y_validation,
                bundle.x_test,
                bundle.y_test,
                seed=model_seed,
                fixed_parameters=dict(model_config["fixed_parameters"]),
                search_grid=dict(model_config["search_grid"]),
            )
            fit_seconds = time.perf_counter() - fit_started
            model_path = directories["models"] / f"{model_name}__seed-{model_seed}.joblib"
            joblib.dump(outcome.model, model_path)
            reloaded_model = joblib.load(model_path)
            reloaded = evaluate_tabular_estimator(
                reloaded_model,
                bundle.x_test,
                bundle.y_test,
                num_classes=len(bundle.target_names),
            )
            reload_difference = float(
                np.max(np.abs(reloaded.probabilities - outcome.test.probabilities))
            )
            reload_differences[f"{model_name}__seed-{model_seed}"] = reload_difference
            predictions[(model_name, model_seed)] = outcome.test.predictions

            model_importances = np.asarray(outcome.model.feature_importances_, dtype=np.float64)
            importances[(model_name, model_seed)] = model_importances
            importance_order = np.argsort(-model_importances, kind="stable")
            ranks = np.empty_like(importance_order)
            ranks[importance_order] = np.arange(1, len(importance_order) + 1)
            for feature_index, (feature_name, importance) in enumerate(
                zip(bundle.feature_names, model_importances, strict=True)
            ):
                importance_rows.append(
                    {
                        "model_name": model_name,
                        "model_seed": model_seed,
                        "scope": "global_model_native",
                        "importance_definition": model_config["importance_definition"],
                        "feature_index": feature_index,
                        "feature_name": feature_name,
                        "importance": float(importance),
                        "rank": int(ranks[feature_index]),
                    }
                )

            for row in outcome.search_rows:
                search_rows.append(
                    {
                        "model_name": model_name,
                        "model_seed": model_seed,
                        "candidate_index": row["candidate_index"],
                        "parameters_json": json.dumps(row["parameters"], sort_keys=True),
                        "validation_accuracy": row["validation_accuracy"],
                        "validation_macro_f1": row["validation_macro_f1"],
                        "validation_log_loss": row["validation_log_loss"],
                        "validation_roc_auc": row["validation_roc_auc"],
                        "selected": row["selected"],
                    }
                )
            predictive_rows.append(
                {
                    "model_name": model_name,
                    "model_seed": model_seed,
                    "selected_candidate_index": outcome.selected_candidate_index,
                    "selected_parameters_json": json.dumps(
                        outcome.selected_parameters, sort_keys=True
                    ),
                    "validation_accuracy": outcome.validation.metrics["accuracy"],
                    "validation_macro_f1": outcome.validation.metrics["macro_f1"],
                    "test_accuracy": outcome.test.metrics["accuracy"],
                    "test_macro_f1": outcome.test.metrics["macro_f1"],
                    "test_macro_precision": outcome.test.metrics["macro_precision"],
                    "test_macro_recall": outcome.test.metrics["macro_recall"],
                    "test_roc_auc": outcome.test.metrics["roc_auc"],
                    "test_log_loss": outcome.test.metrics["log_loss"],
                    "test_confusion_matrix_json": json.dumps(
                        outcome.test.metrics["confusion_matrix"]
                    ),
                    "fit_and_selection_seconds": fit_seconds,
                    "majority_accuracy": outcome.majority_metrics["accuracy"],
                    "majority_macro_f1": outcome.majority_metrics["macro_f1"],
                }
            )
            for position, sample_id in enumerate(bundle.test_indices):
                prediction_rows.append(
                    {
                        "model_name": model_name,
                        "model_seed": model_seed,
                        "sample_id": int(sample_id),
                        "true_label": int(bundle.y_test[position]),
                        "predicted_label": int(outcome.test.predictions[position]),
                        "correct": int(
                            bundle.y_test[position] == outcome.test.predictions[position]
                        ),
                        "probability_class_0": float(outcome.test.probabilities[position, 0]),
                        "probability_class_1": float(outcome.test.probabilities[position, 1]),
                    }
                )

    predictive_summary = _predictive_summary(predictive_rows)
    agreement_rows = _prediction_agreement_rows(predictions, model_names, model_seeds)
    importance_stability = _importance_stability_rows(
        importances,
        model_names,
        model_seeds,
        top_k_values,
    )
    split_sets = [
        set(bundle.train_indices.tolist()),
        set(bundle.validation_indices.tolist()),
        set(bundle.test_indices.tolist()),
    ]
    probability_rows = np.asarray(
        [[row["probability_class_0"], row["probability_class_1"]] for row in prediction_rows],
        dtype=np.float64,
    )
    importance_matrix = np.stack(list(importances.values()))
    technical_checks = {
        "working_tree_clean_at_start": not worktree_dirty_at_start,
        "registered_models_and_seeds_present_once": len(predictive_rows)
        == len(model_names) * len(model_seeds),
        "split_fingerprint_matches_grid_reference": fingerprint == expected_fingerprint,
        "splits_are_disjoint": all(
            split_sets[left].isdisjoint(split_sets[right])
            for left, right in combinations(range(3), 2)
        ),
        "splits_cover_all_samples_once": len(set.union(*split_sets)) == 569
        and sum(len(values) for values in split_sets) == 569,
        "preprocessing_fit_on_training_only": bool(
            np.max(np.abs(bundle.x_train.mean(axis=0))) <= 1e-12
        ),
        "one_validation_selected_candidate_per_model_seed": all(
            sum(
                int(row["selected"])
                for row in search_rows
                if row["model_name"] == model_name and row["model_seed"] == model_seed
            )
            == 1
            for model_name in model_names
            for model_seed in model_seeds
        ),
        "all_predictions_and_probabilities_finite": bool(np.isfinite(probability_rows).all()),
        "all_probability_rows_sum_to_one": bool(
            np.allclose(probability_rows.sum(axis=1), 1.0, rtol=0.0, atol=1e-7)
        ),
        "serialized_models_reload_exactly": all(
            difference <= 1e-12 for difference in reload_differences.values()
        ),
        "global_importances_finite_nonnegative": bool(
            np.isfinite(importance_matrix).all() and np.all(importance_matrix >= 0.0)
        ),
        "global_importances_normalized": bool(
            np.allclose(importance_matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-6)
        ),
        "importance_stability_values_finite": all(
            np.isfinite(float(row["spearman"]))
            and all(
                np.isfinite(float(row[f"top_{top_k}_overlap_fraction"])) for top_k in top_k_values
            )
            for row in importance_stability
        ),
    }
    scientific_checks = {
        "all_models_beat_majority_accuracy": all(
            float(row["test_accuracy"]) > float(row["majority_accuracy"]) for row in predictive_rows
        ),
        "all_models_beat_majority_macro_f1": all(
            float(row["test_macro_f1"]) > float(row["majority_macro_f1"]) for row in predictive_rows
        ),
    }

    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    np.savez_compressed(
        directories["splits"] / "indices.npz",
        train=bundle.train_indices,
        validation=bundle.validation_indices,
        test=bundle.test_indices,
    )
    preprocessing = {
        "fit_split": "training",
        "method": "StandardScaler",
        "tree_scaling_note": (
            "Tree splits do not require standardization; the training-only transform is retained "
            "to match the frozen Grid pipeline and feature order."
        ),
        "feature_names": list(bundle.feature_names),
        "mean": bundle.scaler_mean.tolist(),
        "scale": bundle.scaler_scale.tolist(),
    }
    (directories["preprocessing"] / "standard_scaler.json").write_text(
        json.dumps(preprocessing, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(
        directories["metrics"] / "validation_search.csv",
        search_rows,
        list(search_rows[0]),
    )
    _write_csv(
        directories["metrics"] / "predictive_metrics.csv",
        predictive_rows,
        list(predictive_rows[0]),
    )
    _write_csv(
        directories["metrics"] / "predictive_summary.csv",
        predictive_summary,
        list(predictive_summary[0]),
    )
    _write_csv(
        directories["metrics"] / "test_predictions.csv",
        prediction_rows,
        list(prediction_rows[0]),
    )
    _write_csv(
        directories["metrics"] / "global_feature_importance.csv",
        importance_rows,
        list(importance_rows[0]),
    )
    _write_csv(
        directories["metrics"] / "prediction_seed_agreement.csv",
        agreement_rows,
        list(agreement_rows[0]),
    )
    _write_csv(
        directories["metrics"] / "global_importance_seed_stability.csv",
        importance_stability,
        list(importance_stability[0]),
    )
    _write_figure(
        directories["figures"] / "tabular_baseline_stability.png",
        predictive_rows,
        importance_stability,
        model_names,
        model_seeds,
    )

    completed_at = datetime.now(UTC)
    summary = {
        "experiment": "fixed-split tabular predictive baselines",
        "data_split_seed": data_split_seed,
        "model_seeds": model_seeds,
        "split_sha256": fingerprint,
        "predictive_summary": {
            str(row["model_name"]): {
                key: value for key, value in row.items() if key != "model_name"
            }
            for row in predictive_summary
        },
        "prediction_seed_agreement": agreement_rows,
        "global_importance_seed_stability": importance_stability,
        "technical_checks": technical_checks,
        "scientific_checks": scientific_checks,
        "interpretation_guardrail": (
            "Random Forest impurity importance and XGBoost gain importance are global, "
            "model-native diagnostics. They are not local class-targeted explanations and must "
            "not be compared as though their magnitudes share a scale."
        ),
    }
    (directories["metrics"] / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "run_id": run_id,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "duration_seconds": time.perf_counter() - started_counter,
        "commit_sha": _git_sha(),
        "git_worktree_dirty_at_start": worktree_dirty_at_start,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "data_split_seed": data_split_seed,
        "model_seeds": model_seeds,
        "split_sha256": fingerprint,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
        "serialized_model_reload_max_abs_probability_difference": reload_differences,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (run_dir / "notes.md").write_text(
        "# Run notes\n\n"
        "Random Forest and XGBoost were selected by validation macro F1 using training-only "
        "fits. The selected fitted candidate was evaluated on the test split without refitting "
        "on validation data. Model-native global importance is retained as a diagnostic only; "
        "local TreeSHAP and LIME comparisons are outside this run.\n",
        encoding="utf-8",
    )
    quality_passed = all(technical_checks.values())
    return {
        "status": "ok" if quality_passed else "quality_gate_failed",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "predictive_summary": summary["predictive_summary"],
        "technical_checks": technical_checks,
        "scientific_checks": scientific_checks,
    }
