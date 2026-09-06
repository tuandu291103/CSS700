"""Fixed-split stability analysis across independently trained Grid CNN seeds."""

import csv
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import yaml

from css700.config import resolve_experiment_seeds
from css700.evaluation import spearman_agreement, top_k_overlap

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


def _load_prediction_rows(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "sample_ids": np.asarray([int(row["sample_id"]) for row in rows], dtype=np.int64),
        "labels": np.asarray([int(row["true_label"]) for row in rows], dtype=np.int64),
        "predictions": np.asarray([int(row["predicted_label"]) for row in rows], dtype=np.int64),
        "correct": np.asarray([bool(int(row["correct"])) for row in rows], dtype=np.bool_),
    }


def _feature_matrix(
    path: Path,
    *,
    sample_ids: np.ndarray,
    feature_count: int,
    value_field: str,
    row_filter: Any | None = None,
    absolute: bool = False,
) -> np.ndarray:
    positions = {int(sample_id): position for position, sample_id in enumerate(sample_ids)}
    matrix = np.full((len(sample_ids), feature_count), np.nan, dtype=np.float64)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row_filter is not None and not row_filter(row):
                continue
            sample_position = positions.get(int(row["sample_id"]))
            feature_position = int(row["feature_index"])
            if sample_position is None or not 0 <= feature_position < feature_count:
                raise RuntimeError(f"invalid feature row in {path}")
            value = float(row[value_field])
            matrix[sample_position, feature_position] = abs(value) if absolute else value
    if not np.isfinite(matrix).all():
        raise RuntimeError(f"incomplete or non-finite feature matrix in {path}")
    return matrix


def _source_quality_passed(run_dir: Path) -> bool:
    predictive = json.loads((run_dir / "metrics" / "predictive_metrics.json").read_text())
    predictive_gate = predictive["quality_gate"]
    predictive_ok = all(
        bool(predictive_gate[name])
        for name in (
            "beats_majority_accuracy",
            "beats_majority_macro_f1",
            "checkpoint_reload_exact_within_1e-7",
            "working_tree_clean_at_start",
        )
    )
    explanation_files = (
        "grad_cam_summary.json",
        "integrated_gradients_summary.json",
        "occlusion_summary.json",
        "faithfulness_summary.json",
    )
    return predictive_ok and all(
        all(
            bool(value)
            for value in json.loads((run_dir / "metrics" / name).read_text())["checks"].values()
        )
        for name in explanation_files
    )


def _load_run(run_dir: Path, feature_count: int) -> dict[str, Any]:
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    seeds = resolve_experiment_seeds(config)
    predictions = _load_prediction_rows(run_dir / "model" / "test_predictions.csv")
    predictive = json.loads((run_dir / "metrics" / "predictive_metrics.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    scores = {
        "grad_cam": _feature_matrix(
            run_dir / "attributions" / "grad_cam_features.csv",
            sample_ids=predictions["sample_ids"],
            feature_count=feature_count,
            value_field="raw_attribution",
        ),
        "integrated_gradients_zero": _feature_matrix(
            run_dir / "attributions" / "integrated_gradients_features.csv",
            sample_ids=predictions["sample_ids"],
            feature_count=feature_count,
            value_field="signed_attribution",
            row_filter=lambda row: row["baseline"] == "zero",
            absolute=True,
        ),
        "occlusion_zero": _feature_matrix(
            run_dir / "attributions" / "occlusion_features.csv",
            sample_ids=predictions["sample_ids"],
            feature_count=feature_count,
            value_field="signed_target_logit_drop",
            absolute=True,
        ),
    }
    return {
        "run_dir": run_dir,
        "run_id": manifest["run_id"],
        "commit_sha": manifest["commit_sha"],
        "config": config,
        "data_split_seed": seeds.data_split_seed,
        "model_seed": seeds.model_seed,
        "predictions": predictions,
        "predictive": predictive,
        "faithfulness": json.loads((run_dir / "metrics" / "faithfulness_summary.json").read_text()),
        "scores": scores,
        "indices": {
            name: values.copy()
            for name, values in np.load(run_dir / "splits" / "indices.npz").items()
        },
        "feature_map": (run_dir / "encoding" / "feature_map.csv").read_text(),
        "preprocessing": json.loads(
            (run_dir / "preprocessing" / "standard_scaler.json").read_text()
        ),
        "source_quality_passed": _source_quality_passed(run_dir),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _prediction_pair_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right in combinations(runs, 2):
        left_predictions = left["predictions"]["predictions"]
        right_predictions = right["predictions"]["predictions"]
        labels = left["predictions"]["labels"]
        for position, sample_id in enumerate(left["predictions"]["sample_ids"]):
            rows.append(
                {
                    "left_model_seed": left["model_seed"],
                    "right_model_seed": right["model_seed"],
                    "sample_id": int(sample_id),
                    "true_label": int(labels[position]),
                    "left_prediction": int(left_predictions[position]),
                    "right_prediction": int(right_predictions[position]),
                    "same_predicted_target": int(
                        left_predictions[position] == right_predictions[position]
                    ),
                    "left_correct": int(left_predictions[position] == labels[position]),
                    "right_correct": int(right_predictions[position] == labels[position]),
                }
            )
    return rows


def _explanation_rows(
    runs: list[dict[str, Any]], methods: list[str], top_k_values: list[int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right in combinations(runs, 2):
        left_predictions = left["predictions"]["predictions"]
        right_predictions = right["predictions"]["predictions"]
        for method in methods:
            left_scores = left["scores"][method]
            right_scores = right["scores"][method]
            for position, sample_id in enumerate(left["predictions"]["sample_ids"]):
                spearman = spearman_agreement(left_scores[position], right_scores[position])
                row: dict[str, Any] = {
                    "left_model_seed": left["model_seed"],
                    "right_model_seed": right["model_seed"],
                    "method": method,
                    "sample_id": int(sample_id),
                    "left_prediction": int(left_predictions[position]),
                    "right_prediction": int(right_predictions[position]),
                    "same_predicted_target": int(
                        left_predictions[position] == right_predictions[position]
                    ),
                    "spearman_defined": int(np.isfinite(spearman)),
                    "spearman": float(spearman) if np.isfinite(spearman) else "",
                }
                for k in top_k_values:
                    overlap = top_k_overlap(left_scores[position], right_scores[position], k=k)
                    row[f"top_{k}_intersection"] = overlap["intersection"]
                    row[f"top_{k}_overlap_fraction"] = overlap["overlap_fraction"]
                    row[f"top_{k}_jaccard"] = overlap["jaccard"]
                rows.append(row)
    return rows


def _explanation_summaries(
    rows: list[dict[str, Any]], methods: list[str], top_k_values: list[int]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float | int | None]]]:
    summary_rows: list[dict[str, Any]] = []
    nested: dict[str, dict[str, float | int | None]] = {}
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        for group_name, group_rows in (
            ("all_samples", method_rows),
            (
                "same_predicted_target",
                [row for row in method_rows if row["same_predicted_target"]],
            ),
            (
                "different_predicted_target",
                [row for row in method_rows if not row["same_predicted_target"]],
            ),
        ):
            defined = [float(row["spearman"]) for row in group_rows if row["spearman_defined"]]
            summary = {
                "method": method,
                "group": group_name,
                "comparison_count": len(group_rows),
                "defined_spearman_count": len(defined),
                "median_spearman": float(np.median(defined)) if defined else None,
                **{
                    f"median_top_{k}_overlap_fraction": (
                        float(np.median([row[f"top_{k}_overlap_fraction"] for row in group_rows]))
                        if group_rows
                        else None
                    )
                    for k in top_k_values
                },
            }
            summary_rows.append(summary)
            nested[f"{method}__{group_name}"] = {
                key: value for key, value in summary.items() if key not in {"method", "group"}
            }
    return summary_rows, nested


def _faithfulness_rows(
    runs: list[dict[str, Any]], methods: list[str], replacement: str, contrast: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        faithfulness = run["faithfulness"]
        random_aopc = faithfulness["aopc"][replacement]["random"]["random"][
            "mean_target_logit_drop_aopc"
        ]
        for method in methods:
            method_aopc = faithfulness["aopc"][replacement][method]
            contrast_values = faithfulness["contrasts"][replacement][method][contrast][
                "target_logit_drop_aopc"
            ]
            rows.append(
                {
                    "model_seed": run["model_seed"],
                    "method": method,
                    "replacement": replacement,
                    "top_aopc": method_aopc["top"]["mean_target_logit_drop_aopc"],
                    "random_aopc": random_aopc,
                    "bottom_aopc": method_aopc["bottom"]["mean_target_logit_drop_aopc"],
                    "top_minus_random": contrast_values["mean_difference"],
                    "top_minus_random_ci_low": contrast_values["ci_low"],
                    "top_minus_random_ci_high": contrast_values["ci_high"],
                }
            )
    return rows


def _faithfulness_summary(
    rows: list[dict[str, Any]], methods: list[str]
) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        top_values = np.asarray([row["top_aopc"] for row in method_rows], dtype=np.float64)
        contrast_values = np.asarray(
            [row["top_minus_random"] for row in method_rows], dtype=np.float64
        )
        summary[method] = {
            "seed_count": len(method_rows),
            "mean_top_aopc": float(top_values.mean()),
            "sample_std_top_aopc": float(top_values.std(ddof=1)),
            "minimum_top_aopc": float(top_values.min()),
            "maximum_top_aopc": float(top_values.max()),
            "mean_top_minus_random": float(contrast_values.mean()),
            "sample_std_top_minus_random": float(contrast_values.std(ddof=1)),
            "seeds_with_positive_interval": sum(
                float(row["top_minus_random_ci_low"]) > 0 for row in method_rows
            ),
        }
    return summary


def _write_figure(
    path: Path,
    *,
    predictive_rows: list[dict[str, Any]],
    explanation_rows: list[dict[str, Any]],
    faithfulness_rows: list[dict[str, Any]],
    methods: list[str],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    seeds = [int(row["model_seed"]) for row in predictive_rows]
    positions = np.arange(len(seeds))
    axes[0].plot(
        positions,
        [row["test_accuracy"] for row in predictive_rows],
        marker="o",
        label="accuracy",
    )
    axes[0].plot(
        positions,
        [row["test_macro_f1"] for row in predictive_rows],
        marker="o",
        label="macro F1",
    )
    axes[0].set_xticks(positions, seeds)
    axes[0].set_ylim(0, 1)
    axes[0].set(title="Predictive stability", xlabel="Model seed", ylabel="Test metric")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    stability_values = [
        [
            float(row["spearman"])
            for row in explanation_rows
            if row["method"] == method and row["same_predicted_target"] and row["spearman_defined"]
        ]
        for method in methods
    ]
    axes[1].boxplot(
        stability_values,
        tick_labels=[method.replace("_", "\n") for method in methods],
    )
    axes[1].set(
        title="Same-target explanation stability",
        ylabel="Pairwise per-sample Spearman",
        ylim=(-1, 1),
    )
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].grid(axis="y", alpha=0.2)

    for method in methods:
        method_rows = sorted(
            [row for row in faithfulness_rows if row["method"] == method],
            key=lambda row: int(row["model_seed"]),
        )
        axes[2].plot(
            [str(row["model_seed"]) for row in method_rows],
            [row["top_minus_random"] for row in method_rows],
            marker="o",
            label=method.replace("_", " "),
        )
    axes[2].axhline(0, color="black", linewidth=0.7)
    axes[2].set(
        title="Deletion faithfulness stability",
        xlabel="Model seed",
        ylabel="Top-minus-random logit AOPC",
    )
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_grid_stability_experiment(
    run_dirs: list[Path], config_path: Path, output_dir: Path
) -> dict[str, Any]:
    """Compare predictive, explanation, and faithfulness stability across fixed-split runs."""

    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    worktree_dirty_at_start = _git_worktree_dirty()
    expected_seeds = [int(value) for value in config["model_seeds"]]
    methods = [str(value) for value in config["methods"]]
    top_k_values = [int(value) for value in config["top_k"]]
    feature_count = 30
    loaded = [_load_run(run_dir, feature_count) for run_dir in run_dirs]
    by_seed = {int(run["model_seed"]): run for run in loaded}
    if len(by_seed) != len(loaded) or set(by_seed) != set(expected_seeds):
        raise ValueError("run directories must contain each registered model seed exactly once")
    runs = [by_seed[seed] for seed in expected_seeds]

    started_at = datetime.now(UTC)
    seed_label = "-".join(str(seed) for seed in expected_seeds)
    run_id = (
        f"{started_at.strftime('%Y%m%d-%H%M%S')}__breast-cancer__grid__fixed-split"
        f"__model-seeds-{seed_label}__stability"
    )
    analysis_dir = output_dir / run_id
    metrics_dir = analysis_dir / "metrics"
    figures_dir = analysis_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=False)
    figures_dir.mkdir(parents=True, exist_ok=False)

    predictive_rows = [
        {
            "model_seed": run["model_seed"],
            "run_id": run["run_id"],
            "best_epoch": run["predictive"]["best_epoch"],
            "validation_macro_f1": run["predictive"]["validation"]["macro_f1"],
            "test_accuracy": run["predictive"]["test"]["accuracy"],
            "test_macro_f1": run["predictive"]["test"]["macro_f1"],
        }
        for run in runs
    ]
    prediction_rows = _prediction_pair_rows(runs)
    explanation_rows = _explanation_rows(runs, methods, top_k_values)
    explanation_summary_rows, explanation_summary = _explanation_summaries(
        explanation_rows, methods, top_k_values
    )
    faithfulness_rows = _faithfulness_rows(
        runs,
        methods,
        str(config["faithfulness_replacement"]),
        str(config["faithfulness_contrast"]),
    )
    faithfulness_summary = _faithfulness_summary(faithfulness_rows, methods)
    _write_csv(metrics_dir / "predictive_stability.csv", predictive_rows)
    _write_csv(metrics_dir / "prediction_pair_agreement.csv", prediction_rows)
    _write_csv(metrics_dir / "explanation_stability.csv", explanation_rows)
    _write_csv(metrics_dir / "explanation_stability_summary.csv", explanation_summary_rows)
    _write_csv(metrics_dir / "faithfulness_stability.csv", faithfulness_rows)

    reference = runs[0]
    fixed_indices = all(
        all(
            np.array_equal(run["indices"][name], reference["indices"][name])
            for name in reference["indices"]
        )
        for run in runs[1:]
    )
    fixed_samples = all(
        np.array_equal(run["predictions"]["sample_ids"], reference["predictions"]["sample_ids"])
        and np.array_equal(run["predictions"]["labels"], reference["predictions"]["labels"])
        for run in runs[1:]
    )
    controlled_sections = ("dataset", "preprocessing", "encoding", "model", "explanations")
    controlled_configs = all(
        all(
            run["config"].get(section) == reference["config"].get(section)
            for section in controlled_sections
        )
        for run in runs[1:]
    )
    predictions_matrix = np.stack([run["predictions"]["predictions"] for run in runs])
    consensus_mask = np.all(predictions_matrix == predictions_matrix[0], axis=0)
    pair_agreement: dict[str, float] = {}
    for left_seed, right_seed in combinations(expected_seeds, 2):
        pair_rows = [
            row
            for row in prediction_rows
            if row["left_model_seed"] == left_seed and row["right_model_seed"] == right_seed
        ]
        pair_agreement[f"{left_seed}_versus_{right_seed}"] = float(
            np.mean([row["same_predicted_target"] for row in pair_rows])
        )

    primary_rows = [row for row in explanation_rows if row["same_predicted_target"]]
    checks = {
        "working_tree_clean_at_start": not worktree_dirty_at_start,
        "registered_model_seeds_present_once": [run["model_seed"] for run in runs]
        == expected_seeds,
        "data_split_seed_fixed": all(
            run["data_split_seed"] == int(config["data_split_seed"]) for run in runs
        ),
        "split_indices_identical": fixed_indices,
        "sample_ids_and_labels_identical": fixed_samples,
        "feature_maps_identical": all(
            run["feature_map"] == reference["feature_map"] for run in runs[1:]
        ),
        "training_only_preprocessing_identical": all(
            run["preprocessing"] == reference["preprocessing"] for run in runs[1:]
        ),
        "controlled_config_sections_identical": controlled_configs,
        "all_source_quality_gates_passed": all(run["source_quality_passed"] for run in runs),
        "all_feature_scores_finite": all(
            np.isfinite(scores).all() for run in runs for scores in run["scores"].values()
        ),
        "primary_same_target_stability_available": bool(primary_rows),
        "all_primary_spearman_values_defined": all(
            bool(row["spearman_defined"]) for row in primary_rows
        ),
        "faithfulness_rows_complete": len(faithfulness_rows) == len(expected_seeds) * len(methods),
    }
    test_accuracies = np.asarray(
        [row["test_accuracy"] for row in predictive_rows], dtype=np.float64
    )
    test_macro_f1 = np.asarray([row["test_macro_f1"] for row in predictive_rows], dtype=np.float64)
    summary = {
        "analysis": "fixed-split model-training-seed stability",
        "analysis_commit_sha": _git_sha(),
        "analysis_worktree_dirty_at_start": worktree_dirty_at_start,
        "analysis_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "data_split_seed": int(config["data_split_seed"]),
        "model_seeds": expected_seeds,
        "input_runs": [
            {
                "model_seed": run["model_seed"],
                "run_id": run["run_id"],
                "commit_sha": run["commit_sha"],
            }
            for run in runs
        ],
        "predictive_stability": {
            "test_accuracy_mean": float(test_accuracies.mean()),
            "test_accuracy_sample_std": float(test_accuracies.std(ddof=1)),
            "test_accuracy_minimum": float(test_accuracies.min()),
            "test_accuracy_maximum": float(test_accuracies.max()),
            "test_macro_f1_mean": float(test_macro_f1.mean()),
            "test_macro_f1_sample_std": float(test_macro_f1.std(ddof=1)),
            "test_macro_f1_minimum": float(test_macro_f1.min()),
            "test_macro_f1_maximum": float(test_macro_f1.max()),
            "pairwise_prediction_agreement": pair_agreement,
            "all_model_consensus_sample_count": int(consensus_mask.sum()),
            "all_model_consensus_fraction": float(consensus_mask.mean()),
        },
        "explanation_stability": explanation_summary,
        "faithfulness_stability": faithfulness_summary,
        "checks": checks,
        "interpretation_guardrail": (
            "Primary explanation stability uses model pairs with the same predicted target. "
            "Different-target rows are retained but exploratory. Three fixed-split model seeds "
            "support descriptive stability estimates, not population-level variance claims."
        ),
    }
    (analysis_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (metrics_dir / "stability_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    completed_at = datetime.now(UTC)
    manifest = {
        "run_id": run_id,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "duration_seconds": (completed_at - started_at).total_seconds(),
        "commit_sha": _git_sha(),
        "git_worktree_dirty_at_start": worktree_dirty_at_start,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "python": platform.python_version(),
        "input_run_ids": [run["run_id"] for run in runs],
    }
    (analysis_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (analysis_dir / "notes.md").write_text(
        "# Stability analysis notes\n\n"
        "This analysis holds the split, preprocessing, Grid layout, architecture, and training "
        "hyperparameters fixed while varying only the model-training seed. Predicted-class "
        "explanation stability is interpreted primarily for model pairs sharing the same target.\n",
        encoding="utf-8",
    )
    _write_figure(
        figures_dir / "grid_seed_stability.png",
        predictive_rows=predictive_rows,
        explanation_rows=explanation_rows,
        faithfulness_rows=faithfulness_rows,
        methods=methods,
    )
    return {
        "status": "ok" if all(checks.values()) else "quality_gate_failed",
        "run_id": run_id,
        "analysis_dir": str(analysis_dir),
        "predictive_stability": summary["predictive_stability"],
        "explanation_stability": explanation_summary,
        "faithfulness_stability": faithfulness_summary,
        "checks": checks,
    }
