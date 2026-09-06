"""Cumulative deletion faithfulness evaluation for the frozen three-method Grid pilot."""

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib
import numpy as np
import torch
import yaml

from css700.config import resolve_experiment_seeds
from css700.data import load_breast_cancer_splits
from css700.encoding import GridEncoder
from css700.evaluation import normalized_area_over_curve
from css700.interpretability import (
    cumulative_feature_deletion,
    trace_attributions_to_features,
)
from css700.models import SmallTabularImageCNN

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


def _load_predictions(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "sample_ids": np.asarray([int(row["sample_id"]) for row in rows], dtype=np.int64),
        "labels": np.asarray([int(row["true_label"]) for row in rows], dtype=np.int64),
        "predictions": np.asarray([int(row["predicted_label"]) for row in rows], dtype=np.int64),
        "correct": np.asarray([bool(int(row["correct"])) for row in rows], dtype=np.bool_),
    }


def _build_feature_masks(encoder: GridEncoder) -> np.ndarray:
    if encoder.mapping is None or encoder.rows is None or encoder.columns is None:
        raise RuntimeError("the encoder must be fit before building feature masks")
    masks = np.zeros((len(encoder.mapping), 1, encoder.rows, encoder.columns), dtype=np.bool_)
    for pixel in encoder.mapping:
        masks[pixel.feature_index, pixel.channel, pixel.row, pixel.column] = True
    return masks


def _load_method_scores(run_dir: Path, encoder: GridEncoder) -> dict[str, np.ndarray]:
    grad_cam_maps = np.load(run_dir / "attributions" / "grad_cam_raw.npy")
    scores = {"grad_cam": trace_attributions_to_features(grad_cam_maps, encoder.mapping or ())}
    with np.load(run_dir / "attributions" / "integrated_gradients_raw.npz") as ig_raw:
        if "zero" not in ig_raw.files:
            raise RuntimeError("Integrated Gradients outputs must include the zero baseline")
        scores["integrated_gradients_zero"] = np.abs(
            trace_attributions_to_features(ig_raw["zero"], encoder.mapping or ())
        )
    with np.load(run_dir / "attributions" / "occlusion_raw.npz") as occlusion_raw:
        scores["occlusion_zero"] = np.abs(occlusion_raw["signed_target_logit_drop"])
    return scores


def _macro_f1(labels: np.ndarray, predictions: np.ndarray, classes: np.ndarray) -> float:
    values: list[float] = []
    for class_index in classes:
        true_positive = np.sum((labels == class_index) & (predictions == class_index))
        false_positive = np.sum((labels != class_index) & (predictions == class_index))
        false_negative = np.sum((labels == class_index) & (predictions != class_index))
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(float(2 * true_positive / denominator) if denominator else 0.0)
    return float(np.mean(values))


def _interval(values: np.ndarray, confidence_level: float) -> tuple[float, float]:
    alpha = (1.0 - confidence_level) / 2.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


def _bootstrap_curve_point(
    logit_drops: np.ndarray,
    probability_drops: np.ndarray,
    predictions: np.ndarray,
    *,
    labels: np.ndarray,
    original_predictions: np.ndarray,
    resamples: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    repeat_count, sample_count = predictions.shape
    classes = np.unique(labels)
    macro_f1_values = np.asarray(
        [_macro_f1(labels, row, classes) for row in predictions], dtype=np.float64
    )
    point = {
        "mean_target_logit_drop": float(logit_drops.mean()),
        "median_target_logit_drop": float(np.median(logit_drops)),
        "mean_target_probability_drop": float(probability_drops.mean()),
        "prediction_flip_rate": float((predictions != original_predictions[None, :]).mean()),
        "accuracy": float((predictions == labels[None, :]).mean()),
        "macro_f1": float(macro_f1_values.mean()),
    }
    bootstrap = {
        name: np.empty(resamples, dtype=np.float64) for name in point if "median" not in name
    }
    for bootstrap_position in range(resamples):
        sample_positions = rng.integers(0, sample_count, size=sample_count)
        repeat_positions = (
            rng.integers(0, repeat_count, size=sample_count)
            if repeat_count > 1
            else np.zeros(sample_count, dtype=np.int64)
        )
        selected_predictions = predictions[repeat_positions, sample_positions]
        selected_labels = labels[sample_positions]
        bootstrap["mean_target_logit_drop"][bootstrap_position] = logit_drops[
            repeat_positions, sample_positions
        ].mean()
        bootstrap["mean_target_probability_drop"][bootstrap_position] = probability_drops[
            repeat_positions, sample_positions
        ].mean()
        bootstrap["prediction_flip_rate"][bootstrap_position] = (
            selected_predictions != original_predictions[sample_positions]
        ).mean()
        bootstrap["accuracy"][bootstrap_position] = (selected_predictions == selected_labels).mean()
        bootstrap["macro_f1"][bootstrap_position] = _macro_f1(
            selected_labels, selected_predictions, classes
        )
    for metric, distribution in bootstrap.items():
        low, high = _interval(distribution, confidence_level)
        point[f"{metric}_ci_low"] = low
        point[f"{metric}_ci_high"] = high
    return point


def _bootstrap_mean_summary(
    values: np.ndarray,
    *,
    resamples: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    repeat_count, sample_count = values.shape
    distribution = np.empty(resamples, dtype=np.float64)
    for bootstrap_position in range(resamples):
        sample_positions = rng.integers(0, sample_count, size=sample_count)
        repeat_positions = (
            rng.integers(0, repeat_count, size=sample_count)
            if repeat_count > 1
            else np.zeros(sample_count, dtype=np.int64)
        )
        distribution[bootstrap_position] = values[repeat_positions, sample_positions].mean()
    low, high = _interval(distribution, confidence_level)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci_low": low,
        "ci_high": high,
    }


def _bootstrap_contrast(
    left: np.ndarray,
    right: np.ndarray,
    *,
    resamples: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    sample_count = left.shape[1]
    distribution = np.empty(resamples, dtype=np.float64)
    for bootstrap_position in range(resamples):
        sample_positions = rng.integers(0, sample_count, size=sample_count)
        left_repeats = (
            rng.integers(0, left.shape[0], size=sample_count)
            if left.shape[0] > 1
            else np.zeros(sample_count, dtype=np.int64)
        )
        right_repeats = (
            rng.integers(0, right.shape[0], size=sample_count)
            if right.shape[0] > 1
            else np.zeros(sample_count, dtype=np.int64)
        )
        distribution[bootstrap_position] = (
            left[left_repeats, sample_positions] - right[right_repeats, sample_positions]
        ).mean()
    low, high = _interval(distribution, confidence_level)
    return {
        "mean_difference": float(left.mean() - right.mean()),
        "ci_low": low,
        "ci_high": high,
    }


def _condition_key(method: str, order: str, replacement: str) -> str:
    return f"{method}__{order}__{replacement}"


def _all_numeric_row_values_finite(rows: list[dict[str, Any]]) -> bool:
    return all(
        np.isfinite(float(value))
        for row in rows
        for value in row.values()
        if isinstance(value, (int, float, np.integer, np.floating))
    )


def _result_arrays(result: Any) -> dict[str, np.ndarray | int]:
    return {
        "target_logit_drops": result.target_logit_drops.numpy(),
        "target_probability_drops": result.target_probability_drops.numpy(),
        "predictions": result.predictions.numpy().astype(np.int64, copy=False),
        "target_logits": result.target_logits.numpy(),
        "target_probabilities": result.target_probabilities.numpy(),
        "deleted_feature_counts": result.deleted_feature_counts.numpy(),
        "original_logits": result.original_logits.numpy(),
        "original_probabilities": result.original_probabilities.numpy(),
        "target_classes": result.target_classes.numpy().astype(np.int64, copy=False),
        "batch_forward_calls": result.batch_forward_calls,
        "conceptual_model_evaluations": result.conceptual_model_evaluations,
    }


def _write_curve_figure(
    path: Path,
    *,
    curve_rows: list[dict[str, Any]],
    methods: list[str],
    replacements: list[str],
) -> None:
    figure, axes = plt.subplots(
        len(replacements),
        len(methods),
        figsize=(5 * len(methods), 4 * len(replacements)),
        sharex=True,
    )
    if len(replacements) == 1:
        axes = np.expand_dims(axes, axis=0)
    colors = {"top": "#c0392b", "random": "#7f8c8d", "bottom": "#2980b9"}
    for row_index, replacement in enumerate(replacements):
        random_rows = [
            row
            for row in curve_rows
            if row["method"] == "random" and row["replacement"] == replacement
        ]
        for column_index, method in enumerate(methods):
            axis = axes[row_index, column_index]
            for order in ("top", "random", "bottom"):
                rows = (
                    random_rows
                    if order == "random"
                    else [
                        row
                        for row in curve_rows
                        if row["method"] == method
                        and row["order"] == order
                        and row["replacement"] == replacement
                    ]
                )
                steps = np.asarray([row["k"] for row in rows])
                values = np.asarray([row["mean_target_logit_drop"] for row in rows])
                low = np.asarray([row["mean_target_logit_drop_ci_low"] for row in rows])
                high = np.asarray([row["mean_target_logit_drop_ci_high"] for row in rows])
                axis.plot(steps, values, marker="o", markersize=3, label=order, color=colors[order])
                axis.fill_between(steps, low, high, color=colors[order], alpha=0.12)
            axis.axhline(0.0, color="black", linewidth=0.7)
            axis.set_title(f"{method.replace('_', ' ')}\n{replacement.replace('_', ' ')}")
            axis.set_xlabel("Deleted features (k)")
            axis.set_ylabel("Mean frozen-target logit drop")
            axis.grid(alpha=0.2)
    axes[0, 0].legend(title="Deletion order")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_aopc_figure(
    path: Path,
    *,
    aopc_rows: list[dict[str, Any]],
    methods: list[str],
    replacements: list[str],
) -> None:
    figure, axes = plt.subplots(
        1, len(replacements), figsize=(7 * len(replacements), 5), sharey=True
    )
    if len(replacements) == 1:
        axes = np.asarray([axes])
    offsets = {"top": -0.22, "random": 0.0, "bottom": 0.22}
    colors = {"top": "#c0392b", "random": "#7f8c8d", "bottom": "#2980b9"}
    for axis, replacement in zip(axes, replacements, strict=True):
        for method_position, method in enumerate(methods):
            for order in ("top", "random", "bottom"):
                row = next(
                    row
                    for row in aopc_rows
                    if row["replacement"] == replacement
                    and row["method"] == ("random" if order == "random" else method)
                    and row["order"] == order
                )
                x_position = method_position + offsets[order]
                value = float(row["mean_target_logit_drop_aopc"])
                low = float(row["mean_target_logit_drop_aopc_ci_low"])
                high = float(row["mean_target_logit_drop_aopc_ci_high"])
                axis.errorbar(
                    x_position,
                    value,
                    yerr=[[value - low], [high - value]],
                    fmt="o",
                    color=colors[order],
                    capsize=3,
                    label=order if method_position == 0 else None,
                )
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_xticks(np.arange(len(methods)), [method.replace("_", "\n") for method in methods])
        axis.set_title(f"Replacement: {replacement.replace('_', ' ')}")
        axis.set_ylabel("Normalized target-logit-drop AOPC")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(title="Deletion order")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_faithfulness_experiment(run_dir: Path, config_path: Path) -> dict[str, Any]:
    """Append cumulative deletion curves, controls, uncertainty, and contrasts to a frozen run."""

    output_paths = {
        "config": run_dir / "metrics" / "faithfulness_config.yaml",
        "rankings": run_dir / "metrics" / "faithfulness_rankings.npz",
        "replacements": run_dir / "metrics" / "faithfulness_replacements.npz",
        "raw": run_dir / "metrics" / "faithfulness_raw.npz",
        "samples": run_dir / "metrics" / "faithfulness_sample_curves.csv",
        "curves": run_dir / "metrics" / "faithfulness_curves.csv",
        "aopc": run_dir / "metrics" / "faithfulness_aopc.csv",
        "contrasts": run_dir / "metrics" / "faithfulness_contrasts.csv",
        "summary": run_dir / "metrics" / "faithfulness_summary.json",
        "curves_figure": run_dir / "figures" / "faithfulness_deletion_curves.png",
        "aopc_figure": run_dir / "figures" / "faithfulness_aopc.png",
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Faithfulness outputs already exist: {existing}")
    prerequisites = [
        run_dir / "attributions" / "grad_cam_raw.npy",
        run_dir / "attributions" / "integrated_gradients_raw.npz",
        run_dir / "attributions" / "occlusion_raw.npz",
    ]
    missing = [str(path) for path in prerequisites if not path.exists()]
    if missing:
        raise FileNotFoundError(f"the frozen run is missing prerequisite outputs: {missing}")

    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    methods = [str(value) for value in config["ranking_methods"]]
    if methods != ["grad_cam", "integrated_gradients_zero", "occlusion_zero"]:
        raise ValueError(
            "the primary Grid faithfulness comparison requires the frozen three methods"
        )
    replacements_requested = [str(value) for value in config["replacements"]]
    if replacements_requested != ["zero", "training_median"]:
        raise ValueError("the registered Grid replacements must be zero and training_median")

    worktree_dirty_at_start = _git_worktree_dirty()
    training_config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    seeds = resolve_experiment_seeds(training_config)
    dataset_config = training_config["dataset"]
    bundle = load_breast_cancer_splits(
        seed=seeds.data_split_seed,
        test_size=float(dataset_config["test_size"]),
        validation_size=float(dataset_config["validation_size"]),
        stratify=bool(dataset_config["stratify"]),
    )
    stored_indices = np.load(run_dir / "splits" / "indices.npz")
    if not np.array_equal(stored_indices["test"], bundle.test_indices):
        raise RuntimeError("reconstructed test indices do not match the frozen run")
    encoder = GridEncoder(padding_value=float(training_config["encoding"]["padding_value"])).fit(
        bundle.feature_names
    )
    test_images = encoder.transform(bundle.x_test)
    frozen = _load_predictions(run_dir / "model" / "test_predictions.csv")
    if not np.array_equal(frozen["sample_ids"], bundle.test_indices):
        raise RuntimeError("stored prediction rows do not match the frozen test order")

    checkpoint = torch.load(
        run_dir / "model" / "best_checkpoint.pt", map_location="cpu", weights_only=True
    )
    model = SmallTabularImageCNN(
        input_channels=int(checkpoint["input_channels"]),
        num_classes=int(checkpoint["num_classes"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    masks = _build_feature_masks(encoder)
    feature_count = len(bundle.feature_names)
    k_values = [int(value) for value in config["deletion_steps"]]
    if k_values[0] != 0 or k_values[-1] != feature_count:
        raise ValueError("deletion_steps must include zero and the full feature count")
    replacement_images = {
        "zero": np.zeros((1, *test_images.shape[1:]), dtype=np.float32),
        "training_median": encoder.transform(np.median(bundle.x_train, axis=0, keepdims=True)),
    }
    method_scores = _load_method_scores(run_dir, encoder)
    if list(method_scores) != methods:
        raise RuntimeError("loaded explanation methods do not match the registered order")
    ranking_arrays: dict[str, np.ndarray] = {}
    for method, scores in method_scores.items():
        ranking_arrays[f"{method}__top"] = np.argsort(-scores, axis=1, kind="stable")
        ranking_arrays[f"{method}__bottom"] = np.argsort(scores, axis=1, kind="stable")

    random_rng = np.random.default_rng(int(config["random_seed"]))
    random_rankings = np.argsort(
        random_rng.random((int(config["random_repeats"]), len(test_images), feature_count)),
        axis=2,
        kind="stable",
    )
    ranking_arrays["random"] = random_rankings

    warmup_rankings = torch.from_numpy(ranking_arrays[f"{methods[0]}__top"][:1])
    cumulative_feature_deletion(
        model,
        torch.from_numpy(test_images[:1]),
        rankings=warmup_rankings,
        feature_masks=torch.from_numpy(masks),
        replacement_values=torch.from_numpy(replacement_images["zero"]),
        k_values=[0, 1],
        target_classes=torch.from_numpy(frozen["predictions"][:1]),
        inference_batch_size=int(config["inference_batch_size"]),
    )

    condition_results: dict[str, dict[str, np.ndarray | int]] = {}
    started = perf_counter()
    for replacement_name, replacement_image in replacement_images.items():
        for method in methods:
            for order in ("top", "bottom"):
                key = _condition_key(method, order, replacement_name)
                result = cumulative_feature_deletion(
                    model,
                    torch.from_numpy(test_images),
                    rankings=torch.from_numpy(ranking_arrays[f"{method}__{order}"]),
                    feature_masks=torch.from_numpy(masks),
                    replacement_values=torch.from_numpy(replacement_image),
                    k_values=k_values,
                    target_classes=torch.from_numpy(frozen["predictions"]),
                    inference_batch_size=int(config["inference_batch_size"]),
                )
                condition_results[key] = _result_arrays(result)
        random_key = _condition_key("random", "random", replacement_name)
        random_result = cumulative_feature_deletion(
            model,
            torch.from_numpy(test_images),
            rankings=torch.from_numpy(random_rankings),
            feature_masks=torch.from_numpy(masks),
            replacement_values=torch.from_numpy(replacement_image),
            k_values=k_values,
            target_classes=torch.from_numpy(frozen["predictions"]),
            inference_batch_size=int(config["inference_batch_size"]),
        )
        condition_results[random_key] = _result_arrays(random_result)
    elapsed_seconds = perf_counter() - started

    output_paths["config"].write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    np.savez_compressed(output_paths["rankings"], **ranking_arrays)
    np.savez_compressed(output_paths["replacements"], **replacement_images)
    raw_arrays: dict[str, np.ndarray] = {}
    for key, result in condition_results.items():
        for array_name in (
            "target_logit_drops",
            "target_probability_drops",
            "predictions",
            "target_logits",
            "target_probabilities",
            "deleted_feature_counts",
        ):
            raw_arrays[f"{key}__{array_name}"] = np.asarray(result[array_name])
    np.savez_compressed(output_paths["raw"], **raw_arrays)

    bootstrap_rng = np.random.default_rng(int(config["bootstrap_seed"]))
    bootstrap_resamples = int(config["bootstrap_resamples"])
    confidence_level = float(config["confidence_level"])
    curve_rows: list[dict[str, Any]] = []
    with output_paths["samples"].open("w", newline="", encoding="utf-8") as sample_handle:
        sample_fieldnames = [
            "method",
            "order",
            "replacement",
            "k",
            "sample_id",
            "true_label",
            "original_prediction",
            "repeat_count",
            "mean_target_logit_drop",
            "mean_target_probability_drop",
            "prediction_flip_fraction",
            "correct_prediction_fraction",
        ]
        sample_writer = csv.DictWriter(sample_handle, fieldnames=sample_fieldnames)
        sample_writer.writeheader()
        for key, result in condition_results.items():
            method, order, replacement = key.split("__")
            logit_drops = np.asarray(result["target_logit_drops"])
            probability_drops = np.asarray(result["target_probability_drops"])
            predictions = np.asarray(result["predictions"])
            repeat_count = predictions.shape[0]
            for step_position, k in enumerate(k_values):
                curve_summary = _bootstrap_curve_point(
                    logit_drops[:, step_position],
                    probability_drops[:, step_position],
                    predictions[:, step_position],
                    labels=bundle.y_test,
                    original_predictions=frozen["predictions"],
                    resamples=bootstrap_resamples,
                    confidence_level=confidence_level,
                    rng=bootstrap_rng,
                )
                curve_rows.append(
                    {
                        "method": method,
                        "order": order,
                        "replacement": replacement,
                        "k": k,
                        "sample_count": len(test_images),
                        "repeat_count": repeat_count,
                        **curve_summary,
                    }
                )
                for sample_position, sample_id in enumerate(bundle.test_indices):
                    sample_predictions = predictions[:, step_position, sample_position]
                    sample_writer.writerow(
                        {
                            "method": method,
                            "order": order,
                            "replacement": replacement,
                            "k": k,
                            "sample_id": int(sample_id),
                            "true_label": int(bundle.y_test[sample_position]),
                            "original_prediction": int(frozen["predictions"][sample_position]),
                            "repeat_count": repeat_count,
                            "mean_target_logit_drop": float(
                                logit_drops[:, step_position, sample_position].mean()
                            ),
                            "mean_target_probability_drop": float(
                                probability_drops[:, step_position, sample_position].mean()
                            ),
                            "prediction_flip_fraction": float(
                                (
                                    sample_predictions != frozen["predictions"][sample_position]
                                ).mean()
                            ),
                            "correct_prediction_fraction": float(
                                (sample_predictions == bundle.y_test[sample_position]).mean()
                            ),
                        }
                    )

    with output_paths["curves"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)

    aopc_rows: list[dict[str, Any]] = []
    aopc_arrays: dict[str, dict[str, np.ndarray]] = {}
    aopc_summary: dict[str, dict[str, dict[str, dict[str, float]]]] = {
        replacement: {} for replacement in replacement_images
    }
    for key, result in condition_results.items():
        method, order, replacement = key.split("__")
        logit_aopc = normalized_area_over_curve(
            np.asarray(result["target_logit_drops"]),
            k_values,
            feature_count=feature_count,
            axis=1,
        )
        probability_aopc = normalized_area_over_curve(
            np.asarray(result["target_probability_drops"]),
            k_values,
            feature_count=feature_count,
            axis=1,
        )
        aopc_arrays[key] = {"logit": logit_aopc, "probability": probability_aopc}
        logit_summary = _bootstrap_mean_summary(
            logit_aopc,
            resamples=bootstrap_resamples,
            confidence_level=confidence_level,
            rng=bootstrap_rng,
        )
        probability_summary = _bootstrap_mean_summary(
            probability_aopc,
            resamples=bootstrap_resamples,
            confidence_level=confidence_level,
            rng=bootstrap_rng,
        )
        row = {
            "method": method,
            "order": order,
            "replacement": replacement,
            "sample_count": len(test_images),
            "repeat_count": logit_aopc.shape[0],
            "mean_target_logit_drop_aopc": logit_summary["mean"],
            "median_target_logit_drop_aopc": logit_summary["median"],
            "mean_target_logit_drop_aopc_ci_low": logit_summary["ci_low"],
            "mean_target_logit_drop_aopc_ci_high": logit_summary["ci_high"],
            "mean_target_probability_drop_aopc": probability_summary["mean"],
            "median_target_probability_drop_aopc": probability_summary["median"],
            "mean_target_probability_drop_aopc_ci_low": probability_summary["ci_low"],
            "mean_target_probability_drop_aopc_ci_high": probability_summary["ci_high"],
        }
        aopc_rows.append(row)
        aopc_summary[replacement].setdefault(method, {})[order] = {
            "mean_target_logit_drop_aopc": float(logit_summary["mean"]),
            "mean_target_logit_drop_aopc_ci_low": float(logit_summary["ci_low"]),
            "mean_target_logit_drop_aopc_ci_high": float(logit_summary["ci_high"]),
            "mean_target_probability_drop_aopc": float(probability_summary["mean"]),
            "mean_target_probability_drop_aopc_ci_low": float(probability_summary["ci_low"]),
            "mean_target_probability_drop_aopc_ci_high": float(probability_summary["ci_high"]),
        }
    with output_paths["aopc"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aopc_rows[0]))
        writer.writeheader()
        writer.writerows(aopc_rows)

    contrast_rows: list[dict[str, Any]] = []
    contrast_summary: dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]] = {
        replacement: {} for replacement in replacement_images
    }
    for replacement in replacement_images:
        random_key = _condition_key("random", "random", replacement)
        for method in methods:
            top_key = _condition_key(method, "top", replacement)
            bottom_key = _condition_key(method, "bottom", replacement)
            contrast_summary[replacement][method] = {}
            for contrast_name, left_key, right_key in (
                ("top_minus_random", top_key, random_key),
                ("top_minus_bottom", top_key, bottom_key),
            ):
                contrast_summary[replacement][method][contrast_name] = {}
                for metric_name, array_name in (
                    ("target_logit_drop_aopc", "logit"),
                    ("target_probability_drop_aopc", "probability"),
                ):
                    contrast = _bootstrap_contrast(
                        aopc_arrays[left_key][array_name],
                        aopc_arrays[right_key][array_name],
                        resamples=bootstrap_resamples,
                        confidence_level=confidence_level,
                        rng=bootstrap_rng,
                    )
                    contrast_rows.append(
                        {
                            "method": method,
                            "replacement": replacement,
                            "contrast": contrast_name,
                            "metric": metric_name,
                            **contrast,
                        }
                    )
                    contrast_summary[replacement][method][contrast_name][metric_name] = contrast
    with output_paths["contrasts"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(contrast_rows[0]))
        writer.writeheader()
        writer.writerows(contrast_rows)

    mask_pixel_counts = masks.reshape(len(masks), -1).sum(axis=1)
    padding_mask = encoder.padding_mask()
    expected_permutation = np.arange(feature_count)
    rankings_are_permutations = all(
        bool(np.all(np.sort(values, axis=-1) == expected_permutation))
        for values in ranking_arrays.values()
    )
    regenerated_random = np.argsort(
        np.random.default_rng(int(config["random_seed"])).random(random_rankings.shape),
        axis=2,
        kind="stable",
    )
    all_numeric = [
        np.asarray(value)
        for result in condition_results.values()
        for key, value in result.items()
        if key not in {"batch_forward_calls", "conceptual_model_evaluations"}
    ]
    first_key = next(iter(condition_results))
    reference_targets = np.asarray(condition_results[first_key]["target_classes"])
    zero_step_valid = all(
        bool(
            np.allclose(np.asarray(result["target_logit_drops"])[:, 0], 0.0)
            and np.allclose(np.asarray(result["target_probability_drops"])[:, 0], 0.0)
            and np.all(np.asarray(result["predictions"])[:, 0] == frozen["predictions"][None, :])
        )
        for result in condition_results.values()
    )
    full_endpoint_valid = True
    for replacement in replacement_images:
        keys = [key for key in condition_results if key.endswith(f"__{replacement}")]
        reference = condition_results[keys[0]]
        reference_predictions = np.asarray(reference["predictions"])[0, -1]
        reference_logit_drops = np.asarray(reference["target_logit_drops"])[0, -1]
        for key in keys:
            predictions = np.asarray(condition_results[key]["predictions"])[:, -1]
            logit_drops = np.asarray(condition_results[key]["target_logit_drops"])[:, -1]
            full_endpoint_valid &= bool(
                np.all(predictions == reference_predictions[None, :])
                and np.allclose(logit_drops, reference_logit_drops[None, :], atol=1e-6)
            )
    deleted_counts_valid = all(
        bool(
            np.all(
                np.asarray(result["deleted_feature_counts"]) == np.asarray(k_values)[None, :, None]
            )
        )
        for result in condition_results.values()
    )
    checks = {
        "working_tree_clean_at_start": not worktree_dirty_at_start,
        "all_values_finite": all(bool(np.isfinite(value).all()) for value in all_numeric),
        "one_pixel_per_grid_feature_mask": bool(np.all(mask_pixel_counts == 1)),
        "feature_masks_cover_exactly_named_pixels": bool(
            np.array_equal(masks.any(axis=0), padding_mask)
            and np.array_equal(masks.sum(axis=0), padding_mask.astype(np.int64))
        ),
        "all_rankings_are_complete_permutations": rankings_are_permutations,
        "random_control_has_requested_repeats": random_rankings.shape[0]
        == int(config["random_repeats"]),
        "random_control_reproducible_from_seed": bool(
            np.array_equal(random_rankings, regenerated_random)
        ),
        "zero_and_training_median_replacements_are_distinct": not np.allclose(
            replacement_images["zero"], replacement_images["training_median"], atol=1e-7
        ),
        "targets_match_frozen_predicted_class": bool(
            np.array_equal(reference_targets, frozen["predictions"])
            and all(
                np.array_equal(np.asarray(result["target_classes"]), reference_targets)
                for result in condition_results.values()
            )
        ),
        "zero_deletion_matches_frozen_evaluation": zero_step_valid,
        "deleted_feature_counts_match_k": deleted_counts_valid,
        "full_deletion_endpoint_is_order_invariant": full_endpoint_valid,
        "all_method_scores_share_feature_universe": all(
            scores.shape == (len(test_images), feature_count) for scores in method_scores.values()
        ),
        "all_reported_metrics_finite": all(
            _all_numeric_row_values_finite(rows) for rows in (curve_rows, aopc_rows, contrast_rows)
        ),
    }
    total_calls = sum(int(result["batch_forward_calls"]) for result in condition_results.values())
    total_evaluations = sum(
        int(result["conceptual_model_evaluations"]) for result in condition_results.values()
    )
    summary = {
        "method": "cumulative feature-deletion faithfulness",
        "evaluation_commit_sha": _git_sha(),
        "evaluation_worktree_dirty_at_start": worktree_dirty_at_start,
        "evaluation_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "model_commit_sha": json.loads((run_dir / "manifest.json").read_text())["commit_sha"],
        "target_rule": "predicted class from the frozen checkpoint",
        "ranking_methods": methods,
        "ranking_orders": ["top", "bottom"],
        "control": "100 fixed random permutations shared across methods",
        "deletion_steps": k_values,
        "replacement_rules": {
            "zero": "zero in standardized space (transformed training mean)",
            "training_median": "feature-wise transformed training median",
        },
        "uncertainty": {
            "method": "hierarchical bootstrap over samples and random-control repeats",
            "resamples": bootstrap_resamples,
            "confidence_level": confidence_level,
            "seed": int(config["bootstrap_seed"]),
        },
        "samples": len(test_images),
        "features": feature_count,
        "runtime": {
            "elapsed_seconds": elapsed_seconds,
            "conceptual_model_evaluations_executed": total_evaluations,
            "actual_batched_forward_calls": total_calls,
            "inference_batch_size": int(config["inference_batch_size"]),
            "warmup_excluded": True,
        },
        "aopc": aopc_summary,
        "contrasts": contrast_summary,
        "checks": checks,
        "interpretation_guardrail": (
            "Observed ordering is a scientific result, not a technical quality gate. "
            "Zero-replacement deletion is method-aligned with zero-based Occlusion and may "
            "favor that ranking; use the training-median sensitivity condition alongside it."
        ),
    }
    output_paths["summary"].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_curve_figure(
        output_paths["curves_figure"],
        curve_rows=curve_rows,
        methods=methods,
        replacements=replacements_requested,
    )
    _write_aopc_figure(
        output_paths["aopc_figure"],
        aopc_rows=aopc_rows,
        methods=methods,
        replacements=replacements_requested,
    )
    return {
        "status": "ok" if all(checks.values()) else "quality_gate_failed",
        "run_dir": str(run_dir),
        "runtime": summary["runtime"],
        "aopc": aopc_summary,
        "contrasts": contrast_summary,
        "checks": checks,
    }
