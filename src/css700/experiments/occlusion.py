"""Feature-wise Occlusion Sensitivity and three-method agreement for a frozen Grid CNN."""

import csv
import hashlib
import json
import subprocess
from itertools import combinations
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
from css700.evaluation import spearman_agreement, top_k_overlap
from css700.interpretability import occlusion_sensitivity, trace_attributions_to_features
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
        raise RuntimeError("the encoder must be fit before building occlusion masks")
    masks = np.zeros((len(encoder.mapping), 1, encoder.rows, encoder.columns), dtype=np.bool_)
    for pixel in encoder.mapping:
        masks[pixel.feature_index, pixel.channel, pixel.row, pixel.column] = True
    return masks


def _selected_examples(labels: np.ndarray, predictions: np.ndarray) -> list[int]:
    positions: list[int] = []
    for desired_label in np.unique(labels):
        matches = np.flatnonzero((labels == desired_label) & (labels == predictions))
        if matches.size:
            positions.append(int(matches[0]))
    errors = np.flatnonzero(labels != predictions)
    if errors.size:
        positions.append(int(errors[0]))
    return positions[:3]


def _scores_to_maps(
    scores: np.ndarray, encoder: GridEncoder, input_shape: tuple[int, ...]
) -> np.ndarray:
    maps = np.zeros(input_shape, dtype=np.float32)
    for pixel in encoder.mapping or ():
        maps[:, pixel.channel, pixel.row, pixel.column] = scores[:, pixel.feature_index]
    return maps


def _write_examples_figure(
    path: Path,
    *,
    images: np.ndarray,
    score_maps: np.ndarray,
    magnitude_scores: np.ndarray,
    feature_names: tuple[str, ...],
    sample_ids: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
) -> None:
    positions = _selected_examples(labels, predictions)
    figure, axes = plt.subplots(len(positions), 3, figsize=(12, 3.8 * len(positions)))
    if len(positions) == 1:
        axes = np.expand_dims(axes, axis=0)
    for row, position in enumerate(positions):
        axes[row, 0].imshow(images[position, 0], cmap="coolwarm")
        axes[row, 0].set_title(f"Grid input (sample ID {sample_ids[position]})")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        values = score_maps[position, 0]
        limit = max(float(np.abs(values).max()), 1e-12)
        heatmap = axes[row, 1].imshow(values, cmap="coolwarm", vmin=-limit, vmax=limit)
        axes[row, 1].set_title(
            f"Occlusion logit drop\ntrue={labels[position]}, pred={predictions[position]}"
        )
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])
        figure.colorbar(heatmap, ax=axes[row, 1], fraction=0.046)

        top_indices = np.argsort(-magnitude_scores[position], kind="stable")[:5][::-1]
        axes[row, 2].barh(
            [feature_names[index] for index in top_indices],
            magnitude_scores[position, top_indices],
        )
        axes[row, 2].set_title("Top five |target-logit drops|")
        axes[row, 2].set_xlabel("Absolute target-logit drop")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _method_display_name(name: str) -> str:
    return {
        "grad_cam": "Grad-CAM",
        "integrated_gradients_zero": "IG zero",
        "integrated_gradients_training_median": "IG median",
        "occlusion_zero": "Occlusion",
    }.get(name, name.replace("_", " "))


def _write_agreement_figure(
    path: Path,
    *,
    method_names: list[str],
    agreement_summary: dict[str, dict[str, dict[str, float | int]]],
    primary_k: int,
) -> None:
    method_positions = {name: position for position, name in enumerate(method_names)}
    spearman = np.eye(len(method_names), dtype=np.float64)
    overlap = np.eye(len(method_names), dtype=np.float64)
    for pair_name, group_summaries in agreement_summary.items():
        left_name, right_name = pair_name.split("__versus__", maxsplit=1)
        left = method_positions[left_name]
        right = method_positions[right_name]
        all_samples = group_summaries["all_samples"]
        spearman[left, right] = spearman[right, left] = float(all_samples["median_spearman"])
        overlap[left, right] = overlap[right, left] = float(
            all_samples[f"median_top_{primary_k}_overlap_fraction"]
        )

    labels = [_method_display_name(name) for name in method_names]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for axis, matrix, title, minimum in (
        (axes[0], spearman, "Median per-sample Spearman", -1.0),
        (axes[1], overlap, f"Median top-{primary_k} overlap", 0.0),
    ):
        heatmap = axis.imshow(matrix, cmap="coolwarm", vmin=minimum, vmax=1.0)
        axis.set_xticks(np.arange(len(labels)), labels=labels, rotation=35, ha="right")
        axis.set_yticks(np.arange(len(labels)), labels=labels)
        axis.set_title(title)
        for row in range(len(labels)):
            for column in range(len(labels)):
                axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center")
        figure.colorbar(heatmap, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _group_masks(predictions: np.ndarray, correct: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "all_samples": np.ones(len(predictions), dtype=np.bool_),
        "correct_predictions": correct,
        "incorrect_predictions": ~correct,
        **{
            f"predicted_class_{class_index}": predictions == class_index
            for class_index in np.unique(predictions)
        },
    }


def _method_agreement(
    method_scores: dict[str, np.ndarray],
    *,
    sample_ids: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    correct: np.ndarray,
    top_k_values: list[int],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, float | int]]]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, dict[str, dict[str, float | int]]] = {}
    groups = _group_masks(predictions, correct)
    for left_name, right_name in combinations(method_scores, 2):
        left_scores = method_scores[left_name]
        right_scores = method_scores[right_name]
        pair_rows: list[dict[str, Any]] = []
        for sample_position, sample_id in enumerate(sample_ids):
            row: dict[str, Any] = {
                "left_method": left_name,
                "right_method": right_name,
                "sample_id": int(sample_id),
                "true_label": int(labels[sample_position]),
                "predicted_label": int(predictions[sample_position]),
                "correct": int(correct[sample_position]),
                "spearman": spearman_agreement(
                    left_scores[sample_position], right_scores[sample_position]
                ),
            }
            for k in top_k_values:
                result = top_k_overlap(
                    left_scores[sample_position], right_scores[sample_position], k=k
                )
                row[f"top_{k}_intersection"] = result["intersection"]
                row[f"top_{k}_overlap_fraction"] = result["overlap_fraction"]
                row[f"top_{k}_jaccard"] = result["jaccard"]
            rows.append(row)
            pair_rows.append(row)

        pair_name = f"{left_name}__versus__{right_name}"
        summary[pair_name] = {}
        for group_name, group_mask in groups.items():
            if not group_mask.any():
                continue
            group_rows = [row for row, keep in zip(pair_rows, group_mask, strict=True) if keep]
            summary[pair_name][group_name] = {
                "sample_count": len(group_rows),
                "median_spearman": float(np.median([row["spearman"] for row in group_rows])),
                **{
                    f"median_top_{k}_overlap_fraction": float(
                        np.median([row[f"top_{k}_overlap_fraction"] for row in group_rows])
                    )
                    for k in top_k_values
                },
            }
    return rows, summary


def run_occlusion_experiment(run_dir: Path, config_path: Path) -> dict[str, Any]:
    """Append feature-wise occlusion and full method comparisons to a frozen run."""

    output_paths = {
        "config": run_dir / "attributions" / "occlusion_config.yaml",
        "masks": run_dir / "attributions" / "occlusion_feature_masks.npy",
        "raw": run_dir / "attributions" / "occlusion_raw.npz",
        "features": run_dir / "attributions" / "occlusion_features.csv",
        "aggregate": run_dir / "metrics" / "occlusion_aggregate_features.csv",
        "agreement": run_dir / "metrics" / "three_method_agreement.csv",
        "summary": run_dir / "metrics" / "occlusion_summary.json",
        "examples_figure": run_dir / "figures" / "occlusion_examples.png",
        "agreement_figure": run_dir / "figures" / "three_method_agreement.png",
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Occlusion outputs already exist: {existing}")
    required_outputs = [
        run_dir / "attributions" / "grad_cam_raw.npy",
        run_dir / "attributions" / "integrated_gradients_raw.npz",
    ]
    missing = [str(path) for path in required_outputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"the frozen run is missing prerequisite outputs: {missing}")

    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    if config["replacement"] != "zero_standardized":
        raise ValueError("the Grid pilot currently supports only zero_standardized replacement")
    if config["ranking"] != "absolute_target_logit_drop":
        raise ValueError("the Grid pilot requires absolute_target_logit_drop ranking")

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
    replacement = np.zeros((1, *test_images.shape[1:]), dtype=np.float32)
    occlusion_sensitivity(
        model,
        torch.from_numpy(test_images[:1]),
        feature_masks=torch.from_numpy(masks),
        replacement_values=torch.from_numpy(replacement),
        target_classes=torch.from_numpy(frozen["predictions"][:1]),
        perturbation_batch_size=int(config["perturbation_batch_size"]),
    )
    started = perf_counter()
    result = occlusion_sensitivity(
        model,
        torch.from_numpy(test_images),
        feature_masks=torch.from_numpy(masks),
        replacement_values=torch.from_numpy(replacement),
        target_classes=torch.from_numpy(frozen["predictions"]),
        perturbation_batch_size=int(config["perturbation_batch_size"]),
    )
    elapsed_seconds = perf_counter() - started

    logit_drops = result.logit_drops.numpy()
    probability_drops = result.probability_drops.numpy()
    occluded_target_logits = result.occluded_target_logits.numpy()
    occluded_target_probabilities = result.occluded_target_probabilities.numpy()
    original_logits = result.original_logits.numpy()
    original_probabilities = result.original_probabilities.numpy()
    target_classes = result.target_classes.numpy().astype(np.int64, copy=False)
    predictions = original_logits.argmax(axis=1).astype(np.int64, copy=False)
    original_target_logits = original_logits[np.arange(len(test_images)), target_classes]
    original_target_probabilities = original_probabilities[
        np.arange(len(test_images)), target_classes
    ]
    magnitude_scores = np.abs(logit_drops)

    output_paths["config"].write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    np.save(output_paths["masks"], masks)
    np.savez_compressed(
        output_paths["raw"],
        signed_target_logit_drop=logit_drops,
        signed_target_probability_drop=probability_drops,
        occluded_target_logits=occluded_target_logits,
        occluded_target_probabilities=occluded_target_probabilities,
        original_logits=original_logits,
        original_probabilities=original_probabilities,
        target_classes=target_classes,
    )

    with output_paths["features"].open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "sample_id",
            "true_label",
            "predicted_label",
            "target_class",
            "correct",
            "feature_index",
            "feature_name",
            "channel",
            "row",
            "column",
            "replacement_value",
            "original_target_logit",
            "occluded_target_logit",
            "signed_target_logit_drop",
            "absolute_target_logit_drop",
            "original_target_probability",
            "occluded_target_probability",
            "signed_target_probability_drop",
            "absolute_logit_drop_rank",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        mapping = encoder.mapping or ()
        for sample_position, sample_id in enumerate(bundle.test_indices):
            order = np.argsort(-magnitude_scores[sample_position], kind="stable")
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(order) + 1)
            for feature_index, feature_name in enumerate(bundle.feature_names):
                pixel = mapping[feature_index]
                writer.writerow(
                    {
                        "sample_id": int(sample_id),
                        "true_label": int(bundle.y_test[sample_position]),
                        "predicted_label": int(predictions[sample_position]),
                        "target_class": int(target_classes[sample_position]),
                        "correct": int(frozen["correct"][sample_position]),
                        "feature_index": feature_index,
                        "feature_name": feature_name,
                        "channel": pixel.channel,
                        "row": pixel.row,
                        "column": pixel.column,
                        "replacement_value": 0.0,
                        "original_target_logit": float(original_target_logits[sample_position]),
                        "occluded_target_logit": float(
                            occluded_target_logits[sample_position, feature_index]
                        ),
                        "signed_target_logit_drop": float(
                            logit_drops[sample_position, feature_index]
                        ),
                        "absolute_target_logit_drop": float(
                            magnitude_scores[sample_position, feature_index]
                        ),
                        "original_target_probability": float(
                            original_target_probabilities[sample_position]
                        ),
                        "occluded_target_probability": float(
                            occluded_target_probabilities[sample_position, feature_index]
                        ),
                        "signed_target_probability_drop": float(
                            probability_drops[sample_position, feature_index]
                        ),
                        "absolute_logit_drop_rank": int(ranks[feature_index]),
                    }
                )

    aggregate_summaries: dict[str, list[dict[str, str | float]]] = {}
    groups = {
        "all_predicted_classes": np.ones(len(target_classes), dtype=np.bool_),
        **{
            f"predicted_class_{class_index}": target_classes == class_index
            for class_index in np.unique(target_classes)
        },
    }
    with output_paths["aggregate"].open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "target_group",
            "sample_count",
            "rank",
            "feature_index",
            "feature_name",
            "mean_signed_target_logit_drop",
            "mean_absolute_target_logit_drop",
            "mean_signed_target_probability_drop",
            "positive_support_fraction",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group_name, group_mask in groups.items():
            mean_signed = logit_drops[group_mask].mean(axis=0)
            mean_absolute = magnitude_scores[group_mask].mean(axis=0)
            mean_probability = probability_drops[group_mask].mean(axis=0)
            positive_fraction = (logit_drops[group_mask] > 0).mean(axis=0)
            order = np.argsort(-mean_absolute, kind="stable")
            rows = [
                {
                    "target_group": group_name,
                    "sample_count": int(group_mask.sum()),
                    "rank": rank,
                    "feature_index": int(feature_index),
                    "feature_name": bundle.feature_names[feature_index],
                    "mean_signed_target_logit_drop": float(mean_signed[feature_index]),
                    "mean_absolute_target_logit_drop": float(mean_absolute[feature_index]),
                    "mean_signed_target_probability_drop": float(mean_probability[feature_index]),
                    "positive_support_fraction": float(positive_fraction[feature_index]),
                }
                for rank, feature_index in enumerate(order, start=1)
            ]
            writer.writerows(rows)
            aggregate_summaries[group_name] = [
                {
                    "feature_name": str(row["feature_name"]),
                    "mean_signed_target_logit_drop": float(row["mean_signed_target_logit_drop"]),
                    "mean_absolute_target_logit_drop": float(
                        row["mean_absolute_target_logit_drop"]
                    ),
                }
                for row in rows[:5]
            ]

    grad_cam_maps = np.load(run_dir / "attributions" / "grad_cam_raw.npy")
    grad_cam_scores = trace_attributions_to_features(grad_cam_maps, encoder.mapping or ())
    method_scores = {"grad_cam": grad_cam_scores}
    with np.load(run_dir / "attributions" / "integrated_gradients_raw.npz") as ig_raw:
        if "zero" not in ig_raw.files:
            raise RuntimeError("Integrated Gradients outputs must include the zero baseline")
        method_scores["integrated_gradients_zero"] = np.abs(
            trace_attributions_to_features(ig_raw["zero"], encoder.mapping or ())
        )
        if "training_median" in ig_raw.files:
            method_scores["integrated_gradients_training_median"] = np.abs(
                trace_attributions_to_features(ig_raw["training_median"], encoder.mapping or ())
            )
    method_scores["occlusion_zero"] = magnitude_scores
    top_k_values = [int(value) for value in config["top_k"]]
    agreement_rows, agreement_summary = _method_agreement(
        method_scores,
        sample_ids=bundle.test_indices,
        labels=bundle.y_test,
        predictions=predictions,
        correct=frozen["correct"],
        top_k_values=top_k_values,
    )
    with output_paths["agreement"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(agreement_rows[0]))
        writer.writeheader()
        writer.writerows(agreement_rows)

    padding_mask = encoder.padding_mask()
    mask_pixel_counts = masks.reshape(len(masks), -1).sum(axis=1)
    all_raw_arrays = (
        logit_drops,
        probability_drops,
        occluded_target_logits,
        occluded_target_probabilities,
        original_logits,
        original_probabilities,
    )
    agreement_values = np.asarray([row["spearman"] for row in agreement_rows], dtype=np.float64)
    checks = {
        "working_tree_clean_at_start": not worktree_dirty_at_start,
        "all_values_finite": all(bool(np.isfinite(values).all()) for values in all_raw_arrays),
        "one_pixel_per_grid_feature_mask": bool(np.all(mask_pixel_counts == 1)),
        "feature_masks_cover_exactly_named_pixels": bool(
            np.array_equal(masks.any(axis=0), padding_mask)
            and np.array_equal(masks.sum(axis=0), padding_mask.astype(np.int64))
        ),
        "replacement_is_standardized_zero": bool(np.all(replacement == 0.0)),
        "predictions_match_frozen_evaluation": bool(
            np.array_equal(predictions, frozen["predictions"])
        ),
        "targets_match_predicted_class": bool(
            np.array_equal(target_classes, frozen["predictions"])
        ),
        "all_named_features_scored": logit_drops.shape
        == (len(test_images), len(bundle.feature_names)),
        "all_method_scores_share_feature_universe": all(
            values.shape == logit_drops.shape for values in method_scores.values()
        ),
        "all_agreement_values_finite": bool(np.isfinite(agreement_values).all()),
    }
    perturbed_evaluations = len(test_images) * len(bundle.feature_names)
    summary = {
        "method": "feature-wise Occlusion Sensitivity",
        "explainer_commit_sha": _git_sha(),
        "explainer_worktree_dirty_at_start": worktree_dirty_at_start,
        "explainer_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "model_commit_sha": json.loads((run_dir / "manifest.json").read_text())["commit_sha"],
        "target_rule": "predicted class from the frozen checkpoint",
        "perturbation_unit": "one complete mapped Grid feature pixel at a time",
        "replacement_rule": "zero in standardized input space (transformed training mean)",
        "primary_score": "original target logit minus occluded target logit",
        "secondary_score": "original target probability minus occluded target probability",
        "ranking_rule": "absolute target-logit drop; signed raw values retained",
        "samples": len(test_images),
        "features": len(bundle.feature_names),
        "runtime": {
            "perturbation_batch_size": int(config["perturbation_batch_size"]),
            "warmup_excluded": True,
            "elapsed_seconds": elapsed_seconds,
            "seconds_per_sample": elapsed_seconds / len(test_images),
            "seconds_per_feature_perturbation": elapsed_seconds / perturbed_evaluations,
            "conceptual_original_model_evaluations": len(test_images),
            "conceptual_perturbed_model_evaluations": perturbed_evaluations,
            "conceptual_total_model_evaluations": len(test_images) + perturbed_evaluations,
            "actual_batched_forward_calls": result.batch_forward_calls,
        },
        "top_five_features_by_target_group": aggregate_summaries,
        "method_agreement": agreement_summary,
        "checks": checks,
    }
    output_paths["summary"].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    score_maps = _scores_to_maps(logit_drops, encoder, test_images.shape)
    _write_examples_figure(
        output_paths["examples_figure"],
        images=test_images,
        score_maps=score_maps,
        magnitude_scores=magnitude_scores,
        feature_names=bundle.feature_names,
        sample_ids=bundle.test_indices,
        labels=bundle.y_test,
        predictions=predictions,
    )
    _write_agreement_figure(
        output_paths["agreement_figure"],
        method_names=list(method_scores),
        agreement_summary=agreement_summary,
        primary_k=top_k_values[0],
    )
    return {
        "status": "ok" if all(checks.values()) else "quality_gate_failed",
        "run_dir": str(run_dir),
        "runtime": summary["runtime"],
        "top_five_features_by_target_group": aggregate_summaries,
        "method_agreement": agreement_summary,
        "checks": checks,
    }
