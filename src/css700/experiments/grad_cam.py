"""Grad-CAM generation, feature tracing, and artifact export for a frozen Grid CNN run."""

import csv
import json
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
from css700.interpretability import grad_cam, trace_attributions_to_features
from css700.models import SmallTabularImageCNN

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def _load_expected_predictions(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        return np.asarray(
            [int(row["predicted_label"]) for row in csv.DictReader(handle)], dtype=np.int64
        )


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


def _write_example_figure(
    path: Path,
    *,
    images: np.ndarray,
    normalized_maps: np.ndarray,
    feature_scores: np.ndarray,
    feature_names: tuple[str, ...],
    sample_ids: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
) -> None:
    positions = _selected_examples(labels, predictions)
    figure, axes = plt.subplots(len(positions), 3, figsize=(12, 3.6 * len(positions)))
    if len(positions) == 1:
        axes = np.expand_dims(axes, axis=0)
    for row, position in enumerate(positions):
        axes[row, 0].imshow(images[position, 0], cmap="coolwarm")
        axes[row, 0].set_title(f"Grid input (sample ID {sample_ids[position]})")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        heatmap = axes[row, 1].imshow(normalized_maps[position, 0], cmap="inferno", vmin=0, vmax=1)
        axes[row, 1].set_title(f"Grad-CAM: true={labels[position]}, pred={predictions[position]}")
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])
        figure.colorbar(heatmap, ax=axes[row, 1], fraction=0.046)

        top_indices = np.argsort(-feature_scores[position], kind="stable")[:5][::-1]
        axes[row, 2].barh(
            [feature_names[index] for index in top_indices],
            feature_scores[position, top_indices],
        )
        axes[row, 2].set_title("Top five traced features")
        axes[row, 2].set_xlabel("Raw positive-evidence score")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_grad_cam_experiment(run_dir: Path, *, batch_size: int = 32) -> dict[str, Any]:
    """Explain every frozen test sample and append only Grad-CAM outputs to its run."""

    output_paths = {
        "raw": run_dir / "attributions" / "grad_cam_raw.npy",
        "normalized": run_dir / "attributions" / "grad_cam_normalized.npy",
        "features": run_dir / "attributions" / "grad_cam_features.csv",
        "aggregate": run_dir / "metrics" / "grad_cam_aggregate_features.csv",
        "summary": run_dir / "metrics" / "grad_cam_summary.json",
        "figure": run_dir / "figures" / "grad_cam_examples.png",
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Grad-CAM outputs already exist: {existing}")

    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    seeds = resolve_experiment_seeds(config)
    dataset_config = config["dataset"]
    bundle = load_breast_cancer_splits(
        seed=seeds.data_split_seed,
        test_size=float(dataset_config["test_size"]),
        validation_size=float(dataset_config["validation_size"]),
        stratify=bool(dataset_config["stratify"]),
    )
    stored_indices = np.load(run_dir / "splits" / "indices.npz")
    if not np.array_equal(stored_indices["test"], bundle.test_indices):
        raise RuntimeError("reconstructed test indices do not match the frozen run")
    encoder = GridEncoder(padding_value=float(config["encoding"]["padding_value"]))
    encoder.fit(bundle.feature_names)
    test_images = encoder.transform(bundle.x_test)

    checkpoint = torch.load(
        run_dir / "model" / "best_checkpoint.pt", map_location="cpu", weights_only=True
    )
    model = SmallTabularImageCNN(
        input_channels=int(checkpoint["input_channels"]),
        num_classes=int(checkpoint["num_classes"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    warmup_inputs = torch.from_numpy(test_images[:1])
    grad_cam(model, warmup_inputs, target_layer=model.grad_cam_layer)
    raw_parts: list[np.ndarray] = []
    normalized_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    started = perf_counter()
    for start in range(0, len(test_images), batch_size):
        inputs = torch.from_numpy(test_images[start : start + batch_size])
        result = grad_cam(model, inputs, target_layer=model.grad_cam_layer)
        raw_parts.append(result.raw.numpy())
        normalized_parts.append(result.normalized.numpy())
        target_parts.append(result.target_classes.numpy())
        prediction_parts.append(result.logits.argmax(dim=1).numpy())
    elapsed_seconds = perf_counter() - started

    raw_maps = np.concatenate(raw_parts)
    normalized_maps = np.concatenate(normalized_parts)
    target_classes = np.concatenate(target_parts).astype(np.int64, copy=False)
    predictions = np.concatenate(prediction_parts).astype(np.int64, copy=False)
    expected_predictions = _load_expected_predictions(run_dir / "model" / "test_predictions.csv")
    feature_scores = trace_attributions_to_features(raw_maps, encoder.mapping or ())
    normalized_feature_scores = trace_attributions_to_features(
        normalized_maps, encoder.mapping or ()
    )
    np.save(output_paths["raw"], raw_maps)
    np.save(output_paths["normalized"], normalized_maps)

    with output_paths["features"].open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "sample_id",
            "true_label",
            "predicted_label",
            "target_class",
            "correct",
            "feature_index",
            "feature_name",
            "raw_attribution",
            "display_normalized_attribution",
            "within_sample_rank",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample_position, sample_id in enumerate(bundle.test_indices):
            order = np.argsort(-feature_scores[sample_position], kind="stable")
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(order) + 1)
            for feature_index, feature_name in enumerate(bundle.feature_names):
                writer.writerow(
                    {
                        "sample_id": int(sample_id),
                        "true_label": int(bundle.y_test[sample_position]),
                        "predicted_label": int(predictions[sample_position]),
                        "target_class": int(target_classes[sample_position]),
                        "correct": int(
                            bundle.y_test[sample_position] == predictions[sample_position]
                        ),
                        "feature_index": feature_index,
                        "feature_name": feature_name,
                        "raw_attribution": float(feature_scores[sample_position, feature_index]),
                        "display_normalized_attribution": float(
                            normalized_feature_scores[sample_position, feature_index]
                        ),
                        "within_sample_rank": int(ranks[feature_index]),
                    }
                )

    aggregate_groups = {
        "all_predicted_classes": np.ones(len(target_classes), dtype=np.bool_),
        **{
            f"predicted_class_{class_index}": target_classes == class_index
            for class_index in np.unique(target_classes)
        },
    }
    aggregate_summaries: dict[str, list[dict[str, str | float]]] = {}
    with output_paths["aggregate"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target_group",
                "sample_count",
                "rank",
                "feature_index",
                "feature_name",
                "mean_raw_attribution",
            ],
        )
        writer.writeheader()
        for target_group, group_mask in aggregate_groups.items():
            group_scores = feature_scores[group_mask].mean(axis=0)
            group_order = np.argsort(-group_scores, kind="stable")
            rows = [
                {
                    "target_group": target_group,
                    "sample_count": int(group_mask.sum()),
                    "rank": rank,
                    "feature_index": int(feature_index),
                    "feature_name": bundle.feature_names[feature_index],
                    "mean_raw_attribution": float(group_scores[feature_index]),
                }
                for rank, feature_index in enumerate(group_order, start=1)
            ]
            writer.writerows(rows)
            aggregate_summaries[target_group] = [
                {
                    "feature_name": str(row["feature_name"]),
                    "mean_raw_attribution": float(row["mean_raw_attribution"]),
                }
                for row in rows[:5]
            ]

    padding_mask = encoder.padding_mask()
    padding_values = raw_maps[:, ~padding_mask]
    padding_pixel_count = int((~padding_mask).sum())
    mean_padding_attribution = float(padding_values.mean()) if padding_pixel_count else None
    checks = {
        "all_values_finite": bool(np.isfinite(raw_maps).all()),
        "maps_match_input_shape": list(raw_maps.shape) == list(test_images.shape),
        "predictions_match_frozen_evaluation": bool(
            np.array_equal(predictions, expected_predictions)
        ),
        "targets_match_predicted_class": bool(np.array_equal(target_classes, predictions)),
        "all_named_features_traced": feature_scores.shape == (len(test_images), 30),
    }
    summary = {
        "method": "Grad-CAM",
        "target_rule": "predicted class",
        "target_layer": "features.2 (second convolution, before ReLU)",
        "channel_weighting": "spatial mean of target-class gradients",
        "sign_handling": "standard positive-evidence ReLU",
        "resize_rule": "bilinear interpolation with align_corners=False",
        "ranking_rule": "descending raw positive-evidence score",
        "normalization_use": "per-sample max normalization for display only",
        "samples": len(test_images),
        "features": feature_scores.shape[1],
        "map_shape": list(raw_maps.shape),
        "batch_size": batch_size,
        "warmup_samples_excluded_from_timing": 1,
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_sample": elapsed_seconds / len(test_images),
        "conceptual_forward_backward_passes_per_sample": 1,
        "padding_pixels_excluded_from_feature_tracing": padding_pixel_count,
        "mean_raw_padding_attribution": mean_padding_attribution,
        "checks": checks,
        "aggregate_warning": (
            "Mixed predicted-class averages are exploratory and are not global model importance. "
            "Use the class-stratified summaries for interpretation."
        ),
        "top_five_features_by_target_group": aggregate_summaries,
    }
    output_paths["summary"].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_example_figure(
        output_paths["figure"],
        images=test_images,
        normalized_maps=normalized_maps,
        feature_scores=feature_scores,
        feature_names=bundle.feature_names,
        sample_ids=bundle.test_indices,
        labels=bundle.y_test,
        predictions=predictions,
    )
    return {
        "status": "ok" if all(checks.values()) else "quality_gate_failed",
        "run_dir": str(run_dir),
        "samples": len(test_images),
        "seconds_per_sample": summary["seconds_per_sample"],
        "top_five_features_by_target_group": summary["top_five_features_by_target_group"],
        "checks": checks,
    }
