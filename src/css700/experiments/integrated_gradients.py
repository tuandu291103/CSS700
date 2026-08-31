"""Integrated Gradients tracing and Grad-CAM agreement for a frozen Grid CNN run."""

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

from css700.data import load_breast_cancer_splits
from css700.encoding import GridEncoder
from css700.evaluation import spearman_agreement, top_k_overlap
from css700.interpretability import integrated_gradients, trace_attributions_to_features
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


def _build_baselines(
    train_values: np.ndarray,
    encoder: GridEncoder,
    requested_names: list[str],
    *,
    equivalence_atol: float,
) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, dict[str, float]]]:
    vectors = {
        "zero": np.zeros((1, train_values.shape[1]), dtype=np.float64),
        "training_mean": train_values.mean(axis=0, keepdims=True),
        "training_median": np.median(train_values, axis=0, keepdims=True),
    }
    unknown = sorted(set(requested_names) - set(vectors))
    if unknown:
        raise ValueError(f"unsupported Integrated Gradients baselines: {unknown}")

    requested_images = {name: encoder.transform(vectors[name]) for name in requested_names}
    canonical_images: dict[str, np.ndarray] = {}
    aliases: dict[str, str] = {}
    for name in requested_names:
        equivalent_name = next(
            (
                canonical_name
                for canonical_name, image in canonical_images.items()
                if np.allclose(requested_images[name], image, rtol=0.0, atol=equivalence_atol)
            ),
            None,
        )
        if equivalent_name is None:
            canonical_images[name] = requested_images[name]
            aliases[name] = name
        else:
            aliases[name] = equivalent_name

    zero_image = requested_images["zero"] if "zero" in requested_images else np.zeros((1, 1, 1, 1))
    diagnostics = {
        name: {
            "maximum_absolute_value": float(np.abs(image).max()),
            "maximum_absolute_difference_from_zero": float(np.abs(image - zero_image).max()),
        }
        for name, image in requested_images.items()
    }
    return canonical_images, aliases, diagnostics


def _converged(
    absolute_delta: np.ndarray,
    relative_delta: np.ndarray,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> np.ndarray:
    return (absolute_delta <= absolute_tolerance) | (relative_delta <= relative_tolerance)


def _attribute_baseline(
    model: SmallTabularImageCNN,
    test_images: np.ndarray,
    target_classes: np.ndarray,
    baseline_image: np.ndarray,
    *,
    sample_batch_size: int,
    internal_batch_size: int,
    n_steps: int,
    retry_n_steps: int,
    integration_method: str,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, np.ndarray | float]:
    samples = len(test_images)
    attributions = np.empty_like(test_images, dtype=np.float32)
    convergence_delta = np.empty(samples, dtype=np.float32)
    relative_delta = np.empty(samples, dtype=np.float32)
    output_difference = np.empty(samples, dtype=np.float32)
    predictions = np.empty(samples, dtype=np.int64)
    targets = np.empty(samples, dtype=np.int64)
    steps_used = np.full(samples, n_steps, dtype=np.int64)

    warmup = torch.from_numpy(test_images[:1])
    integrated_gradients(
        model,
        warmup,
        baselines=torch.from_numpy(baseline_image),
        target_classes=torch.from_numpy(target_classes[:1]),
        n_steps=min(8, n_steps),
        integration_method=integration_method,
        internal_batch_size=max(8, min(internal_batch_size, n_steps)),
    )
    started = perf_counter()
    for start in range(0, samples, sample_batch_size):
        stop = min(start + sample_batch_size, samples)
        inputs = torch.from_numpy(test_images[start:stop])
        result = integrated_gradients(
            model,
            inputs,
            baselines=torch.from_numpy(baseline_image),
            target_classes=torch.from_numpy(target_classes[start:stop]),
            n_steps=n_steps,
            integration_method=integration_method,
            internal_batch_size=max(internal_batch_size, stop - start),
        )
        attributions[start:stop] = result.attributions.numpy()
        convergence_delta[start:stop] = result.convergence_delta.numpy()
        relative_delta[start:stop] = result.relative_convergence_delta.numpy()
        output_difference[start:stop] = result.output_difference.numpy()
        predictions[start:stop] = result.logits.argmax(dim=1).numpy()
        targets[start:stop] = result.target_classes.numpy()

    initial_converged = _converged(
        np.abs(convergence_delta),
        relative_delta,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    retry_positions = np.flatnonzero(~initial_converged)
    for retry_start in range(0, len(retry_positions), sample_batch_size):
        positions = retry_positions[retry_start : retry_start + sample_batch_size]
        inputs = torch.from_numpy(test_images[positions])
        result = integrated_gradients(
            model,
            inputs,
            baselines=torch.from_numpy(baseline_image),
            target_classes=torch.from_numpy(target_classes[positions]),
            n_steps=retry_n_steps,
            integration_method=integration_method,
            internal_batch_size=max(internal_batch_size, len(positions)),
        )
        attributions[positions] = result.attributions.numpy()
        convergence_delta[positions] = result.convergence_delta.numpy()
        relative_delta[positions] = result.relative_convergence_delta.numpy()
        output_difference[positions] = result.output_difference.numpy()
        predictions[positions] = result.logits.argmax(dim=1).numpy()
        targets[positions] = result.target_classes.numpy()
        steps_used[positions] = retry_n_steps

    elapsed_seconds = perf_counter() - started
    final_converged = _converged(
        np.abs(convergence_delta),
        relative_delta,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    return {
        "attributions": attributions,
        "convergence_delta": convergence_delta,
        "relative_delta": relative_delta,
        "output_difference": output_difference,
        "predictions": predictions,
        "targets": targets,
        "steps_used": steps_used,
        "initial_converged": initial_converged,
        "final_converged": final_converged,
        "elapsed_seconds": elapsed_seconds,
    }


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


def _display_baseline_name(name: str) -> str:
    if name == "zero":
        return "zero / train mean"
    return name.replace("_", " ")


def _write_examples_figure(
    path: Path,
    *,
    images: np.ndarray,
    attribution_maps: dict[str, np.ndarray],
    feature_magnitudes: dict[str, np.ndarray],
    feature_names: tuple[str, ...],
    sample_ids: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
) -> None:
    positions = _selected_examples(labels, predictions)
    baseline_names = list(attribution_maps)
    columns = 2 + len(baseline_names)
    figure, axes = plt.subplots(
        len(positions), columns, figsize=(4 * columns, 3.6 * len(positions))
    )
    if len(positions) == 1:
        axes = np.expand_dims(axes, axis=0)
    primary_baseline = baseline_names[0]
    for row, position in enumerate(positions):
        axes[row, 0].imshow(images[position, 0], cmap="coolwarm")
        axes[row, 0].set_title(f"Grid input (sample ID {sample_ids[position]})")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        for offset, baseline_name in enumerate(baseline_names, start=1):
            values = attribution_maps[baseline_name][position, 0]
            limit = max(float(np.abs(values).max()), 1e-12)
            heatmap = axes[row, offset].imshow(values, cmap="coolwarm", vmin=-limit, vmax=limit)
            axes[row, offset].set_title(
                f"IG {_display_baseline_name(baseline_name)}\n"
                f"true={labels[position]}, pred={predictions[position]}",
                fontsize=10,
            )
            axes[row, offset].set_xticks([])
            axes[row, offset].set_yticks([])
            figure.colorbar(heatmap, ax=axes[row, offset], fraction=0.046)

        scores = feature_magnitudes[primary_baseline][position]
        top_indices = np.argsort(-scores, kind="stable")[:5][::-1]
        axes[row, -1].barh([feature_names[index] for index in top_indices], scores[top_indices])
        axes[row, -1].set_title(
            f"Top five |IG|\n{_display_baseline_name(primary_baseline)}", fontsize=10
        )
        axes[row, -1].set_xlabel("Absolute attribution")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_agreement_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    baselines = list(dict.fromkeys(str(row["baseline"]) for row in rows))
    spearman = [
        [float(row["spearman_with_grad_cam"]) for row in rows if row["baseline"] == baseline]
        for baseline in baselines
    ]
    top_five = [
        [float(row["top_5_overlap_fraction"]) for row in rows if row["baseline"] == baseline]
        for baseline in baselines
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    display_names = [_display_baseline_name(name) for name in baselines]
    axes[0].boxplot(spearman, tick_labels=display_names)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(title="IG versus Grad-CAM rank agreement", ylabel="Spearman correlation")
    axes[1].boxplot(top_five, tick_labels=display_names)
    axes[1].set(title="IG versus Grad-CAM top-5 agreement", ylabel="Overlap fraction", ylim=(0, 1))
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _distribution_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "q1": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q3": float(np.quantile(values, 0.75)),
        "maximum": float(np.max(values)),
    }


def run_integrated_gradients_experiment(run_dir: Path, config_path: Path) -> dict[str, Any]:
    """Append Integrated Gradients and paired Grad-CAM comparisons to a frozen run."""

    output_paths = {
        "config": run_dir / "attributions" / "integrated_gradients_config.yaml",
        "baselines": run_dir / "attributions" / "integrated_gradients_baselines.npz",
        "raw": run_dir / "attributions" / "integrated_gradients_raw.npz",
        "features": run_dir / "attributions" / "integrated_gradients_features.csv",
        "aggregate": run_dir / "metrics" / "integrated_gradients_aggregate_features.csv",
        "agreement": run_dir / "metrics" / "integrated_gradients_grad_cam_agreement.csv",
        "baseline_agreement": (run_dir / "metrics" / "integrated_gradients_baseline_agreement.csv"),
        "summary": run_dir / "metrics" / "integrated_gradients_summary.json",
        "examples_figure": run_dir / "figures" / "integrated_gradients_examples.png",
        "agreement_figure": run_dir / "figures" / "integrated_gradients_grad_cam_agreement.png",
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Integrated Gradients outputs already exist: {existing}")
    if not (run_dir / "attributions" / "grad_cam_raw.npy").exists():
        raise FileNotFoundError("the frozen run must contain Grad-CAM outputs before comparison")

    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    worktree_dirty_at_start = _git_worktree_dirty()
    training_config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    seed = int(training_config["experiment"]["seed"])
    dataset_config = training_config["dataset"]
    bundle = load_breast_cancer_splits(
        seed=seed,
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

    canonical_baselines, baseline_aliases, baseline_diagnostics = _build_baselines(
        bundle.x_train,
        encoder,
        [str(name) for name in config["baselines"]],
        equivalence_atol=float(config["baseline_equivalence_atol"]),
    )
    results: dict[str, dict[str, np.ndarray | float]] = {}
    for baseline_name, baseline_image in canonical_baselines.items():
        results[baseline_name] = _attribute_baseline(
            model,
            test_images,
            frozen["predictions"],
            baseline_image,
            sample_batch_size=int(config["sample_batch_size"]),
            internal_batch_size=int(config["internal_batch_size"]),
            n_steps=int(config["n_steps"]),
            retry_n_steps=int(config["retry_n_steps"]),
            integration_method=str(config["integration_method"]),
            absolute_tolerance=float(config["absolute_convergence_tolerance"]),
            relative_tolerance=float(config["relative_convergence_tolerance"]),
        )

    raw_attributions = {
        name: np.asarray(result["attributions"]) for name, result in results.items()
    }
    signed_feature_scores = {
        name: trace_attributions_to_features(values, encoder.mapping or ())
        for name, values in raw_attributions.items()
    }
    magnitude_feature_scores = {
        name: np.abs(values) for name, values in signed_feature_scores.items()
    }
    np.savez_compressed(output_paths["raw"], **raw_attributions)
    np.savez_compressed(output_paths["baselines"], **canonical_baselines)
    output_paths["config"].write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    alias_groups = {
        canonical: sorted(name for name, target in baseline_aliases.items() if target == canonical)
        for canonical in canonical_baselines
    }
    with output_paths["features"].open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "baseline",
            "baseline_aliases",
            "sample_id",
            "true_label",
            "predicted_label",
            "target_class",
            "correct",
            "feature_index",
            "feature_name",
            "signed_attribution",
            "absolute_attribution",
            "absolute_rank",
            "convergence_delta",
            "relative_convergence_delta",
            "integration_steps",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for baseline_name, feature_scores in signed_feature_scores.items():
            result = results[baseline_name]
            for sample_position, sample_id in enumerate(bundle.test_indices):
                magnitudes = np.abs(feature_scores[sample_position])
                order = np.argsort(-magnitudes, kind="stable")
                ranks = np.empty_like(order)
                ranks[order] = np.arange(1, len(order) + 1)
                for feature_index, feature_name in enumerate(bundle.feature_names):
                    writer.writerow(
                        {
                            "baseline": baseline_name,
                            "baseline_aliases": "|".join(alias_groups[baseline_name]),
                            "sample_id": int(sample_id),
                            "true_label": int(bundle.y_test[sample_position]),
                            "predicted_label": int(frozen["predictions"][sample_position]),
                            "target_class": int(np.asarray(result["targets"])[sample_position]),
                            "correct": int(frozen["correct"][sample_position]),
                            "feature_index": feature_index,
                            "feature_name": feature_name,
                            "signed_attribution": float(
                                feature_scores[sample_position, feature_index]
                            ),
                            "absolute_attribution": float(magnitudes[feature_index]),
                            "absolute_rank": int(ranks[feature_index]),
                            "convergence_delta": float(
                                np.asarray(result["convergence_delta"])[sample_position]
                            ),
                            "relative_convergence_delta": float(
                                np.asarray(result["relative_delta"])[sample_position]
                            ),
                            "integration_steps": int(
                                np.asarray(result["steps_used"])[sample_position]
                            ),
                        }
                    )

    aggregate_summaries: dict[str, dict[str, list[dict[str, str | float]]]] = {}
    with output_paths["aggregate"].open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "baseline",
            "target_group",
            "sample_count",
            "rank",
            "feature_index",
            "feature_name",
            "mean_signed_attribution",
            "mean_absolute_attribution",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for baseline_name, feature_scores in signed_feature_scores.items():
            target_classes = np.asarray(results[baseline_name]["targets"])
            groups = {
                "all_predicted_classes": np.ones(len(target_classes), dtype=np.bool_),
                **{
                    f"predicted_class_{class_index}": target_classes == class_index
                    for class_index in np.unique(target_classes)
                },
            }
            aggregate_summaries[baseline_name] = {}
            for target_group, group_mask in groups.items():
                mean_signed = feature_scores[group_mask].mean(axis=0)
                mean_absolute = np.abs(feature_scores[group_mask]).mean(axis=0)
                order = np.argsort(-mean_absolute, kind="stable")
                rows = [
                    {
                        "baseline": baseline_name,
                        "target_group": target_group,
                        "sample_count": int(group_mask.sum()),
                        "rank": rank,
                        "feature_index": int(feature_index),
                        "feature_name": bundle.feature_names[feature_index],
                        "mean_signed_attribution": float(mean_signed[feature_index]),
                        "mean_absolute_attribution": float(mean_absolute[feature_index]),
                    }
                    for rank, feature_index in enumerate(order, start=1)
                ]
                writer.writerows(rows)
                aggregate_summaries[baseline_name][target_group] = [
                    {
                        "feature_name": str(row["feature_name"]),
                        "mean_signed_attribution": float(row["mean_signed_attribution"]),
                        "mean_absolute_attribution": float(row["mean_absolute_attribution"]),
                    }
                    for row in rows[:5]
                ]

    grad_cam_maps = np.load(run_dir / "attributions" / "grad_cam_raw.npy")
    grad_cam_features = trace_attributions_to_features(grad_cam_maps, encoder.mapping or ())
    agreement_rows: list[dict[str, Any]] = []
    top_k_values = [int(value) for value in config["top_k"]]
    for baseline_name, magnitude_scores in magnitude_feature_scores.items():
        for sample_position, sample_id in enumerate(bundle.test_indices):
            row: dict[str, Any] = {
                "baseline": baseline_name,
                "sample_id": int(sample_id),
                "true_label": int(bundle.y_test[sample_position]),
                "predicted_label": int(frozen["predictions"][sample_position]),
                "correct": int(frozen["correct"][sample_position]),
                "spearman_with_grad_cam": spearman_agreement(
                    magnitude_scores[sample_position], grad_cam_features[sample_position]
                ),
            }
            for k in top_k_values:
                overlap = top_k_overlap(
                    magnitude_scores[sample_position], grad_cam_features[sample_position], k=k
                )
                row[f"top_{k}_intersection"] = overlap["intersection"]
                row[f"top_{k}_overlap_fraction"] = overlap["overlap_fraction"]
                row[f"top_{k}_jaccard"] = overlap["jaccard"]
            agreement_rows.append(row)
    with output_paths["agreement"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(agreement_rows[0]))
        writer.writeheader()
        writer.writerows(agreement_rows)

    baseline_agreement_rows: list[dict[str, Any]] = []
    for left_baseline, right_baseline in combinations(magnitude_feature_scores, 2):
        left_scores = magnitude_feature_scores[left_baseline]
        right_scores = magnitude_feature_scores[right_baseline]
        for sample_position, sample_id in enumerate(bundle.test_indices):
            row = {
                "left_baseline": left_baseline,
                "right_baseline": right_baseline,
                "sample_id": int(sample_id),
                "true_label": int(bundle.y_test[sample_position]),
                "predicted_label": int(frozen["predictions"][sample_position]),
                "correct": int(frozen["correct"][sample_position]),
                "spearman": spearman_agreement(
                    left_scores[sample_position], right_scores[sample_position]
                ),
            }
            for k in top_k_values:
                overlap = top_k_overlap(
                    left_scores[sample_position], right_scores[sample_position], k=k
                )
                row[f"top_{k}_intersection"] = overlap["intersection"]
                row[f"top_{k}_overlap_fraction"] = overlap["overlap_fraction"]
                row[f"top_{k}_jaccard"] = overlap["jaccard"]
            baseline_agreement_rows.append(row)
    with output_paths["baseline_agreement"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(baseline_agreement_rows[0]))
        writer.writeheader()
        writer.writerows(baseline_agreement_rows)

    agreement_summaries: dict[str, dict[str, dict[str, float | int]]] = {}
    for baseline_name in canonical_baselines:
        baseline_rows = [row for row in agreement_rows if row["baseline"] == baseline_name]
        agreement_summaries[baseline_name] = {}
        group_masks = {
            "all_samples": np.ones(len(baseline_rows), dtype=np.bool_),
            "correct_predictions": np.asarray(
                [bool(row["correct"]) for row in baseline_rows], dtype=np.bool_
            ),
            "incorrect_predictions": np.asarray(
                [not bool(row["correct"]) for row in baseline_rows], dtype=np.bool_
            ),
            **{
                f"predicted_class_{class_index}": np.asarray(
                    [int(row["predicted_label"]) == class_index for row in baseline_rows],
                    dtype=np.bool_,
                )
                for class_index in sorted({int(row["predicted_label"]) for row in baseline_rows})
            },
        }
        for group_name, group_mask in group_masks.items():
            if not group_mask.any():
                continue
            group_rows = [row for row, keep in zip(baseline_rows, group_mask, strict=True) if keep]
            agreement_summaries[baseline_name][group_name] = {
                "sample_count": len(group_rows),
                "median_spearman": float(
                    np.median([row["spearman_with_grad_cam"] for row in group_rows])
                ),
                **{
                    f"median_top_{k}_overlap_fraction": float(
                        np.median([row[f"top_{k}_overlap_fraction"] for row in group_rows])
                    )
                    for k in top_k_values
                },
            }

    baseline_agreement_summary: dict[str, dict[str, float | int]] = {}
    for left_baseline, right_baseline in combinations(magnitude_feature_scores, 2):
        pair_rows = [
            row
            for row in baseline_agreement_rows
            if row["left_baseline"] == left_baseline and row["right_baseline"] == right_baseline
        ]
        pair_name = f"{left_baseline}_versus_{right_baseline}"
        baseline_agreement_summary[pair_name] = {
            "sample_count": len(pair_rows),
            "median_spearman": float(np.median([row["spearman"] for row in pair_rows])),
            **{
                f"median_top_{k}_overlap_fraction": float(
                    np.median([row[f"top_{k}_overlap_fraction"] for row in pair_rows])
                )
                for k in top_k_values
            },
        }

    convergence_summary: dict[str, dict[str, Any]] = {}
    for baseline_name, result in results.items():
        absolute_delta = np.abs(np.asarray(result["convergence_delta"]))
        relative_delta = np.asarray(result["relative_delta"])
        convergence_summary[baseline_name] = {
            "initial_pass_count": int(np.asarray(result["initial_converged"]).sum()),
            "final_pass_count": int(np.asarray(result["final_converged"]).sum()),
            "retry_count": int((np.asarray(result["steps_used"]) > int(config["n_steps"])).sum()),
            "absolute_delta": _distribution_summary(absolute_delta),
            "relative_delta": _distribution_summary(relative_delta),
            "elapsed_seconds": float(result["elapsed_seconds"]),
            "seconds_per_sample": float(result["elapsed_seconds"]) / len(test_images),
        }

    agreement_values = np.asarray(
        [row["spearman_with_grad_cam"] for row in agreement_rows], dtype=np.float64
    )
    baseline_agreement_values = np.asarray(
        [row["spearman"] for row in baseline_agreement_rows], dtype=np.float64
    )
    checks = {
        "working_tree_clean_at_start": not worktree_dirty_at_start,
        "zero_and_training_mean_deduplicated": baseline_aliases.get("training_mean")
        == baseline_aliases.get("zero"),
        "distinct_training_median_retained": baseline_aliases.get("training_median")
        != baseline_aliases.get("zero"),
        "all_attributions_finite": all(
            bool(np.isfinite(values).all()) for values in raw_attributions.values()
        ),
        "maps_match_input_shape": all(
            values.shape == test_images.shape for values in raw_attributions.values()
        ),
        "predictions_match_frozen_evaluation": all(
            bool(np.array_equal(np.asarray(result["predictions"]), frozen["predictions"]))
            for result in results.values()
        ),
        "targets_match_predicted_class": all(
            bool(np.array_equal(np.asarray(result["targets"]), frozen["predictions"]))
            for result in results.values()
        ),
        "all_named_features_traced": all(
            values.shape == (len(test_images), len(bundle.feature_names))
            for values in signed_feature_scores.values()
        ),
        "all_samples_pass_convergence_after_retry": all(
            bool(np.asarray(result["final_converged"]).all()) for result in results.values()
        ),
        "all_agreement_values_finite": bool(
            np.isfinite(agreement_values).all() and np.isfinite(baseline_agreement_values).all()
        ),
    }
    summary = {
        "method": "Integrated Gradients",
        "explainer_commit_sha": _git_sha(),
        "explainer_worktree_dirty_at_start": worktree_dirty_at_start,
        "explainer_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "model_commit_sha": json.loads((run_dir / "manifest.json").read_text())["commit_sha"],
        "target_rule": "predicted class from the frozen checkpoint",
        "ranking_rule": "absolute attribution magnitude; signed raw values retained",
        "integration_method": config["integration_method"],
        "initial_steps": int(config["n_steps"]),
        "retry_steps": int(config["retry_n_steps"]),
        "samples": len(test_images),
        "features": len(bundle.feature_names),
        "baseline_aliases": baseline_aliases,
        "baseline_diagnostics": baseline_diagnostics,
        "convergence": convergence_summary,
        "top_five_features_by_baseline_and_target_group": aggregate_summaries,
        "grad_cam_agreement": agreement_summaries,
        "baseline_sensitivity_agreement": baseline_agreement_summary,
        "checks": checks,
    }
    output_paths["summary"].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_examples_figure(
        output_paths["examples_figure"],
        images=test_images,
        attribution_maps=raw_attributions,
        feature_magnitudes=magnitude_feature_scores,
        feature_names=bundle.feature_names,
        sample_ids=bundle.test_indices,
        labels=bundle.y_test,
        predictions=frozen["predictions"],
    )
    _write_agreement_figure(output_paths["agreement_figure"], agreement_rows)
    return {
        "status": "ok" if all(checks.values()) else "quality_gate_failed",
        "run_dir": str(run_dir),
        "canonical_baselines": list(canonical_baselines),
        "baseline_aliases": baseline_aliases,
        "convergence": convergence_summary,
        "grad_cam_agreement": agreement_summaries,
        "baseline_sensitivity_agreement": baseline_agreement_summary,
        "checks": checks,
    }
