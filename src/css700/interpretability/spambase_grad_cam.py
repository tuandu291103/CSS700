"""Validation Grad-CAM execution and explicit non-feature-padding diagnostics."""

import time

import numpy as np
import torch

from css700.encoding.grid import FeaturePixel
from css700.interpretability.grad_cam import grad_cam
from css700.interpretability.tracing import trace_attributions_to_features
from css700.models import SmallTabularImageCNN


def explain_validation(
    source,
    *,
    batch_size=32,
    max_seconds=120.0,
    gradient_logit_rtol=1e-5,
    gradient_logit_atol=1e-6,
):
    """Use frozen predicted-logit targets without updating weights or accumulating grads."""
    if batch_size <= 0 or max_seconds <= 0:
        raise ValueError("positive explanation budget required")
    old_threads = torch.get_num_threads()
    old_determinism = torch.are_deterministic_algorithms_enabled()
    old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    started = time.perf_counter()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            model = SmallTabularImageCNN().cpu()
            model.load_state_dict(source.checkpoint["model_state_dict"])
            model.eval()
            before = {k: v.clone() for k, v in model.state_dict().items()}
            hook_count = len(model.grad_cam_layer._forward_hooks)
            images = source.arrays["validation_images"]
            expected = source.predictions
            parts = {
                k: []
                for k in (
                    "raw_maps",
                    "normalized_maps",
                    "target_classes",
                    "logits",
                    "inference_logits",
                )
            }
            for start in range(0, len(images), batch_size):
                if time.perf_counter() - started > max_seconds:
                    raise TimeoutError("fixed Grad-CAM wall-time budget exceeded")
                inputs = torch.from_numpy(images[start : start + batch_size].copy())
                with torch.no_grad():
                    inference_logits = model(inputs)
                result = grad_cam(
                    model,
                    inputs,
                    target_layer=model.features[2],
                    target_classes=torch.from_numpy(
                        expected["predictions"][start : start + batch_size].copy()
                    ),
                )
                for key, tensor in (
                    ("raw_maps", result.raw),
                    ("normalized_maps", result.normalized),
                    ("target_classes", result.target_classes),
                    ("logits", result.logits),
                    ("inference_logits", inference_logits),
                ):
                    parts[key].append(tensor.numpy())
            if time.perf_counter() - started > max_seconds:
                raise TimeoutError("fixed Grad-CAM wall-time budget exceeded")
            outputs = {k: np.concatenate(v) for k, v in parts.items()}
            mapping = [FeaturePixel(**record) for record in source.feature_map]
            outputs["feature_scores"] = trace_attributions_to_features(outputs["raw_maps"], mapping)
            outputs["normalized_feature_scores"] = trace_attributions_to_features(
                outputs["normalized_maps"], mapping
            )
            probe = outputs["raw_maps"].copy()
            probe[:, ~source.arrays["occupied_mask"]] = 1e6
            checks = {
                "all_outputs_finite": all(np.isfinite(v).all() for v in outputs.values()),
                "raw_maps_input_sized": outputs["raw_maps"].shape == images.shape,
                "all_57_features_traced": outputs["feature_scores"].shape == (len(images), 57),
                "nonnegative_raw_maps": bool(np.all(outputs["raw_maps"] >= 0)),
                "display_maps_bounded": bool(
                    np.all(outputs["normalized_maps"] >= 0)
                    and np.all(outputs["normalized_maps"] <= 1)
                ),
                "frozen_inference_logits_exact": bool(
                    np.array_equal(outputs["inference_logits"], expected["logits"])
                ),
                "gradient_logits_numerically_match": bool(
                    np.allclose(
                        outputs["logits"],
                        expected["logits"],
                        rtol=gradient_logit_rtol,
                        atol=gradient_logit_atol,
                    )
                ),
                "gradient_predicted_classes_exact": bool(
                    np.array_equal(outputs["logits"].argmax(axis=1), expected["predictions"])
                ),
                "frozen_predicted_targets_exact": bool(
                    np.array_equal(outputs["target_classes"], expected["predictions"])
                ),
                "feature_values_exact": bool(
                    np.array_equal(
                        outputs["feature_scores"],
                        outputs["raw_maps"][:, source.arrays["occupied_mask"]],
                    )
                ),
                "padding_excluded_even_with_permuted_mapping": bool(
                    np.array_equal(
                        trace_attributions_to_features(probe, list(reversed(mapping))),
                        outputs["feature_scores"],
                    )
                ),
                "weights_unchanged": all(
                    torch.equal(before[k], v) for k, v in model.state_dict().items()
                ),
                "no_parameter_gradients_accumulated": all(
                    p.grad is None for p in model.parameters()
                ),
                "hooks_removed": len(model.grad_cam_layer._forward_hooks) == hook_count,
            }
            return outputs, checks
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_determinism, warn_only=old_warn)


def map_diagnostics(outputs, source):
    """Raw-score summaries; zero/tied maps are retained, never silently filtered."""
    raw, scores = outputs["raw_maps"], outputs["feature_scores"]
    mask = source.arrays["occupied_mask"]
    p = source.predictions
    names = [
        r["feature_name"] for r in sorted(source.feature_map, key=lambda r: r["feature_index"])
    ]
    totals = raw.sum(axis=(1, 2, 3), dtype=np.float64)
    padding = raw[:, ~mask].sum(axis=1, dtype=np.float64)
    feature_totals = scores.sum(axis=1)
    if not np.allclose(totals, padding + feature_totals, rtol=1e-12, atol=1e-15):
        raise ValueError("feature/padding mass partition mismatch")
    order = np.argsort(-scores, axis=1, kind="stable")
    records = []
    for j, row_id in enumerate(p["sample_ids"]):
        positive = [int(k) for k in order[j] if scores[j, k] > 0]
        records.append(
            {
                "sample_id": int(row_id),
                "true_label": int(p["labels"][j]),
                "predicted_class": int(p["predictions"][j]),
                "target_class": int(outputs["target_classes"][j]),
                "correct": bool(p["labels"][j] == p["predictions"][j]),
                "raw_map_mass": float(totals[j]),
                "feature_mass": float(feature_totals[j]),
                "padding_mass": float(padding[j]),
                "padding_fraction": float(padding[j] / totals[j]) if totals[j] > 0 else None,
                "zero_raw_map": bool(totals[j] == 0),
                "zero_feature_map": not positive,
                "positive_feature_count": len(positive),
                "positive_top_5": [
                    {"feature_index": k, "feature_name": names[k], "raw_score": float(scores[j, k])}
                    for k in positive[:5]
                ],
                "positive_top_10": [
                    {"feature_index": k, "feature_name": names[k], "raw_score": float(scores[j, k])}
                    for k in positive[:10]
                ],
                "positive_tie_at_5": bool(
                    len(positive) > 5 and scores[j, positive[4]] == scores[j, positive[5]]
                ),
                "positive_tie_at_10": bool(
                    len(positive) > 10 and scores[j, positive[9]] == scores[j, positive[10]]
                ),
            }
        )
    aggregate = {}
    groups = {
        "all_predicted_classes": np.ones(len(scores), dtype=bool),
        **{f"predicted_class_{c}": p["predictions"] == c for c in (0, 1)},
    }
    weights = p["group_equal_weights"]
    for name, selected in groups.items():
        if not selected.any():
            aggregate[name] = {"rows": 0, "row_primary": [], "group_equal": []}
            continue
        rankings = {}
        for weighting, w in (
            ("row_primary", np.ones(selected.sum())),
            ("group_equal", weights[selected]),
        ):
            means = np.average(scores[selected], axis=0, weights=w)
            rankings[weighting] = [
                {
                    "rank": rank,
                    "feature_index": int(k),
                    "feature_name": names[k],
                    "mean_raw_score": float(means[k]),
                }
                for rank, k in enumerate(np.argsort(-means, kind="stable"), 1)
            ]
        aggregate[name] = {"rows": int(selected.sum()), **rankings}
    fractions = [r["padding_fraction"] for r in records if r["padding_fraction"] is not None]
    summary = {
        "rows": len(records),
        "zero_raw_maps": sum(r["zero_raw_map"] for r in records),
        "zero_feature_maps": sum(r["zero_feature_map"] for r in records),
        "rows_with_positive_padding": int(np.sum(padding > 0)),
        "padding_fraction_nonzero_maps": {
            "defined_rows": len(fractions),
            "mean": float(np.mean(fractions)) if fractions else None,
            "median": float(np.median(fractions)) if fractions else None,
            "max": float(np.max(fractions)) if fractions else None,
        },
        "rows_with_fewer_than_5_positive_features": sum(
            r["positive_feature_count"] < 5 for r in records
        ),
        "rows_with_fewer_than_10_positive_features": sum(
            r["positive_feature_count"] < 10 for r in records
        ),
        "positive_top_5_boundary_ties": sum(r["positive_tie_at_5"] for r in records),
        "positive_top_10_boundary_ties": sum(r["positive_tie_at_10"] for r in records),
    }
    return records, aggregate, summary


def selected_examples(predictions):
    """First source-ordered validation row for each true/predicted class pair."""
    positions = []
    for true, predicted in ((0, 0), (1, 1), (0, 1), (1, 0)):
        matches = np.flatnonzero(
            (predictions["labels"] == true) & (predictions["predictions"] == predicted)
        )
        if len(matches):
            positions.append(int(matches[0]))
    return positions


def write_examples(path, source, outputs, records):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    positions = selected_examples(source.predictions)
    figure, axes = plt.subplots(
        len(positions), 2, figsize=(12, 3.1 * len(positions)), squeeze=False, layout="constrained"
    )
    palette = plt.get_cmap("inferno").with_extremes(bad="lightgray")
    for row, j in enumerate(positions):
        record = records[j]
        shown = np.ma.masked_array(
            outputs["normalized_maps"][j, 0], mask=~source.arrays["occupied_mask"][0]
        )
        heatmap = axes[row, 0].imshow(shown, cmap=palette, vmin=0, vmax=1)
        share = record["padding_fraction"]
        axes[row, 0].set_title(
            (
                f"Row {record['sample_id']}: true {record['true_label']}, "
                f"predicted {record['predicted_class']}\nPadding CAM share: {share:.1%}"
            )
            if share is not None
            else f"Row {record['sample_id']}: zero map"
        )
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        figure.colorbar(heatmap, ax=axes[row, 0], shrink=0.8, label="Display-normalized CAM")
        top = record["positive_top_5"][::-1]
        if top:
            axes[row, 1].barh([v["feature_name"] for v in top], [v["raw_score"] for v in top])
        else:
            axes[row, 1].text(
                0.5,
                0.5,
                "No positive feature scores",
                ha="center",
                transform=axes[row, 1].transAxes,
            )
        axes[row, 1].set(
            title="Up to five positive traced features",
            xlabel="Raw Grad-CAM score (not causal importance)",
        )
    figure.suptitle(
        "Spambase validation only — seed 42, predicted-class Grad-CAM\n"
        "0: non-spam; 1: spam. Gray cells are padding, excluded from feature rankings.",
        fontsize=12,
    )
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return [int(source.predictions["sample_ids"][j]) for j in positions]
