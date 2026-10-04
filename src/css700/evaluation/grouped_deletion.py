"""Signed deletion metrics and paired whole-feature-group bootstrap estimates."""

import numpy as np


def softmax64(logits):
    x = np.asarray(logits, dtype=np.float64)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def deletion_effects(logits, original, targets, steps):
    index = targets[None, None, None, :, None]
    target_logits = np.take_along_axis(logits.astype(np.float64), index, axis=-1)[..., 0]
    target_probs = np.take_along_axis(softmax64(logits), index, axis=-1)[..., 0]
    base_logits = original[np.arange(len(targets)), targets].astype(np.float64)
    base_probs = softmax64(original)[np.arange(len(targets)), targets]
    drops = {"logit": base_logits - target_logits, "probability": base_probs - target_probs}
    aopc = {k: np.trapezoid(v, x=np.asarray(steps), axis=2) / steps[-1] for k, v in drops.items()}
    return drops, aopc


def _cluster_estimate(values, codes, counts, draws, weighting):
    sums = np.bincount(codes, weights=values, minlength=len(counts))
    if weighting == "row_primary":
        point = values.mean()
        distribution = (draws @ sums) / (draws @ counts)
    else:
        means = sums / counts
        point = means.mean()
        distribution = (draws @ means) / len(counts)
    return float(point), distribution


def grouped_summaries(
    logits,
    original,
    targets,
    labels,
    groups,
    steps,
    *,
    resamples=1000,
    seed=20261005,
    minimum_groups=10,
):
    drops, aopc = deletion_effects(logits, original, targets, steps)
    predictions = logits.argmax(-1)
    masks = {
        "all": np.ones(len(labels), bool),
        "true_0": labels == 0,
        "true_1": labels == 1,
        "predicted_0": targets == 0,
        "predicted_1": targets == 1,
        "correct": targets == labels,
        "incorrect": targets != labels,
    }
    rng, curves, summaries = np.random.default_rng(seed), [], []
    for group_name, selected in masks.items():
        _, codes, counts = np.unique(groups[selected], return_inverse=True, return_counts=True)
        g = len(counts)
        if not g:
            continue
        positions = rng.integers(g, size=(resamples, g))
        draws = np.zeros((resamples, g), dtype=np.float64)
        for i, row in enumerate(positions):
            draws[i] = np.bincount(row, minlength=g)
        for weighting in ("row_primary", "group_equal"):
            w = np.ones(selected.sum()) if weighting == "row_primary" else 1 / counts[codes]
            for r, reference in enumerate(("zero", "training_median")):
                for metric, values in aopc.items():
                    vectors = {
                        "top": values[r, 0, selected],
                        "bottom": values[r, 1, selected],
                        "random": values[r, 2:, :][:, selected].mean(0),
                    }
                    vectors["top_minus_random"] = vectors["top"] - vectors["random"]
                    vectors["top_minus_bottom"] = vectors["top"] - vectors["bottom"]
                    for order, vector in vectors.items():
                        point, distribution = _cluster_estimate(
                            vector, codes, counts, draws, weighting
                        )
                        interval = (
                            np.quantile(distribution, [0.025, 0.975]).tolist()
                            if g >= minimum_groups
                            else None
                        )
                        summaries.append(
                            {
                                "group": group_name,
                                "rows": int(selected.sum()),
                                "feature_groups": g,
                                "weighting": weighting,
                                "replacement": reference,
                                "metric": metric,
                                "order_or_contrast": order,
                                "mean": point,
                                "ci95": interval,
                            }
                        )
                for order, indices in (
                    ("top", slice(0, 1)),
                    ("bottom", slice(1, 2)),
                    ("random", slice(2, None)),
                ):
                    for j, k in enumerate(steps):
                        pred = predictions[r, indices, j][:, selected]
                        y = labels[selected][None]
                        f1 = []
                        for c in (0, 1):
                            tp = ((pred == c) & (y == c)) @ w
                            fp = ((pred == c) & (y != c)) @ w
                            fn = ((pred != c) & (y == c)) @ w
                            denominator = 2 * tp + fp + fn
                            f1.append(
                                np.divide(
                                    2 * tp,
                                    denominator,
                                    out=np.zeros_like(tp),
                                    where=denominator != 0,
                                )
                            )
                        curves.append(
                            {
                                "group": group_name,
                                "weighting": weighting,
                                "replacement": reference,
                                "order": order,
                                "deleted_features": int(k),
                                "mean_logit_drop": float(
                                    np.average(
                                        drops["logit"][r, indices, j][:, selected],
                                        axis=-1,
                                        weights=w,
                                    ).mean()
                                ),
                                "mean_probability_drop": float(
                                    np.average(
                                        drops["probability"][r, indices, j][:, selected],
                                        axis=-1,
                                        weights=w,
                                    ).mean()
                                ),
                                "flip_rate": float(
                                    np.average(pred != targets[selected], axis=-1, weights=w).mean()
                                ),
                                "accuracy": float(np.average(pred == y, axis=-1, weights=w).mean()),
                                "macro_f1": float(np.mean(f1)),
                            }
                        )
    return curves, summaries, aopc


def plot_deletion_curves(path, curves, *, rows, random_repeats):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, layout="constrained")
    for ax, reference in zip(axes, ("zero", "training_median"), strict=True):
        for order, color in (("top", "#b13e41"), ("random", "#737373"), ("bottom", "#267898")):
            points = [
                r
                for r in curves
                if r["group"] == "all"
                and r["weighting"] == "row_primary"
                and r["replacement"] == reference
                and r["order"] == order
            ]
            ax.plot(
                [r["deleted_features"] for r in points],
                [r["mean_logit_drop"] for r in points],
                label=order,
                color=color,
                marker="o",
                markersize=3,
            )
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set(
            title=reference.replace("_", " "), xlabel="Named features replaced (padding excluded)"
        )
        ax.legend(frameon=False)
    axes[0].set_ylabel("Mean frozen-target logit drop (signed)")
    fig.suptitle(
        "Spambase validation only: Grad-CAM deletion faithfulness\n"
        f"{rows} rows; fixed model; random = mean of {random_repeats} group-shared orders"
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
