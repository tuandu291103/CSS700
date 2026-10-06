"""Paired magnitude-ranking agreement with duplicate-aware conditional uncertainty."""

import numpy as np

from css700.evaluation.local_agreement import paired_rank_rows

COMPARISONS = (
    "occlusion_zero_vs_grad_cam",
    "occlusion_median_vs_grad_cam",
    "occlusion_zero_vs_median",
)
METRICS = ("spearman", "top_5_overlap_fraction", "top_10_overlap_fraction")


def rank_records(scores, cam, predictions, labels, ids):
    rows = []
    for name, left, right in zip(
        COMPARISONS, (scores[0], scores[1], scores[0]), (cam, cam, scores[1]), strict=True
    ):
        rows.extend(
            paired_rank_rows(left, right, predictions, predictions, labels, ids, comparison=name)
        )
    return rows


def grouped_rank_summary(
    records, groups, labels, targets, *, resamples=1000, seed=20261006, minimum_groups=10
):
    masks = {
        "all": np.ones(len(labels), bool),
        "true_0": labels == 0,
        "true_1": labels == 1,
        "predicted_0": targets == 0,
        "predicted_1": targets == 1,
        "correct": targets == labels,
        "incorrect": targets != labels,
    }
    by_pair = {name: [r for r in records if r["comparison"] == name] for name in COMPARISONS}
    if any(len(rows) != len(groups) for rows in by_pair.values()):
        raise ValueError("one record per source row/comparison required")
    summaries, rng = [], np.random.default_rng(seed)
    for group, selected in masks.items():
        _, codes = np.unique(groups[selected], return_inverse=True)
        g = len(np.unique(codes))
        if not g:
            continue
        positions = rng.integers(g, size=(resamples, g))
        draws = np.array([np.bincount(row, minlength=g) for row in positions], dtype=np.float64)
        for comparison, pair in by_pair.items():
            rows = [r for r, keep in zip(pair, selected, strict=True) if keep]
            eligible = np.array([r["primary_eligible"] for r in rows])
            counts = np.bincount(codes[eligible], minlength=g)
            present = counts > 0
            for weighting in ("row_primary", "group_equal"):
                result = {
                    "comparison": comparison,
                    "group": group,
                    "weighting": weighting,
                    "total_rows": len(rows),
                    "total_feature_groups": g,
                    "eligible_rows": int(eligible.sum()),
                    "eligible_feature_groups": int(present.sum()),
                    "constant_vector_rows": sum(r["constant_vector"] for r in rows),
                    "cutoff_tie_rows_eligible": {
                        f"{side}_top_{k}": sum(
                            r[f"{side}_top_{k}_cutoff_tie"] for r in rows if r["primary_eligible"]
                        )
                        for side in ("left", "right")
                        for k in (5, 10)
                    },
                }
                for metric in METRICS:
                    values = np.array([r[metric] for r in rows if r["primary_eligible"]])
                    sums = np.bincount(codes[eligible], weights=values, minlength=g)
                    if weighting == "group_equal":
                        sums = np.divide(sums, counts, out=np.zeros(g), where=present)
                        denominator = present.astype(float)
                    else:
                        denominator = counts.astype(float)
                    mass = draws @ denominator
                    valid = mass > 0
                    distribution = (draws[valid] @ sums) / mass[valid]
                    result[metric] = {
                        "mean": float(sums.sum() / denominator.sum())
                        if denominator.sum()
                        else None,
                        "ci95": np.quantile(distribution, [0.025, 0.975]).tolist()
                        if present.sum() >= minimum_groups and valid.all()
                        else None,
                        "valid_bootstrap_draws": int(valid.sum()),
                    }
                summaries.append(result)
    return summaries


def plot_agreement(path, records):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, layout="constrained")
    for ax, name, title in zip(
        axes,
        COMPARISONS,
        ("Occlusion mean vs Grad-CAM", "Occlusion median vs Grad-CAM", "Occlusion mean vs median"),
        strict=True,
    ):
        pair = [r for r in records if r["comparison"] == name]
        values = [r["spearman"] for r in pair if r["primary_eligible"]]
        ax.hist(values, bins=np.linspace(-1, 1, 21), color="#267898", edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set(title=f"{title}\n{len(values)}/{len(pair)} eligible rows", xlabel="Spearman rho")
    axes[0].set_ylabel("Validation rows (not independent groups)")
    fig.suptitle("Spambase validation: absolute Occlusion effects vs positive Grad-CAM support")
    fig.savefig(path, dpi=150)
    plt.close(fig)
