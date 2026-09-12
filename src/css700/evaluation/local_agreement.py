"""Paired descriptive rank comparisons with explicit exclusions and tie diagnostics."""

from typing import Any

import numpy as np

from css700.evaluation.ranking import spearman_agreement, top_k_overlap


def paired_rank_rows(
    left: np.ndarray,
    right: np.ndarray,
    left_predictions: np.ndarray,
    right_predictions: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    *,
    comparison: str,
) -> list[dict[str, Any]]:
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] != len(sample_ids):
        raise ValueError("rank comparison shapes mismatch")
    if left.shape[1] < 10 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("rank comparison needs finite scores for at least ten features")
    rows = []
    for i, sample_id in enumerate(sample_ids):
        a, b = np.abs(left[i]), np.abs(right[i])
        same = bool(left_predictions[i] == right_predictions[i])
        constant = bool(np.ptp(a) == 0 or np.ptp(b) == 0)
        correct_count = int(left_predictions[i] == labels[i]) + int(
            right_predictions[i] == labels[i]
        )
        row = {
            "comparison": comparison,
            "sample_id": int(sample_id),
            "true_label": int(labels[i]),
            "left_prediction": int(left_predictions[i]),
            "right_prediction": int(right_predictions[i]),
            "same_target": same,
            "constant_vector": constant,
            "primary_eligible": same and not constant,
            "exclusion_reason": "|".join(
                reason
                for reason, flag in (("different_target", not same), ("constant_vector", constant))
                if flag
            ),
            "correctness_group": ("neither_correct", "one_correct", "both_correct")[correct_count],
            "spearman": None if constant else spearman_agreement(a, b),
        }
        for k in (5, 10):
            for side, scores in (("left", a), ("right", b)):
                ordered = np.sort(scores)[::-1]
                row[f"{side}_top_{k}_cutoff_tie"] = (
                    bool(ordered[k - 1] == ordered[k]) if k < len(scores) else False
                )
            values = top_k_overlap(a, b, k=k) if not constant else {}
            for metric in ("intersection", "overlap_fraction", "jaccard"):
                row[f"top_{k}_{metric}"] = values.get(metric)
        rows.append(row)
    return rows


def summarize_rank_rows(
    rows: list[dict], *, resamples: int, seed: int, min_group: int = 10
) -> list[dict]:
    summaries = []
    for comparison in sorted({row["comparison"] for row in rows}):
        pair = [row for row in rows if row["comparison"] == comparison]
        groups = {"all": pair}
        for target in (0, 1):
            groups[f"left_predicted_class_{target}"] = [
                r for r in pair if r["left_prediction"] == target
            ]
        for group in ("both_correct", "one_correct", "neither_correct"):
            groups[group] = [r for r in pair if r["correctness_group"] == group]
        for group, members in groups.items():
            eligible = [r for r in members if r["primary_eligible"]]
            summary = {
                "comparison": comparison,
                "group": group,
                "total_samples": len(members),
                "eligible_samples": len(eligible),
                "different_target_samples": sum(not r["same_target"] for r in members),
                "constant_vector_samples": sum(r["constant_vector"] for r in members),
            }
            draws = (
                np.random.default_rng(seed).integers(
                    0, len(eligible), size=(resamples, len(eligible))
                )
                if len(eligible) >= min_group
                else None
            )
            for metric in (
                "spearman",
                "top_5_overlap_fraction",
                "top_5_jaccard",
                "top_10_overlap_fraction",
                "top_10_jaccard",
            ):
                values = np.array([r[metric] for r in eligible], dtype=float)
                summary[metric] = {
                    "mean": float(values.mean()) if len(values) else None,
                    "median": float(np.median(values)) if len(values) else None,
                    "quartiles": np.quantile(values, [0.25, 0.75]).tolist()
                    if len(values)
                    else None,
                    "mean_95_percentile_interval": np.quantile(
                        values[draws].mean(axis=1), [0.025, 0.975]
                    ).tolist()
                    if draws is not None
                    else None,
                }
            summary["cutoff_tie_rows"] = {
                f"{side}_top_{k}": sum(r[f"{side}_top_{k}_cutoff_tie"] for r in eligible)
                for side in ("left", "right")
                for k in (5, 10)
            }
            summaries.append(summary)
    return summaries
