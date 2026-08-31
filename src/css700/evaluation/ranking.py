"""Ranking agreement metrics used by the proposal."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import spearmanr


def spearman_agreement(left: ArrayLike, right: ArrayLike) -> float:
    """Return Spearman correlation for two equal-length feature-score vectors."""

    left_array = np.asarray(left, dtype=np.float64).ravel()
    right_array = np.asarray(right, dtype=np.float64).ravel()
    if left_array.shape != right_array.shape or left_array.size < 2:
        raise ValueError("score vectors must have the same shape and contain at least two values")
    result = spearmanr(left_array, right_array)
    return float(result.statistic)


def top_k_overlap(left: ArrayLike, right: ArrayLike, *, k: int) -> dict[str, float | int]:
    """Return intersection count, overlap fraction, and Jaccard similarity for top-k sets."""

    left_array = np.asarray(left, dtype=np.float64).ravel()
    right_array = np.asarray(right, dtype=np.float64).ravel()
    if left_array.shape != right_array.shape:
        raise ValueError("score vectors must have the same shape")
    if k <= 0 or k > left_array.size:
        raise ValueError("k must be between 1 and the number of features")

    left_top = set(np.argsort(-left_array, kind="stable")[:k].tolist())
    right_top = set(np.argsort(-right_array, kind="stable")[:k].tolist())
    intersection = len(left_top & right_top)
    union = len(left_top | right_top)
    return {
        "k": k,
        "intersection": intersection,
        "overlap_fraction": intersection / k,
        "jaccard": intersection / union,
    }
