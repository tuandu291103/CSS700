"""Metrics for cumulative feature-deletion faithfulness curves."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def normalized_area_over_curve(
    values: ArrayLike,
    k_values: ArrayLike,
    *,
    feature_count: int,
    axis: int = -1,
) -> NDArray[np.float64]:
    """Integrate a deletion-effect curve and normalize by the full feature count."""

    array = np.asarray(values, dtype=np.float64)
    steps = np.asarray(k_values, dtype=np.float64)
    if feature_count <= 0:
        raise ValueError("feature_count must be positive")
    if steps.ndim != 1 or len(steps) < 2:
        raise ValueError("k_values must contain at least two one-dimensional steps")
    if np.any(np.diff(steps) <= 0) or steps[0] != 0 or steps[-1] != feature_count:
        raise ValueError("k_values must increase from zero through the full feature count")
    if array.shape[axis] != len(steps):
        raise ValueError("the integration axis must match k_values")
    return np.asarray(np.trapezoid(array, x=steps, axis=axis) / feature_count)
