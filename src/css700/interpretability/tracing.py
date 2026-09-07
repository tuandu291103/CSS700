"""Trace image-space attribution tensors through a documented feature mapping."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from css700.encoding.grid import FeaturePixel
from css700.encoding.mapping import audit_feature_mapping


def trace_attributions_to_features(
    attributions: ArrayLike,
    mapping: tuple[FeaturePixel, ...] | list[FeaturePixel],
    *,
    absolute: bool = False,
) -> NDArray[np.float64]:
    """Return ``[samples, features]`` values from ``[samples, channels, rows, cols]``.

    Signed values are preserved by default. ``absolute=True`` is an explicit ranking
    transformation and must not replace retained raw attribution values in experiment artifacts.
    """

    array = np.asarray(attributions, dtype=np.float64)
    if array.ndim == 3:
        array = array[:, np.newaxis, :, :]
    if array.ndim != 4:
        raise ValueError("attributions must have shape [samples, channels, rows, columns]")

    audit = audit_feature_mapping(mapping, tuple(array.shape[1:]))
    if not audit["one_to_one"]:
        raise ValueError("pixel collisions require an explicit attribution-allocation policy")
    values = np.empty((array.shape[0], len(mapping)), dtype=np.float64)
    for pixel in mapping:
        values[:, pixel.feature_index] = array[:, pixel.channel, pixel.row, pixel.column]
    return np.abs(values) if absolute else values
