"""Trace image-space attribution tensors through a documented feature mapping."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from css700.encoding.grid import FeaturePixel


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

    values = np.empty((array.shape[0], len(mapping)), dtype=np.float64)
    for position, pixel in enumerate(mapping):
        outside_tensor = (
            pixel.channel >= array.shape[1]
            or pixel.row >= array.shape[2]
            or pixel.column >= array.shape[3]
        )
        if outside_tensor:
            raise ValueError(
                f"mapping for {pixel.feature_name!r} is outside the attribution tensor"
            )
        values[:, position] = array[:, pixel.channel, pixel.row, pixel.column]
    return np.abs(values) if absolute else values
