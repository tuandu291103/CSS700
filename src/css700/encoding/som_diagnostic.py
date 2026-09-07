"""Provisional native-BMU diagnostic, NOT an authoritative SOM2IMG encoder."""

from dataclasses import dataclass
from math import ceil, sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray

from css700.encoding.grid import FeaturePixel


@dataclass(frozen=True)
class SomDiagnostic:
    mapping: tuple[FeaturePixel, ...]
    weights: NDArray[np.float64]
    quantization_error: float


def fit_som_diagnostic(
    training_values: ArrayLike,
    feature_names: tuple[str, ...] | list[str],
    *,
    layout_seed: int,
    sigma: float,
    learning_rate: float,
    update_steps: int,
) -> SomDiagnostic:
    """Fit only transposed training features; do not render colliding values.

    Settings are investigator-chosen diagnostic choices, not paper defaults. Uses
    rectangular Euclidean BMUs, Gaussian neighborhood, asymptotic decay, random
    training-feature initialization and a shuffled sequence of update steps.
    No labels, validation values, or test values are inputs to this function.
    """

    from minisom import MiniSom

    values = np.asarray(training_values, dtype=np.float64)
    names = tuple(feature_names)
    if values.ndim != 2 or min(values.shape) == 0 or values.shape[1] != len(names):
        raise ValueError("training_values must have shape [training samples, named features]")
    if not np.isfinite(values).all():
        raise ValueError("training_values must be finite")
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(
        names
    ):
        raise ValueError("feature names must be nonempty and unique")
    if not np.isfinite(sigma) or sigma <= 0 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("sigma and learning_rate must be finite and positive")
    if isinstance(update_steps, bool) or not isinstance(update_steps, int) or update_steps <= 0:
        raise ValueError("update_steps must be a positive integer")
    rows = ceil(sqrt(len(names)))
    columns = ceil(len(names) / rows)
    feature_vectors = np.ascontiguousarray(values.T)
    som = MiniSom(
        rows,
        columns,
        input_len=len(values),
        sigma=sigma,
        learning_rate=learning_rate,
        decay_function="asymptotic_decay",
        sigma_decay_function="asymptotic_decay",
        neighborhood_function="gaussian",
        topology="rectangular",
        activation_distance="euclidean",
        random_seed=layout_seed,
    )
    som.random_weights_init(feature_vectors)
    som.train(feature_vectors, update_steps, random_order=True, use_epochs=False)
    mapping = []
    for index, (name, vector) in enumerate(zip(names, feature_vectors, strict=True)):
        row, column = som.winner(vector)
        mapping.append(
            FeaturePixel(
                index, name, int(row), int(column), mapping_version="som-bmu-diagnostic-v1"
            )
        )
    return SomDiagnostic(
        tuple(mapping), som.get_weights().copy(), float(som.quantization_error(feature_vectors))
    )
