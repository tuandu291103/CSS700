"""Breast Cancer Wisconsin pilot data with deterministic, training-only preprocessing."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DataBundle:
    """Preprocessed arrays plus stable indices needed to reproduce every comparison."""

    x_train: NDArray[np.float64]
    x_validation: NDArray[np.float64]
    x_test: NDArray[np.float64]
    y_train: NDArray[np.int64]
    y_validation: NDArray[np.int64]
    y_test: NDArray[np.int64]
    train_indices: NDArray[np.int64]
    validation_indices: NDArray[np.int64]
    test_indices: NDArray[np.int64]
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    scaler_mean: NDArray[np.float64]
    scaler_scale: NDArray[np.float64]


def load_breast_cancer_splits(
    *,
    seed: int = 42,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    stratify: bool = True,
) -> DataBundle:
    """Load the pilot dataset and fit the StandardScaler on training rows only.

    ``test_size`` and ``validation_size`` are proportions of the full dataset.
    """

    if test_size <= 0 or validation_size <= 0 or test_size + validation_size >= 1:
        raise ValueError("test_size and validation_size must be positive and sum to less than 1")

    dataset = load_breast_cancer()
    x = np.asarray(dataset.data, dtype=np.float64)
    y = np.asarray(dataset.target, dtype=np.int64)
    indices = np.arange(x.shape[0], dtype=np.int64)
    stratify_y = y if stratify else None

    train_validation_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_y,
    )

    validation_fraction_of_remainder = validation_size / (1.0 - test_size)
    remaining_y = y[train_validation_idx] if stratify else None
    train_idx, validation_idx = train_test_split(
        train_validation_idx,
        test_size=validation_fraction_of_remainder,
        random_state=seed,
        stratify=remaining_y,
    )

    scaler = StandardScaler().fit(x[train_idx])
    x_train = scaler.transform(x[train_idx])
    x_validation = scaler.transform(x[validation_idx])
    x_test = scaler.transform(x[test_idx])

    return DataBundle(
        x_train=x_train,
        x_validation=x_validation,
        x_test=x_test,
        y_train=y[train_idx],
        y_validation=y[validation_idx],
        y_test=y[test_idx],
        train_indices=train_idx,
        validation_indices=validation_idx,
        test_indices=test_idx,
        feature_names=tuple(str(name) for name in dataset.feature_names),
        target_names=tuple(str(name) for name in dataset.target_names),
        scaler_mean=np.asarray(scaler.mean_),
        scaler_scale=np.asarray(scaler.scale_),
    )
