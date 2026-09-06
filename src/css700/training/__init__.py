"""Reproducible model training utilities with optional-stack lazy imports."""

from importlib import import_module
from typing import Any

__all__ = [
    "TabularPredictionResult",
    "TabularTrainingOutcome",
    "TrainingOutcome",
    "evaluate_classifier",
    "evaluate_tabular_estimator",
    "expand_search_grid",
    "select_candidate_index",
    "train_grid_cnn",
    "train_tabular_baseline",
]

_TABULAR_EXPORTS = {
    "TabularPredictionResult",
    "TabularTrainingOutcome",
    "evaluate_tabular_estimator",
    "expand_search_grid",
    "select_candidate_index",
    "train_tabular_baseline",
}
_GRID_EXPORTS = {"TrainingOutcome", "evaluate_classifier", "train_grid_cnn"}


def __getattr__(name: str) -> Any:
    """Load the tabular or PyTorch training stack only when requested."""

    if name in _TABULAR_EXPORTS:
        return getattr(import_module("css700.training.tabular"), name)
    if name in _GRID_EXPORTS:
        return getattr(import_module("css700.training.grid_cnn"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
