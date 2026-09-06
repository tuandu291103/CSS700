"""Validation-selected training for reproducible tabular tree baselines."""

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class TabularPredictionResult:
    """Ordered predictions, probabilities, and classification metrics."""

    probabilities: NDArray[np.float64]
    predictions: NDArray[np.int64]
    metrics: dict[str, Any]


@dataclass(frozen=True)
class TabularTrainingOutcome:
    """Best validation-selected estimator and its frozen evaluations."""

    model: Any
    search_rows: tuple[dict[str, Any], ...]
    selected_candidate_index: int
    selected_parameters: dict[str, Any]
    validation: TabularPredictionResult
    test: TabularPredictionResult
    majority_class: int
    majority_metrics: dict[str, Any]


def expand_search_grid(search_grid: dict[str, list[Any]]) -> tuple[dict[str, Any], ...]:
    """Expand a YAML-ordered parameter grid into deterministic candidate order."""

    if not search_grid:
        return ({},)
    keys = list(search_grid)
    values = [search_grid[key] for key in keys]
    if any(not isinstance(options, list) or not options for options in values):
        raise ValueError("every search-grid value must be a non-empty list")
    return tuple(dict(zip(keys, combination, strict=True)) for combination in product(*values))


def select_candidate_index(validation_scores: list[float]) -> int:
    """Select the first maximum validation score using a numerical tie tolerance."""

    if not validation_scores or not np.isfinite(validation_scores).all():
        raise ValueError("validation scores must be a non-empty finite sequence")
    best_index = 0
    best_score = float(validation_scores[0])
    for index, score in enumerate(validation_scores[1:], start=1):
        if float(score) > best_score + 1e-12:
            best_index = index
            best_score = float(score)
    return best_index


def _classification_metrics(
    labels: NDArray[np.int64],
    predictions: NDArray[np.int64],
    probabilities: NDArray[np.float64] | None,
    *,
    num_classes: int,
) -> dict[str, Any]:
    class_labels = list(range(num_classes))
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "macro_precision": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "per_class_precision": precision_score(
            labels, predictions, labels=class_labels, average=None, zero_division=0
        ).tolist(),
        "per_class_recall": recall_score(
            labels, predictions, labels=class_labels, average=None, zero_division=0
        ).tolist(),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=class_labels).tolist(),
    }
    if probabilities is not None:
        metrics["log_loss"] = float(log_loss(labels, probabilities, labels=class_labels))
        if num_classes == 2:
            metrics["roc_auc"] = float(roc_auc_score(labels, probabilities[:, 1]))
        else:
            metrics["roc_auc"] = float(
                roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
            )
    return metrics


def evaluate_tabular_estimator(
    model: Any,
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
    *,
    num_classes: int,
) -> TabularPredictionResult:
    """Evaluate an estimator while preserving input sample order."""

    model_classes = np.asarray(model.classes_, dtype=np.int64)
    expected_classes = np.arange(num_classes, dtype=np.int64)
    if not np.array_equal(model_classes, expected_classes):
        raise ValueError("model class order does not match the dataset class order")
    raw_probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    if raw_probabilities.shape != (len(labels), num_classes):
        raise ValueError("model returned an unexpected probability shape")
    probability_sums = raw_probabilities.sum(axis=1, keepdims=True)
    if not np.isfinite(raw_probabilities).all() or np.any(probability_sums <= 0.0):
        raise ValueError("model returned invalid probabilities")
    probabilities = raw_probabilities / probability_sums
    predictions = model_classes[np.argmax(probabilities, axis=1)].astype(np.int64, copy=False)
    return TabularPredictionResult(
        probabilities=probabilities,
        predictions=predictions,
        metrics=_classification_metrics(
            labels,
            predictions,
            probabilities,
            num_classes=num_classes,
        ),
    )


def _build_estimator(
    model_name: str,
    *,
    fixed_parameters: dict[str, Any],
    candidate_parameters: dict[str, Any],
    seed: int,
) -> Any:
    parameters = {**fixed_parameters, **candidate_parameters}
    parameters["random_state"] = seed
    parameters["n_jobs"] = 1
    if model_name == "random_forest":
        return RandomForestClassifier(**parameters)
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:  # pragma: no cover - core-only environments
            raise RuntimeError(
                "XGBoost is required; install the project research dependencies"
            ) from error
        return XGBClassifier(**parameters)
    raise ValueError(f"unsupported tabular baseline: {model_name}")


def train_tabular_baseline(
    model_name: str,
    train_features: NDArray[np.float64],
    train_labels: NDArray[np.int64],
    validation_features: NDArray[np.float64],
    validation_labels: NDArray[np.int64],
    test_features: NDArray[np.float64],
    test_labels: NDArray[np.int64],
    *,
    seed: int,
    fixed_parameters: dict[str, Any],
    search_grid: dict[str, list[Any]],
) -> TabularTrainingOutcome:
    """Select on validation macro F1 and evaluate the untouched test split once."""

    candidates = expand_search_grid(search_grid)
    num_classes = int(np.unique(train_labels).size)
    models: list[Any] = []
    validation_results: list[TabularPredictionResult] = []
    rows: list[dict[str, Any]] = []
    for candidate_index, candidate_parameters in enumerate(candidates):
        model = _build_estimator(
            model_name,
            fixed_parameters=fixed_parameters,
            candidate_parameters=candidate_parameters,
            seed=seed,
        )
        model.fit(train_features, train_labels)
        validation = evaluate_tabular_estimator(
            model,
            validation_features,
            validation_labels,
            num_classes=num_classes,
        )
        models.append(model)
        validation_results.append(validation)
        rows.append(
            {
                "candidate_index": candidate_index,
                "parameters": candidate_parameters,
                "validation_accuracy": validation.metrics["accuracy"],
                "validation_macro_f1": validation.metrics["macro_f1"],
                "validation_log_loss": validation.metrics["log_loss"],
                "validation_roc_auc": validation.metrics["roc_auc"],
            }
        )

    selected_index = select_candidate_index(
        [float(result.metrics["macro_f1"]) for result in validation_results]
    )
    selected_model = models[selected_index]
    selected_validation = validation_results[selected_index]
    search_rows = tuple(
        {**row, "selected": int(row["candidate_index"] == selected_index)} for row in rows
    )
    test = evaluate_tabular_estimator(
        selected_model,
        test_features,
        test_labels,
        num_classes=num_classes,
    )
    majority_class = int(np.bincount(train_labels).argmax())
    majority_predictions = np.full_like(test_labels, majority_class)
    majority_metrics = _classification_metrics(
        test_labels,
        majority_predictions,
        None,
        num_classes=num_classes,
    )
    return TabularTrainingOutcome(
        model=selected_model,
        search_rows=search_rows,
        selected_candidate_index=selected_index,
        selected_parameters={**fixed_parameters, **candidates[selected_index]},
        validation=selected_validation,
        test=test,
        majority_class=majority_class,
        majority_metrics=majority_metrics,
    )
