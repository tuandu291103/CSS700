"""Auditable quartile-membership LIME; surrogate fit is not SHAP additivity."""

from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from css700.interpretability.tree_shap import binary_probabilities, training_background


@dataclass(frozen=True)
class Neighborhood:
    design: np.ndarray
    inverse: np.ndarray
    distances: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class LimeDesign:
    fit: Neighborhood
    evaluation: Neighborhood
    metadata: dict


def derived_seed(explainer_seed: int, sample_id: int, role: str) -> int:
    """Stable per-ID streams; independent of traversal order and model family."""
    if role not in ("fit", "evaluation") or any(
        type(v) is not int or v < 0 for v in (explainer_seed, sample_id)
    ):
        raise ValueError("invalid LIME seed, sample ID or role")
    return int(
        np.random.SeedSequence(
            [explainer_seed, sample_id, 0 if role == "fit" else 1]
        ).generate_state(1, dtype=np.uint32)[0]
    )


def make_explainer(training: np.ndarray, names: tuple[str, ...], seed: int) -> Any:
    # The audited sampler is private: fail closed outside the inspected release.
    if version("lime") != "0.2.0.1":
        raise ValueError("LIME adapter requires inspected release 0.2.0.1")
    from lime.lime_tabular import LimeTabularExplainer

    return LimeTabularExplainer(
        training,
        mode="classification",
        training_labels=None,
        feature_names=list(names),
        class_names=["malignant", "benign"],
        categorical_features=[],
        discretize_continuous=True,
        discretizer="quartile",
        sample_around_instance=False,
        feature_selection="none",
        kernel_width=0.75 * np.sqrt(len(names)),
        random_state=seed,
    )


def build_lime_design(
    training: np.ndarray,
    training_ids: np.ndarray,
    heldout_ids: np.ndarray,
    row: np.ndarray,
    names: tuple[str, ...],
    *,
    sample_id: int,
    explainer_seed: int,
    fit_samples: int = 5000,
    evaluation_samples: int = 2000,
) -> LimeDesign:
    training, ids = training_background(training, training_ids, heldout_ids)
    row = np.asarray(row, dtype=np.float64)
    if (
        len(names) != training.shape[1]
        or len(set(names)) != len(names)
        or any(not isinstance(n, str) or not n for n in names)
        or row.shape != (len(names),)
        or not np.isfinite(row).all()
        or np.any(np.ptp(training, axis=0) == 0)
        or any(type(n) is not int or n < 2 for n in (fit_samples, evaluation_samples))
    ):
        raise ValueError(
            "invalid LIME features, row or budget; constant training columns unsupported"
        )
    seeds = {role: derived_seed(explainer_seed, sample_id, role) for role in ("fit", "evaluation")}
    neighborhoods = {}
    for role, count in (("fit", fit_samples), ("evaluation", evaluation_samples + 1)):
        explainer = make_explainer(training, names, seeds[role])
        # Use the native sampler unchanged and retain its actual model inputs.
        design, inverse = explainer._LimeTabularExplainer__data_inverse(row, count)
        reconstructed = (
            explainer.discretizer.discretize(inverse) == explainer.discretizer.discretize(row)
        ).astype(float)
        if (
            design.shape != (count, len(names))
            or inverse.shape != design.shape
            or not np.isfinite(inverse).all()
            or not np.array_equal(design, reconstructed)
            or not np.array_equal(inverse[0], row)
            or not np.all(design[0] == 1)
            or not np.all(explainer.scaler.mean_ == 0)
            or not np.all(explainer.scaler.scale_ == 1)
        ):
            raise ValueError("native LIME neighborhood/feature representation mismatch")
        if role == "evaluation":
            design, inverse = design[1:], inverse[1:]  # No forced anchor in fit evaluation.
        distances = np.linalg.norm(design - 1, axis=1)
        weights = np.exp(-(distances**2) / (2 * (0.75 * np.sqrt(len(names))) ** 2))
        neighborhoods[role] = Neighborhood(design, inverse, distances, weights)
    quartiles = [
        np.unique(np.percentile(training[:, j], [25, 50, 75])).tolist() for j in range(len(names))
    ]
    metadata = {
        "sample_id": sample_id,
        "explainer_seed": explainer_seed,
        "derived_seeds": seeds,
        "feature_names": list(names),
        "training_ids": ids.tolist(),
        "quartile_boundaries": quartiles,
        "training_min": training.min(axis=0).tolist(),
        "training_max": training.max(axis=0).tolist(),
        "anchor_bin_ids": explainer.discretizer.discretize(row).astype(int).tolist(),
        "kernel_width": float(0.75 * np.sqrt(len(names))),
        "fit_samples_including_anchor": fit_samples,
        "evaluation_samples_excluding_forced_anchor": evaluation_samples,
    }
    return LimeDesign(neighborhoods["fit"], neighborhoods["evaluation"], metadata)


def fit_diagnostics(actual: np.ndarray, predicted: np.ndarray, weights: np.ndarray) -> dict:
    actual, predicted, weights = (np.asarray(v, dtype=float) for v in (actual, predicted, weights))
    if (
        actual.ndim != 1
        or actual.size < 2
        or predicted.shape != actual.shape
        or weights.shape != actual.shape
        or not all(np.isfinite(v).all() for v in (actual, predicted, weights))
        or np.any(weights <= 0)
    ):
        raise ValueError("invalid local-fit diagnostic inputs")
    error = predicted - actual
    mse = float(np.average(error**2, weights=weights))
    variance = float(
        np.average((actual - np.average(actual, weights=weights)) ** 2, weights=weights)
    )
    return {
        "weighted_r2": None if variance <= 1e-15 else 1 - mse / variance,
        "weighted_rmse": float(np.sqrt(mse)),
        "weighted_mae": float(np.average(abs(error), weights=weights)),
        "weighted_target_variance": variance,
        "r2_undefined_near_constant_target": variance <= 1e-15,
        "effective_sample_size": float(weights.sum() ** 2 / np.square(weights).sum()),
        "surrogate_out_of_probability_bounds_fraction": float(
            np.mean((predicted < 0) | (predicted > 1))
        ),
    }


def fit_probability_lime(model: Any, design: LimeDesign, *, target: int | None = None) -> dict:
    from lime.lime_base import LimeBase

    fit, evaluation = design.fit, design.evaluation
    probabilities = binary_probabilities(model, np.concatenate((fit.inverse, evaluation.inverse)))
    target = int(probabilities[0].argmax()) if target is None else target
    if type(target) is not int or target not in (0, 1):
        raise ValueError("invalid LIME target class")
    width = design.metadata["kernel_width"]
    base = LimeBase(lambda d: np.exp(-(d**2) / (2 * width**2)), verbose=False, random_state=0)
    ridge = Ridge(alpha=1.0, fit_intercept=True, solver="cholesky")
    intercept, pairs, native_score, local_pred = base.explain_instance_with_data(
        fit.design,
        probabilities[: len(fit.design)],
        fit.distances,
        target,
        num_features=fit.design.shape[1],
        feature_selection="none",
        model_regressor=ridge,
    )
    ordered = sorted(pairs)
    if [int(p[0]) for p in ordered] != list(range(fit.design.shape[1])):
        raise ValueError("LIME feature IDs are missing, duplicated or out of range")
    coefficients = np.array([p[1] for p in ordered])
    fit_prediction = intercept + fit.design @ coefficients
    eval_prediction = intercept + evaluation.design @ coefficients
    if (
        not all(np.isfinite(v).all() for v in (coefficients, fit_prediction, eval_prediction))
        or not np.allclose(fit_prediction[0], local_pred, atol=1e-10, rtol=1e-8)
        or not np.allclose(coefficients, ridge.coef_, atol=1e-10, rtol=1e-8)
    ):
        raise ValueError("LIME coefficient identity or surrogate reconstruction failure")
    fit_metrics = fit_diagnostics(
        probabilities[: len(fit.design), target], fit_prediction, fit.weights
    )
    if fit_metrics["weighted_r2"] is not None and not np.isclose(
        fit_metrics["weighted_r2"], native_score, atol=1e-10, rtol=1e-8
    ):
        raise ValueError("independent local R2 differs from LIME fit score")
    return {
        "coefficients": coefficients,
        "probabilities": probabilities,
        "fit_surrogate_predictions": fit_prediction,
        "evaluation_surrogate_predictions": eval_prediction,
        "diagnostics": {
            "target_class": target,
            "intercept": float(intercept),
            "anchor_model_probability": float(probabilities[0, target]),
            "anchor_surrogate_prediction": float(fit_prediction[0]),
            "anchor_absolute_error": float(abs(fit_prediction[0] - probabilities[0, target])),
            "native_fit_score": float(native_score),
            "fit": fit_metrics,
            "evaluation": fit_diagnostics(
                probabilities[len(fit.design) :, target], eval_prediction, evaluation.weights
            ),
            "model_input_rows": len(probabilities),
            "predict_proba_calls": 1,
        },
    }
