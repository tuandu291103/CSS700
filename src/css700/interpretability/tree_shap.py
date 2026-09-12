"""Explicit binary probability TreeSHAP with independent reconstruction checks."""

import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from css700.interpretability.tree_routing import (
    TreeRoutingMismatch,
    random_forest_routing_audit,
)


def training_background(
    training_values: np.ndarray,
    training_ids: np.ndarray,
    heldout_ids: np.ndarray,
    *,
    size: int | None = None,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray]:
    """Select only training rows; never accept held-out values or labels."""

    values = np.asarray(training_values, dtype=np.float64)
    train, heldout = np.asarray(training_ids), np.asarray(heldout_ids)
    if (
        values.ndim != 2
        or min(values.shape) == 0
        or train.shape != (len(values),)
        or heldout.ndim != 1
        or not np.isfinite(values).all()
        or not np.issubdtype(train.dtype, np.integer)
        or not np.issubdtype(heldout.dtype, np.integer)
    ):
        raise ValueError("invalid training values or sample IDs")
    if len(np.unique(train)) != len(train) or len(np.unique(heldout)) != len(heldout):
        raise ValueError("sample IDs must be unique within partitions")
    if np.intersect1d(train, heldout).size:
        raise ValueError("background training IDs overlap held-out IDs")
    if size is None:
        positions = np.arange(len(train))
    else:
        if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= len(train):
            raise ValueError("background size must be an eligible positive integer")
        positions = np.sort(np.random.default_rng(seed).permutation(len(train))[:size])
    return values[positions].copy(), train[positions].copy()


def binary_probabilities(model: Any, values: np.ndarray) -> np.ndarray:
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("expected class order [0, 1]; refuse implicit class-axis relabeling")
    result = np.asarray(model.predict_proba(values), dtype=np.float64)
    if (
        result.shape != (len(values), 2)
        or not np.isfinite(result).all()
        or np.any(result < 0)
        or np.any(result > 1)
        or not np.allclose(result.sum(axis=1), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("invalid binary probabilities")
    return result  # Do not normalize: this is the scalar function being explained.


def normalize_binary_shap(
    raw: np.ndarray,
    base: Any,
    *,
    samples: int,
    features: int,
    family: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Normalize verified model-specific output contracts, never guess axes."""

    raw, base = np.asarray(raw, dtype=np.float64), np.asarray(base, dtype=np.float64)
    if family not in ("random_forest", "xgboost"):
        raise ValueError("unsupported tree family")
    if raw.shape == (samples, features, 2) and base.shape == (2,):
        rule = "two_native_class_outputs"
    elif family == "xgboost" and raw.shape == (samples, features) and base.size == 1:
        raw = np.stack((-raw, raw), axis=-1)
        positive_base = float(base.reshape(-1)[0])
        base = np.array([1 - positive_base, positive_base])
        rule = "class_0_complement_of_native_class_1_probability"
    else:
        raise ValueError(
            f"unrecognized {family} SHAP shapes: values={raw.shape}, base={base.shape}"
        )
    if not np.isfinite(raw).all() or not np.isfinite(base).all():
        raise ValueError("nonfinite SHAP values or base values")
    return raw, base, rule


def reconstruction_checks(
    values: np.ndarray,
    base: np.ndarray,
    probabilities: np.ndarray,
    background_probability_mean: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> tuple[np.ndarray, dict[str, bool]]:
    residuals = values.sum(axis=1) + base - probabilities
    checks = {
        "probability_reconstruction": bool(
            np.all(np.abs(residuals) <= atol + rtol * np.abs(probabilities))
        ),
        "base_matches_background_probability_mean": bool(
            np.all(
                np.abs(base - background_probability_mean)
                <= atol + rtol * np.abs(background_probability_mean)
            )
        ),
        "binary_contributions_complementary": bool(
            np.allclose(values[:, :, 0], -values[:, :, 1], atol=atol, rtol=rtol)
        ),
    }
    return residuals, checks


@dataclass(frozen=True)
class TreeShapResult:
    values: np.ndarray
    base: np.ndarray
    probabilities: np.ndarray
    residuals: np.ndarray
    checks: dict[str, bool]
    output_rule: str
    elapsed_seconds: float


def verified_numeric_booster(model: Any) -> Any:
    """Use unchanged fitted trees when the sklearn wrapper enables unused categories.

    SHAP 0.52 rejects enable_categorical=True even for entirely numerical trees.
    Never disable its guard for actual categorical splits or mutate the estimator.
    """

    booster = model.get_booster()
    if booster.feature_types is not None and "c" in booster.feature_types:
        raise ValueError("categorical feature metadata is outside this numerical pilot")
    raw = json.loads(booster.save_raw(raw_format="json"))
    trees = raw["learner"]["gradient_booster"]["model"]["trees"]
    if not trees or any(
        any(value != 0 for value in tree["split_type"])
        or any(
            tree[key]
            for key in ("categories", "categories_nodes", "categories_segments", "categories_sizes")
        )
        for tree in trees
    ):
        raise ValueError("categorical or unrecognized tree splits are outside this pilot")
    return booster


class ProbabilityTreeSHAP:
    """Fit an explicit empirical-background explainer, not a new predictor."""

    def __init__(
        self,
        model: Any,
        background: np.ndarray,
        feature_names: tuple[str, ...],
        *,
        family: str,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ) -> None:
        import shap

        self.model, self.family = model, family
        self.names = tuple(feature_names)
        self.background = np.asarray(background, dtype=np.float64).copy()
        if (
            len(set(self.names)) != len(self.names)
            or not self.names
            or any(not name for name in self.names)
        ):
            raise ValueError("feature names must be nonempty and unique")
        if (
            self.background.ndim != 2
            or self.background.shape[1] != len(self.names)
            or not len(self.background)
            or not np.isfinite(self.background).all()
            or model.n_features_in_ != len(self.names)
        ):
            raise ValueError("background and model feature dimensions do not match")
        if hasattr(model, "feature_names_in_") and tuple(model.feature_names_in_) != self.names:
            raise ValueError("model feature order does not match")
        if min(atol, rtol) < 0 or not np.isfinite([atol, rtol]).all():
            raise ValueError("invalid reconstruction tolerances")
        self.atol, self.rtol = atol, rtol
        self.routing_audit = None
        if family == "random_forest":
            if shap.__version__ != "0.52.0":
                raise ValueError("review the RF routing guard before changing SHAP 0.52.0")
            self.routing_audit = random_forest_routing_audit(model, self.background)
            if not self.routing_audit["passed"]:
                raise TreeRoutingMismatch(self.routing_audit)
        self.background_mean = binary_probabilities(model, self.background).mean(axis=0)
        explanation_model = model
        self.adapter = "original_fitted_estimator"
        if family == "xgboost":
            explanation_model = verified_numeric_booster(model)
            self.adapter = "original_fitted_booster_verified_numerical_splits"
        self.explainer = shap.TreeExplainer(
            explanation_model,
            data=shap.maskers.Independent(self.background, max_samples=len(self.background)),
            feature_names=list(self.names),
            model_output="probability",
            feature_perturbation="interventional",
        )
        if not np.array_equal(self.explainer.data, self.background):
            raise RuntimeError("TreeSHAP changed the background rows/values/order")
        if (
            self.explainer.feature_perturbation != "interventional"
            or self.explainer.model.model_output != "probability"
        ):
            raise RuntimeError("TreeSHAP output/perturbation contract changed")

    def explain(self, values: np.ndarray, *, batch_size: int = 32) -> TreeShapResult:
        values = np.asarray(values, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != len(self.names)
            or not len(values)
            or not np.isfinite(values).all()
        ):
            raise ValueError("invalid explanation values")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        started = time.perf_counter()
        parts = []
        for start in range(0, len(values), batch_size):
            chunk = values[start : start + batch_size]
            raw = self.explainer.shap_values(chunk, approximate=False, check_additivity=True)
            normalized, base, rule = normalize_binary_shap(
                raw,
                self.explainer.expected_value,
                samples=len(chunk),
                features=len(self.names),
                family=self.family,
            )
            parts.append(normalized)
        result = np.concatenate(parts)
        probabilities = binary_probabilities(self.model, values)
        residuals, checks = reconstruction_checks(
            result,
            base,
            probabilities,
            self.background_mean,
            atol=self.atol,
            rtol=self.rtol,
        )
        return TreeShapResult(
            result, base, probabilities, residuals, checks, rule, time.perf_counter() - started
        )
