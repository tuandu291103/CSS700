"""Fixed, paired surrogate diagnostics; no explainer selection or new model calls."""

import numpy as np
from sklearn.linear_model import Ridge

from css700.interpretability.lime_tabular import LimeDesign, fit_diagnostics

VARIANTS = ("reference", "narrow_raw", "narrow_matched_mass", "continuous_same_neighbors")


def variant_inputs(design: LimeDesign, training_scale: np.ndarray, variant: str) -> dict:
    if variant not in VARIANTS:
        raise ValueError("unregistered locality variant")
    fit, evaluation = design.fit, design.evaluation
    scale = np.asarray(training_scale, dtype=float)
    if scale.shape != (fit.design.shape[1],) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("invalid training-only feature scale")
    x, xe = fit.design.copy(), evaluation.design.copy()
    w, we = fit.weights.copy(), evaluation.weights.copy()
    if variant == "continuous_same_neighbors":
        x = (fit.inverse - fit.inverse[0]) / scale
        xe = (evaluation.inverse - fit.inverse[0]) / scale
    multiplier = 1.0
    if variant.startswith("narrow"):
        width = 0.25 * np.sqrt(x.shape[1])
        w = np.exp(-(fit.distances**2) / (2 * width**2))
        we = np.exp(-(evaluation.distances**2) / (2 * width**2))
        if variant == "narrow_matched_mass":
            multiplier = float(fit.weights.sum() / w.sum())
            w *= multiplier  # Evaluation metric ratios do not require this global rescaling.
    if not all(np.isfinite(a).all() for a in (x, xe, w, we)) or np.any(w <= 0) or np.any(we <= 0):
        raise ValueError("nonfinite or zero-weight locality design")
    return dict(
        fit_design=x,
        evaluation_design=xe,
        fit_weights=w,
        evaluation_weights=we,
        weight_multiplier=multiplier,
    )


def anchor_leverage(x: np.ndarray, weights: np.ndarray, *, alpha: float = 1.0) -> float:
    centered = x - np.average(x, axis=0, weights=weights)
    gram = centered.T @ (weights[:, None] * centered) + alpha * np.eye(x.shape[1])
    return float(
        weights[0] * (1 / weights.sum() + centered[0] @ np.linalg.solve(gram, centered[0]))
    )


def fit_locality_variant(
    design: LimeDesign,
    probabilities: np.ndarray,
    training_scale: np.ndarray,
    variant: str,
) -> dict:
    inputs = variant_inputs(design, training_scale, variant)
    x, xe = inputs["fit_design"], inputs["evaluation_design"]
    w, we = inputs["fit_weights"], inputs["evaluation_weights"]
    p = np.asarray(probabilities, dtype=float)
    if (
        p.shape != (len(x) + len(xe), 2)
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or np.any(p > 1)
        or not np.allclose(p.sum(axis=1), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("invalid stored class probabilities")
    target = int(p[0].argmax())
    y, ye = p[: len(x), target], p[len(x) :, target]
    model = Ridge(alpha=1, fit_intercept=True, solver="cholesky").fit(x, y, sample_weight=w)
    fit_prediction, evaluation_prediction = model.predict(x), model.predict(xe)
    if not all(np.isfinite(a).all() for a in (model.coef_, fit_prediction, evaluation_prediction)):
        raise ValueError("invalid surrogate coefficients or predictions")
    nonanchor = w[1:]
    bin_changes = np.sum(design.fit.design[1:] == 0, axis=1)
    input_distance = np.linalg.norm(
        (design.fit.inverse[1:] - design.fit.inverse[0]) / training_scale, axis=1
    )
    diagnostics = {
        "target_class": target,
        "intercept": float(model.intercept_),
        "anchor_model_probability": float(y[0]),
        "anchor_surrogate_prediction": float(fit_prediction[0]),
        "anchor_absolute_error": float(abs(fit_prediction[0] - y[0])),
        "fit": fit_diagnostics(y, fit_prediction, w),
        "own_evaluation": fit_diagnostics(ye, evaluation_prediction, we),
        "reference_evaluation": fit_diagnostics(
            ye, evaluation_prediction, design.evaluation.weights
        ),
        "fit_weight_sum": float(w.sum()),
        "alpha_over_weight_sum": float(1 / w.sum()),
        "fit_weight_multiplier": inputs["weight_multiplier"],
        "anchor_weight_fraction": float(w[0] / w.sum()),
        "anchor_leverage": anchor_leverage(x, w),
        "nonanchor_effective_sample_size": float(nonanchor.sum() ** 2 / np.square(nonanchor).sum()),
        "nonanchor_weighted_changed_bins": float(np.average(bin_changes, weights=nonanchor)),
        "nonanchor_weighted_input_distance": float(np.average(input_distance, weights=nonanchor)),
        "nonanchor_rows_at_most_five_bin_changes": int(np.sum(bin_changes <= 5)),
    }
    return dict(
        coefficients=model.coef_,
        fit_surrogate_predictions=fit_prediction,
        evaluation_surrogate_predictions=evaluation_prediction,
        diagnostics=diagnostics,
    )


def verify_reference(result: dict, original_arrays: dict, original_diagnostics: dict) -> None:
    for key in ("coefficients", "fit_surrogate_predictions", "evaluation_surrogate_predictions"):
        if not np.allclose(result[key], original_arrays[key], atol=1e-10, rtol=1e-8):
            raise ValueError(f"reference does not reproduce accepted v1: {key}")
    current = result["diagnostics"]
    if current["target_class"] != original_diagnostics["target_class"]:
        raise ValueError("reference target class mismatch")
    for key in (
        "intercept",
        "anchor_model_probability",
        "anchor_surrogate_prediction",
        "anchor_absolute_error",
    ):
        if not np.isclose(current[key], original_diagnostics[key], atol=1e-10, rtol=1e-8):
            raise ValueError(f"reference diagnostic mismatch: {key}")
    for new, old in (("fit", "fit"), ("reference_evaluation", "evaluation")):
        for key, expected in original_diagnostics[old].items():
            value = current[new][key]
            if expected is None:
                valid = value is None
            else:
                valid = value is not None and np.isclose(value, expected, atol=1e-10, rtol=1e-8)
            if not valid:
                raise ValueError(f"reference diagnostic mismatch: {old}.{key}")


def paired_contrasts(records: list[dict]) -> list[dict]:
    indexed = {}
    for row in records:
        key = (row["sample_id"], row["explainer_seed"], row["family"], row["variant"])
        if key in indexed:
            raise ValueError("duplicate locality comparison identity")
        indexed[key] = row
    identities = {key[:3] for key in indexed}
    pairs = []
    for sample_id, seed, family in sorted(identities):
        if any((sample_id, seed, family, v) not in indexed for v in VARIANTS):
            raise ValueError("incomplete locality variants")
        reference = indexed[(sample_id, seed, family, "reference")]
        for variant in VARIANTS[1:]:
            row = indexed[(sample_id, seed, family, variant)]
            if row["target_class"] != reference["target_class"]:
                raise ValueError("paired locality target mismatch")
            pairs.append(
                dict(
                    sample_id=sample_id,
                    explainer_seed=seed,
                    family=family,
                    variant=variant,
                    target_class=row["target_class"],
                    delta_anchor_error=row["anchor_absolute_error"]
                    - reference["anchor_absolute_error"],
                    delta_reference_rmse=row["reference_evaluation"]["weighted_rmse"]
                    - reference["reference_evaluation"]["weighted_rmse"],
                )
            )
    return pairs


def descriptive_summaries(records: list[dict], pairs: list[dict]) -> dict:
    def stats(values):
        finite = [v for v in values if v is not None]
        return {
            "count": len(values),
            "defined": len(finite),
            "median": float(np.median(finite)) if finite else None,
            "min": float(min(finite)) if finite else None,
            "max": float(max(finite)) if finite else None,
        }

    summaries, contrasts = [], []
    for family in sorted({r["family"] for r in records}):
        for variant in VARIANTS:
            for seed in (None, 11, 29, 47):
                selected = [
                    r
                    for r in records
                    if r["family"] == family
                    and r["variant"] == variant
                    and (seed is None or r["explainer_seed"] == seed)
                ]
                if not selected:
                    continue
                summaries.append(
                    dict(
                        family=family,
                        variant=variant,
                        explainer_seed=seed,
                        aggregation="descriptive_repeated_fits"
                        if seed is None
                        else "separate_seed",
                        fits=len(selected),
                        anchor_error=stats([r["anchor_absolute_error"] for r in selected]),
                        reference_rmse=stats(
                            [r["reference_evaluation"]["weighted_rmse"] for r in selected]
                        ),
                        reference_r2=stats(
                            [r["reference_evaluation"]["weighted_r2"] for r in selected]
                        ),
                        own_rmse=stats([r["own_evaluation"]["weighted_rmse"] for r in selected]),
                        anchor_leverage=stats([r["anchor_leverage"] for r in selected]),
                        weight_sum=stats([r["fit_weight_sum"] for r in selected]),
                        anchor_weight_fraction=stats(
                            [r["anchor_weight_fraction"] for r in selected]
                        ),
                        effective_sample_size=stats(
                            [r["fit"]["effective_sample_size"] for r in selected]
                        ),
                        nonanchor_effective_sample_size=stats(
                            [r["nonanchor_effective_sample_size"] for r in selected]
                        ),
                        changed_bins=stats(
                            [r["nonanchor_weighted_changed_bins"] for r in selected]
                        ),
                        input_distance=stats(
                            [r["nonanchor_weighted_input_distance"] for r in selected]
                        ),
                        anchor_outside_probability_range=sum(
                            not 0 <= r["anchor_surrogate_prediction"] <= 1 for r in selected
                        ),
                    )
                )
                paired = [
                    r
                    for r in pairs
                    if r["family"] == family
                    and r["variant"] == variant
                    and (seed is None or r["explainer_seed"] == seed)
                ]
                if paired:
                    contrasts.append(
                        dict(
                            family=family,
                            variant=variant,
                            explainer_seed=seed,
                            paired_fits=len(paired),
                            delta_anchor_error=stats([r["delta_anchor_error"] for r in paired]),
                            delta_reference_rmse=stats([r["delta_reference_rmse"] for r in paired]),
                            both_errors_lower=sum(
                                r["delta_anchor_error"] < 0 and r["delta_reference_rmse"] < 0
                                for r in paired
                            ),
                        )
                    )
    return {
        "conditions": summaries,
        "contrasts": contrasts,
        "interpretation": (
            "Paired descriptions only; no intervals, selection or independent-seed claim."
        ),
    }
