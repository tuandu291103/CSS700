"""Copied RF tree representation with equivalent routing on finite float32 inputs."""

from typing import Any

import numpy as np

from css700.interpretability.tree_routing import random_forest_routing_audit


def floor_float32_thresholds(thresholds: np.ndarray) -> np.ndarray:
    """Largest float32 <= each threshold, represented exactly in float64.

    For finite float32 z, z <= t iff z <= floor32(t). Nearest rounding alone
    does not have this property when it rounds t upward. Reject unsupported
    thresholds instead of saturating them or relying on undefined casts.
    """

    thresholds = np.asarray(thresholds, dtype=np.float64)
    if not np.isfinite(thresholds).all() or np.any(np.abs(thresholds) > np.finfo(np.float32).max):
        raise ValueError("thresholds must be finite and within float32 range")
    with np.errstate(over="ignore", under="ignore"):
        nearest = thresholds.astype(np.float32)
        floor = np.where(
            nearest.astype(np.float64) > thresholds,
            np.nextafter(nearest, np.float32(-np.inf)),
            nearest,
        )
        successor = np.nextafter(floor, np.float32(np.inf)).astype(np.float64)
    result = floor.astype(np.float64)
    if not np.isfinite(result).all() or not np.all(
        (result <= thresholds) & (thresholds < successor)
    ):
        raise ValueError("thresholds are outside the supported finite float32 lattice")
    return result


def equivalent_forest_payload(model: Any, background: np.ndarray) -> tuple[dict, np.ndarray, dict]:
    """Copy a numerical binary RF for SHAP; never mutate fitted predictor arrays.

    Both explained inputs and empirical background values undergo the predictor's
    own float32 input conversion. Coordinate-wise conversion commutes with feature
    replacement, so this preserves the original empirical-background game.
    """

    from sklearn.ensemble import RandomForestClassifier

    if (
        not isinstance(model, RandomForestClassifier)
        or model.n_outputs_ != 1
        or not np.array_equal(model.classes_, [0, 1])
    ):
        raise ValueError("adapter requires a fitted single-output binary Random Forest")
    source_audit = random_forest_routing_audit(model, background)
    source = np.asarray(background, dtype=np.float64)
    effective = source.astype(np.float32).astype(np.float64)
    trees, witnesses = [], []
    changed = checked = mismatches = 0
    for index, estimator in enumerate(model.estimators_):
        tree = estimator.tree_
        nodes = np.flatnonzero(tree.children_left >= 0)
        old = tree.threshold[nodes]
        new = floor_float32_thresholds(old)
        thresholds = tree.threshold.copy()
        thresholds[nodes] = new
        features = tree.feature[nodes]
        reference = effective[:, features] <= old
        adapted = effective[:, features] <= new
        checked += reference.size
        mismatches += int(np.count_nonzero(reference != adapted))
        changed += int(np.count_nonzero(old != new))
        if len(nodes):
            # The two adjacent representable values bracketing t certify every
            # finite float32 value at that node, not just the observed rows.
            upper = np.nextafter(new.astype(np.float32), np.float32(np.inf)).astype(np.float64)
            if not np.all((new <= old) & (old < upper)):
                raise RuntimeError("threshold lattice certificate failed")
            witnesses.append({"tree_index": index, "certified_internal_nodes": len(nodes)})
        values = tree.value[:, 0, :].astype(np.float64, copy=True)
        values /= values.sum(axis=1, keepdims=True)
        values /= len(model.estimators_)
        default_left = getattr(tree, "missing_go_to_left", np.zeros(tree.node_count, dtype=bool))
        trees.append(
            {
                "children_left": tree.children_left.copy(),
                "children_right": tree.children_right.copy(),
                "children_default": np.where(default_left, tree.children_left, tree.children_right),
                "features": tree.feature.copy(),
                "thresholds": thresholds,
                "values": values,
                "node_sample_weight": tree.weighted_n_node_samples.copy(),
            }
        )
    if mismatches:
        raise RuntimeError("adapted background routing differs from predictor")
    payload = {
        "trees": trees,
        "input_dtype": np.float32,
        "internal_dtype": np.float64,
        "objective": "binary_crossentropy",
        "tree_output": "probability",
        "base_offset": 0.0,
    }
    audit = {
        "passed": True,
        "adapter": "float32_equivalent_v1",
        "original_native_routing_audit": source_audit,
        "scope": "finite float32 query/background values and their coordinate-wise mixtures",
        "threshold_rule": "largest finite float32 value not exceeding stored threshold",
        "threshold_certificate": "floor32(t) <= t < nextafter32(floor32(t), +infinity)",
        "certified_internal_nodes": sum(w["certified_internal_nodes"] for w in witnesses),
        "changed_internal_threshold_representations": changed,
        "background_node_comparisons": checked,
        "adapted_background_routing_mismatches": mismatches,
        "max_background_input_projection": float(np.abs(effective - source).max()),
        "original_model_mutated": False,
        "missing_or_nonfinite_inputs_supported": False,
    }
    return payload, effective, audit
