"""Training-only routing guard for SHAP 0.52's interventional RF implementation."""

from typing import Any

import numpy as np


class TreeRoutingMismatch(ValueError):
    """Retain routing evidence without accepting explanations of a different function."""

    def __init__(self, audit: dict) -> None:
        self.audit = audit
        super().__init__(
            "Random Forest and SHAP interventional split comparisons differ on training "
            "background values; probability reconstruction alone is insufficient"
        )


def random_forest_routing_audit(model: Any, background: np.ndarray) -> dict:
    """Compare every internal-node decision over the declared training background.

    sklearn predicts using float32 inputs and stored float64 thresholds. SHAP 0.52's
    independent routine uses float32 thresholds, casts query inputs to float32, and
    retains background precision. Check query and background roles separately. This
    conservative check includes nodes not reached by an unmodified row because
    interventional feature mixtures can change the path. Counts are node/value
    comparisons, not counts of changed final predictions. Passing is not a proof for
    arbitrary future inputs. Neither the predictor nor the explainer is modified.
    """

    source = np.asarray(background, dtype=np.float64)
    if (
        source.ndim != 2
        or not len(source)
        or source.shape[1] != model.n_features_in_
        or not np.isfinite(source).all()
    ):
        raise ValueError("invalid routing audit background")
    with np.errstate(over="ignore"):
        predictor_inputs = source.astype(np.float32).astype(np.float64)
    if not np.isfinite(predictor_inputs).all():
        raise ValueError("routing inputs cannot be represented as finite float32")
    counts = {"background_role": 0, "query_role": 0}
    examples: list[dict] = []
    checked = internal_nodes = 0
    for tree_index, estimator in enumerate(model.estimators_):
        tree = estimator.tree_
        nodes = np.flatnonzero(tree.children_left >= 0)
        features, thresholds = tree.feature[nodes], tree.threshold[nodes]
        narrowed = thresholds.astype(np.float32).astype(np.float64)
        expected = predictor_inputs[:, features] <= thresholds
        internal_nodes += len(nodes)
        checked += expected.size
        for role, values in (("background_role", source), ("query_role", predictor_inputs)):
            actual = values[:, features] <= narrowed
            mismatch = actual != expected
            counts[role] += int(mismatch.sum())
            for row, column in np.argwhere(mismatch):
                if len(examples) == 20:
                    break
                examples.append(
                    {
                        "role": role,
                        "tree_index": tree_index,
                        "node_index": int(nodes[column]),
                        "background_position": int(row),
                        "feature_index": int(features[column]),
                        "source_value": float(source[row, features[column]]),
                        "predictor_input_value": float(predictor_inputs[row, features[column]]),
                        "stored_threshold": float(thresholds[column]),
                        "shap_independent_threshold": float(narrowed[column]),
                        "predictor_goes_left": bool(expected[row, column]),
                        "shap_goes_left": bool(actual[row, column]),
                    }
                )
    return {
        "passed": not any(counts.values()),
        "scope": "training background only; all internal-node comparisons, including mixtures",
        "implementation_reference": "SHAP 0.52.0 independent Node float threshold",
        "background_rows": len(source),
        "internal_nodes": internal_nodes,
        "checked_comparisons_per_role": checked,
        "mismatch_counts": counts,
        "example_limit": 20,
        "examples": examples,
    }
