"""Run optional-stack fixtures in a process that never imports the CNN runtime."""

import hashlib
import itertools
import math
import sys

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from css700.interpretability import ProbabilityTreeSHAP
from css700.interpretability.tree_routing import TreeRoutingMismatch


def check_brute_force_forest_shap() -> None:
    """Independent exact subset enumeration on a three-feature boundary fixture."""

    bits = np.array(list(itertools.product([0, 1], repeat=3)))
    low = np.nextafter(np.float32(16), np.float32(np.inf))
    high = np.nextafter(low, np.float32(np.inf))
    data = np.where(bits, high, low).astype(float)
    labels = ((bits[:, 0] | bits[:, 1]) ^ bits[:, 2]).astype(int)
    model = RandomForestClassifier(
        n_estimators=3, bootstrap=False, max_features=None, random_state=42, n_jobs=1
    ).fit(data, labels)
    before = joblib.hash(model)
    background = np.nextafter(data, np.inf)  # non-representable values; predictor rounds them
    explainer = ProbabilityTreeSHAP(
        model,
        background,
        ("a", "b", "c"),
        family="random_forest",
        random_forest_adapter="float32_equivalent_v1",
    )
    queries = np.vstack((data, np.nextafter(data, -np.inf)))
    result = explainer.explain(queries)
    assert all(result.checks.values()), result.checks
    assert explainer.routing_audit["adapted_background_routing_mismatches"] == 0
    assert explainer.routing_audit["certified_internal_nodes"] > 0
    expected = np.zeros_like(result.values)
    for i, query in enumerate(queries):
        coalition = {}
        for subset in range(8):
            mixture = background.copy()
            columns = [j for j in range(3) if subset & (1 << j)]
            mixture[:, columns] = query[columns]
            native = model.predict_proba(mixture)
            np.testing.assert_allclose(
                explainer.explainer.model.predict(mixture), native, atol=1e-12
            )
            coalition[subset] = native.mean(axis=0)
        for j in range(3):
            for subset in range(8):
                if not subset & (1 << j):
                    k = subset.bit_count()
                    weight = math.factorial(k) * math.factorial(2 - k) / math.factorial(3)
                    expected[i, j] += weight * (coalition[subset | (1 << j)] - coalition[subset])
    np.testing.assert_allclose(result.values, expected, atol=1e-7, rtol=1e-7)
    assert joblib.hash(model) == before


def main() -> None:
    features = np.zeros((120, 12))
    labels = np.tile([0, 1], 60)
    features[:, 0] = labels
    names = tuple(f"f{i}" for i in range(12))
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=5, max_features=None, random_state=3, n_jobs=1
        ),
        "xgboost": XGBClassifier(
            n_estimators=5, max_depth=2, random_state=3, n_jobs=1, enable_categorical=True
        ),
    }
    for family, model in models.items():
        model.fit(features, labels)
        before = (
            hashlib.sha256(model.get_booster().save_raw()).hexdigest()
            if family == "xgboost"
            else None
        )
        explainer = ProbabilityTreeSHAP(model, features, names, family=family)
        result = explainer.explain(features[:4], batch_size=3)
        assert explainer.explainer.data.shape == (120, 12)  # No default 100-row cap.
        assert result.values.shape == (4, 12, 2)
        assert all(result.checks.values()), result.checks
        np.testing.assert_allclose(result.values[:, 1:, :], 0, atol=1e-10)
        repeated = explainer.explain(features[:4], batch_size=1)
        np.testing.assert_allclose(repeated.values, result.values, atol=1e-10, rtol=1e-8)
        if family == "xgboost":
            assert before == hashlib.sha256(model.get_booster().save_raw()).hexdigest()
            assert model.enable_categorical is True
    constant = RandomForestClassifier(n_estimators=3, bootstrap=False, random_state=42).fit(
        np.zeros_like(features), labels
    )
    result = ProbabilityTreeSHAP(
        constant, np.zeros_like(features), names, family="random_forest"
    ).explain(features[:3])
    np.testing.assert_array_equal(result.values, 0)
    assert all(result.checks.values())
    low = np.nextafter(np.float32(16), np.float32(np.inf))
    high = np.nextafter(low, np.float32(np.inf))
    edge = np.zeros_like(features)
    edge[:, 0] = np.where(labels == 0, low, high)
    boundary_model = RandomForestClassifier(
        n_estimators=1, max_features=None, bootstrap=False, random_state=3
    ).fit(edge, labels)
    try:
        ProbabilityTreeSHAP(boundary_model, edge, names, family="random_forest")
    except TreeRoutingMismatch as error:
        assert error.audit["mismatch_counts"]["query_role"] == 60
    else:
        raise AssertionError("near-threshold RF fixture must fail before explanation")
    repaired = ProbabilityTreeSHAP(
        boundary_model,
        edge,
        names,
        family="random_forest",
        random_forest_adapter="float32_equivalent_v1",
    ).explain(edge[:4])
    assert all(repaired.checks.values())
    np.testing.assert_allclose(repaired.values[:, 1:, :], 0, atol=1e-10)
    check_brute_force_forest_shap()
    assert "torch" not in sys.modules
    print("TreeSHAP fixtures passed; numerical booster unchanged; no PyTorch imported")


if __name__ == "__main__":
    main()
