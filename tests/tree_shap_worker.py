"""Run optional-stack fixtures in a process that never imports the CNN runtime."""

import hashlib
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from css700.interpretability import ProbabilityTreeSHAP


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
    assert "torch" not in sys.modules
    print("TreeSHAP fixtures passed; numerical booster unchanged; no PyTorch imported")


if __name__ == "__main__":
    main()
