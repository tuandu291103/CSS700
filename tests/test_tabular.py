import numpy as np

from css700.training import (
    expand_search_grid,
    select_candidate_index,
    train_tabular_baseline,
)


def _easy_tabular_split(samples_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.asarray([0, 1], dtype=np.int64), samples_per_class)
    signal = np.where(labels[:, None] == 0, -1.0, 1.0)
    features = signal + rng.normal(0.0, 0.05, size=(len(labels), 4))
    return features.astype(np.float64), labels


def test_search_grid_order_and_first_tie_selection() -> None:
    candidates = expand_search_grid({"depth": [2, 4], "leaf": [1, 3]})
    assert candidates == (
        {"depth": 2, "leaf": 1},
        {"depth": 2, "leaf": 3},
        {"depth": 4, "leaf": 1},
        {"depth": 4, "leaf": 3},
    )
    assert select_candidate_index([0.8, 0.9, 0.9]) == 1


def test_random_forest_selection_is_reproducible_and_retains_test_order() -> None:
    train_features, train_labels = _easy_tabular_split(20, 1)
    validation_features, validation_labels = _easy_tabular_split(6, 2)
    test_features, test_labels = _easy_tabular_split(6, 3)
    arguments = {
        "seed": 11,
        "fixed_parameters": {"n_estimators": 20},
        "search_grid": {"max_depth": [1, None], "min_samples_leaf": [1, 2]},
    }
    first = train_tabular_baseline(
        "random_forest",
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        test_features,
        test_labels,
        **arguments,
    )
    second = train_tabular_baseline(
        "random_forest",
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        test_features,
        test_labels,
        **arguments,
    )

    assert sum(int(row["selected"]) for row in first.search_rows) == 1
    assert first.validation.metrics["macro_f1"] == 1.0
    assert first.test.metrics["accuracy"] == 1.0
    assert first.test.predictions.shape == test_labels.shape
    np.testing.assert_array_equal(first.test.predictions, second.test.predictions)
    np.testing.assert_allclose(first.test.probabilities, second.test.probabilities, rtol=0, atol=0)
