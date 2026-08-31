import numpy as np

from css700.data import load_breast_cancer_splits


def test_splits_are_disjoint_stratified_and_training_scaled() -> None:
    bundle = load_breast_cancer_splits(seed=42)

    train = set(bundle.train_indices.tolist())
    validation = set(bundle.validation_indices.tolist())
    test = set(bundle.test_indices.tolist())
    assert not (train & validation or train & test or validation & test)
    assert len(train | validation | test) == 569
    assert bundle.x_train.shape[1] == len(bundle.feature_names) == 30
    assert np.abs(bundle.x_train.mean(axis=0)).max() < 1e-10
    assert set(np.unique(bundle.y_train)) == {0, 1}
    assert set(np.unique(bundle.y_validation)) == {0, 1}
    assert set(np.unique(bundle.y_test)) == {0, 1}


def test_split_is_reproducible() -> None:
    first = load_breast_cancer_splits(seed=7)
    second = load_breast_cancer_splits(seed=7)
    np.testing.assert_array_equal(first.train_indices, second.train_indices)
    np.testing.assert_array_equal(first.validation_indices, second.validation_indices)
    np.testing.assert_array_equal(first.test_indices, second.test_indices)
