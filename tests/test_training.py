import numpy as np
import pytest

torch = pytest.importorskip("torch")

from css700.training import train_grid_cnn  # noqa: E402


def _easy_split(samples_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.asarray([0, 1], dtype=np.int64), samples_per_class)
    signal = np.where(labels[:, None, None, None] == 0, -1.0, 1.0)
    images = signal + rng.normal(0, 0.05, size=(len(labels), 1, 3, 3))
    return images.astype(np.float32), labels


def test_grid_training_is_reproducible_and_retains_ordered_predictions() -> None:
    train_images, train_labels = _easy_split(12, 1)
    validation_images, validation_labels = _easy_split(4, 2)
    test_images, test_labels = _easy_split(4, 3)
    arguments = {
        "seed": 11,
        "epochs": 8,
        "batch_size": 8,
        "learning_rate": 0.01,
        "early_stopping_patience": 3,
        "device_name": "cpu",
    }

    first = train_grid_cnn(
        train_images,
        train_labels,
        validation_images,
        validation_labels,
        test_images,
        test_labels,
        **arguments,
    )
    second = train_grid_cnn(
        train_images,
        train_labels,
        validation_images,
        validation_labels,
        test_images,
        test_labels,
        **arguments,
    )

    assert first.best_epoch == second.best_epoch
    assert first.test.predictions.shape == test_labels.shape
    assert first.test.metrics["accuracy"] == 1.0
    np.testing.assert_array_equal(first.test.predictions, second.test.predictions)
    np.testing.assert_allclose(first.test.logits, second.test.logits, rtol=0, atol=0)
