"""Synthetic checks for experiment 02; no research test set used here."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from packaging.requirements import Requirement

from scripts.wisconsin_models import (
    PROTOCOL_PATH,
    SmallGridCNN,
    cnn_probabilities,
    metrics,
    seed_cpu,
    state_digest,
    summarize,
    train_cnn,
)


class WisconsinModelTests(unittest.TestCase):
    def test_cnn_shape_parameters_and_gradient(self):
        model = SmallGridCNN()
        inputs = torch.zeros(4, 1, 5, 6, requires_grad=True)
        scores = model(inputs)
        self.assertEqual(tuple(scores.shape), (4, 2))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1282)
        scores.sum().backward()
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_metrics_labels_confusion_and_ties(self):
        result = metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.8])
        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["macro_f1"], 0.5)
        self.assertEqual(metrics([0, 1], [0.5, 0.5])["confusion_matrix"], [[1, 0], [1, 0]])

    def test_metrics_reject_invalid_probabilities(self):
        for values in ([0.2], [0.2, float("nan")], [0.2, 1.1]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                metrics([0, 1], values)

    def test_fixed_seed_initialization(self):
        seed_cpu(7)
        one = state_digest(SmallGridCNN())
        seed_cpu(7)
        self.assertEqual(one, state_digest(SmallGridCNN()))

    def test_training_deterministic_and_restores_minimum_validation_loss(self):
        rng = np.random.default_rng(11)
        images = rng.normal(size=(20, 1, 5, 6)).astype(np.float32)
        labels = np.array([0, 1] * 10, dtype=np.int64)
        config = json.loads(PROTOCOL_PATH.read_text())["cnn"]
        config.update(max_epochs=4, patience=2)
        first, diagnostics = train_cnn(images[:12], labels[:12], images[12:], labels[12:], config, 9)
        second, repeated = train_cnn(images[:12], labels[:12], images[12:], labels[12:], config, 9)
        self.assertEqual(diagnostics, repeated)
        self.assertEqual(state_digest(first), state_digest(second))
        losses = [entry["validation_loss"] for entry in diagnostics["history"]]
        self.assertEqual(diagnostics["selected_validation_loss"], min(losses))
        self.assertEqual(diagnostics["best_epoch"], losses.index(min(losses)) + 1)

    def test_checkpoint_roundtrip(self):
        model = SmallGridCNN()
        images = np.zeros((2, 1, 5, 6), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            torch.save(model.state_dict(), path)
            restored = SmallGridCNN()
            restored.load_state_dict(torch.load(path, weights_only=True))
            np.testing.assert_array_equal(cnn_probabilities(model, images), cnn_probabilities(restored, images))

    def test_summary_uses_every_seed_and_sample_sd(self):
        scores = [metrics([0, 1], p) for p in ([0.2, 0.8], [0.8, 0.2])]
        result = summarize([{"model": "test", "metrics": {"test": score}} for score in scores])["test"]
        self.assertEqual(result["n_runs"], 2)
        self.assertEqual(result["test"]["accuracy"]["mean"], 0.5)
        self.assertAlmostEqual(result["test"]["accuracy"]["sample_sd"], np.sqrt(0.5))

    def test_requirements_contains_only_valid_pinned_packages(self):
        root = Path(__file__).resolve().parents[1]
        for line in (root / "requirements-learning.txt").read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            requirement = Requirement(line)
            self.assertTrue(any(spec.operator == "==" for spec in requirement.specifier))

    def test_recorded_metrics_match_saved_probabilities(self):
        report = json.loads((PROTOCOL_PATH.parent / "results.json").read_text())
        for run in report["runs"]:
            for partition in ("validation", "test"):
                true = report["data"]["partitions"][partition]["y_true"]
                probabilities = run["predictions"][partition]["probability_m"]
                self.assertEqual(metrics(true, probabilities), run["metrics"][partition])
                self.assertEqual((np.asarray(probabilities) > 0.5).astype(int).tolist(),
                                 run["predictions"][partition]["predicted_label"])

    def test_recorded_cnn_selection_and_seed_coverage(self):
        report = json.loads((PROTOCOL_PATH.parent / "results.json").read_text())
        runs = [run for run in report["runs"] if run["model"] == "grid_cnn"]
        self.assertEqual([run["seed"] for run in runs], report["protocol"]["model_seeds"])
        for run in runs:
            diagnostic = run["diagnostics"]
            losses = [entry["validation_loss"] for entry in diagnostic["history"]]
            self.assertEqual(diagnostic["best_epoch"], losses.index(min(losses)) + 1)
            self.assertEqual(diagnostic["selected_validation_loss"], min(losses))


if __name__ == "__main__":
    unittest.main()
