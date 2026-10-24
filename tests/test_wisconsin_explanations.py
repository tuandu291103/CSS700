"""Analytic controls and saved-output checks; no classifier fitting or held-out tuning."""

import json
import unittest

import numpy as np
import torch
from torch import nn

from scripts.wisconsin_explanations import (
    PROTOCOL,
    agreement,
    choose_cases,
    deletion_check,
    gradcam,
    integrated_gradients,
    kernel_shap,
    lime_effects,
    predictor,
    summarize,
)
from scripts.wisconsin_learning import FEATURES, ROOT
from scripts.wisconsin_models import sha256, state_digest


class LinearImage(nn.Module):
    def forward(self, inputs):
        weights = torch.arange(1, inputs[0].numel() + 1, dtype=inputs.dtype)
        score = (inputs.flatten(1) * weights).sum(1)
        return torch.stack([torch.zeros_like(score), score], 1)


class ToyCAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = nn.Conv2d(1, 1, 1, bias=False)
        with torch.no_grad():
            self.conv2.weight.fill_(2)

    def forward(self, inputs):
        score = self.conv2(inputs).mean((1, 2, 3))
        return torch.stack([torch.zeros_like(score), score], 1)


class AttributionTests(unittest.TestCase):
    def test_ig_exact_linear_effects_completeness_and_target(self):
        image = np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2)
        baseline = np.full_like(image, 0.5)
        positive, diagnostics = integrated_gradients(LinearImage(), image, baseline, steps=8, maximum_steps=8)
        negative, _ = integrated_gradients(LinearImage(), image, baseline, sign=-1, steps=8, maximum_steps=8)
        expected = (image - baseline) * np.arange(1, 5).reshape(1, 1, 2, 2)
        np.testing.assert_allclose(positive, expected, atol=1e-5)
        np.testing.assert_allclose(negative, -expected, atol=1e-5)
        self.assertTrue(diagnostics["numerical_tolerance_met"])

    def test_gradcam_analytic_layer_map_relu_and_unchanged_weights(self):
        model = ToyCAM()
        before = state_digest(model)
        image = np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2)
        np.testing.assert_allclose(gradcam(model, image), image * 2 / 4)
        np.testing.assert_array_equal(gradcam(model, image, sign=-1), np.zeros_like(image))
        self.assertEqual(before, state_digest(model))

    def test_tabular_prediction_reencodes_feature_positions(self):
        mapping = [{"feature": name, "feature_index": i, "row": i // 6, "column": i % 6}
                   for i, name in enumerate(FEATURES)]
        rows = np.arange(60).reshape(2, 30).astype(np.float32)
        np.testing.assert_allclose(predictor(LinearImage(), mapping, sign=-1)(rows), -rows @ np.arange(1, 31))

    def test_kernel_shap_matches_linear_effects(self):
        weights = np.array([1.5, -2.0, 0.0, 4.0])
        row, baseline = np.array([2.0, 1.0, 0.1, -0.5]), np.full(4, 0.2)
        values, diagnostic = kernel_shap(lambda x: x @ weights + 2.0, row, baseline, seed=3)
        np.testing.assert_allclose(values, (row - baseline) * weights, atol=1e-10)
        self.assertLess(abs(diagnostic["additivity_residual"]), 1e-10)

    def test_lime_local_effect_rescales_standardized_coefficients(self):
        rng = np.random.default_rng(9)
        training = rng.normal(size=(400, 4)) * np.array([1, 2, 0.5, 3])
        weights = np.array([1.5, -2, 0, 4])
        row, baseline = training[0], np.median(training, axis=0)
        values, diagnostic = lime_effects(lambda x: x @ weights + 2.0, row, baseline, training, 4)
        np.testing.assert_allclose(values, (row - baseline) * weights, atol=0.05)
        self.assertGreater(diagnostic["weighted_neighborhood_r2"], 0.999)
        np.testing.assert_allclose(values, np.array(diagnostic["standardized_local_coefficients"]) *
                                   (row - baseline) / diagnostic["training_scale"])

    def test_primary_selection_balanced_and_error_separate(self):
        labels = np.array([0, 1] * 9)
        predicted = labels.copy()
        predicted[16] = 1
        primary, selected = choose_cases(labels, predicted)
        self.assertEqual(primary, list(range(12)))
        self.assertEqual(selected, [*range(12), 16])

    def test_constant_map_agreement_is_undefined(self):
        self.assertIsNone(agreement(np.zeros(30), np.arange(30))["absolute_spearman"])
        self.assertAlmostEqual(agreement(np.arange(30), -np.arange(30))["top5_jaccard"], 1.0)

    def test_signed_deletion_and_matched_reproducible_random_control(self):
        weights = np.arange(1, 31)
        def predict(x):
            return x @ weights
        scores = {"linear": weights, "constant": np.zeros(30)}
        result = deletion_check(predict, np.ones(30), np.zeros(30), scores)
        self.assertEqual(result, deletion_check(predict, np.ones(30), np.zeros(30), scores))
        self.assertEqual(result["5"]["methods"]["linear"]["selected_indices"], [29, 28, 27, 26, 25])
        self.assertEqual(result["5"]["methods"]["linear"]["score_drop"], 140)
        self.assertFalse(result["5"]["methods"]["constant"]["eligible"])

    def test_saved_report_hashes_shapes_and_summaries(self):
        report = json.loads((PROTOCOL.parent / "results.json").read_text())
        self.assertEqual(report["protocol_sha256"], sha256(PROTOCOL))
        self.assertEqual(report["source_code_sha256"], sha256(ROOT / "scripts/wisconsin_explanations.py"))
        self.assertEqual(report["summary"], summarize(report["cases"]))
        self.assertEqual(report["summary"]["primary_n"], 12)
        for case in report["cases"]:
            self.assertEqual(case["target_sign"], 1 if case["original_predicted_label"] else -1)
            for vector in case["attributions"].values():
                self.assertEqual(len(vector), 30)
                self.assertTrue(np.isfinite(vector).all())
            self.assertLess(case["logistic_regression"]["shap_max_error_vs_exact"], 1e-5)


if __name__ == "__main__":
    unittest.main()
