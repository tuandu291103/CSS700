"""Analytic controls and saved-evidence checks for Experiment 08."""

import json
import unittest

import numpy as np
import torch
from torch import nn

from scripts.som2img_attributions import (
    PROTOCOL,
    choose_cases,
    compare,
    decode,
    encode,
    load_frozen,
    occlusion,
    pairwise_stability,
    predictor,
    replacement,
    summarize,
)
from scripts.wisconsin_explanations import integrated_gradients
from scripts.wisconsin_learning import ROOT
from scripts.wisconsin_models import seed_cpu, sha256


class LinearGrid(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, images):
        margin = (images[:, 0] * self.weights).flatten(1).sum(1)
        return torch.stack((torch.zeros_like(margin), margin), dim=1)


class AttributionControls(unittest.TestCase):
    def setUp(self):
        seed_cpu(10)
        self.fm = np.array([[2, 5, 1], [0, 4, 3]])
        self.row = np.array([0.2, 0.7, -0.3, 1.3, 0.8, 0.1], dtype=np.float32)
        self.baseline = np.full(6, 0.1, dtype=np.float32)
        self.coefficients = np.array([1, -2, 3, 0.5, -1, 2], dtype=np.float32)
        self.model = LinearGrid(encode(self.coefficients[None], self.fm)[0, 0])

    def test_roundtrip_preserves_out_of_range_values(self):
        np.testing.assert_array_equal(decode(encode(self.row[None], self.fm)[:, 0], self.fm, 6), self.row[None])

    def test_encoder_rejects_nonfinite_and_duplicate_map(self):
        for values, fm in ((np.full((1, 6), np.nan), self.fm), (self.row[None], np.zeros((2, 3), int))):
            with self.assertRaises(ValueError):
                encode(values, fm)

    def test_selection_uses_only_labels_and_order(self):
        self.assertEqual(choose_cases([1, 0, 1, 1, 0, 0], 2), [0, 1, 2, 4])

    def test_selection_rejects_insufficient_class(self):
        with self.assertRaises(ValueError):
            choose_cases([1, 0, 1], 2)

    def test_fixed_target_sign_reverses_scores(self):
        a = predictor(self.model, self.fm, 1)(self.row[None])
        b = predictor(self.model, self.fm, -1)(self.row[None])
        np.testing.assert_array_equal(a, -b)

    def test_linear_ig_decodes_to_exact_named_effects(self):
        values, diagnostics = integrated_gradients(
            self.model, encode(self.row[None], self.fm), encode(self.baseline[None], self.fm), steps=32)
        actual = decode(values[:, 0], self.fm, 6)[0]
        np.testing.assert_allclose(actual, self.coefficients * (self.row - self.baseline), atol=1e-6)
        self.assertTrue(diagnostics["numerical_tolerance_met"])

    def test_occlusion_agrees_with_linear_effects(self):
        values, error = occlusion(self.model, self.row, self.baseline, self.fm, -1)
        np.testing.assert_allclose(values, -self.coefficients * (self.row - self.baseline), atol=1e-6)
        self.assertLess(error, 1e-6)

    def test_joint_feature_permutation_preserves_linear_explanation(self):
        fixed = np.arange(6).reshape(2, 3)
        model = LinearGrid(encode(self.coefficients[None], fixed)[0, 0])
        a, _ = occlusion(self.model, self.row, self.baseline, self.fm, 1)
        b, _ = occlusion(model, self.row, self.baseline, fixed, 1)
        np.testing.assert_allclose(a, b, atol=1e-6)

    def test_unchanged_reference_coordinate_has_zero_ig(self):
        baseline = self.row.copy()
        baseline[0] = 0
        values, _ = integrated_gradients(self.model, encode(self.row[None], self.fm), encode(baseline[None], self.fm), steps=32)
        decoded = decode(values[:, 0], self.fm, 6)[0]
        np.testing.assert_array_equal(decoded[1:], np.zeros(5))

    def test_constant_magnitudes_are_not_perfect_agreement(self):
        result = compare(np.array([-1, 1, -1, 1, -1, 1]), np.arange(6))
        self.assertIsNone(result["absolute_spearman"])
        self.assertIsNone(result["top5_jaccard"])

    def test_signed_and_magnitude_agreement_differ(self):
        result = compare(np.arange(6), -np.arange(6))
        self.assertAlmostEqual(result["signed_spearman"], -1)
        self.assertAlmostEqual(result["absolute_spearman"], 1)
        self.assertEqual(result["top5_jaccard"], 1)

    def test_replacement_linear_sum_and_matched_random(self):
        predict = predictor(self.model, self.fm, 1)
        effects = self.coefficients * (self.row - self.baseline)
        config = {"seed": 22, "k": [1, 3, 5], "random_repeats": 100}
        output = replacement(predict, self.row, self.baseline, {"a": effects, "b": effects.copy()}, config, 2)
        repeat = replacement(predict, self.row, self.baseline, {"a": effects, "b": effects.copy()}, config, 2)
        self.assertEqual(output, repeat)
        for entry in output.values():
            a = entry["methods"]["a"]
            self.assertEqual(a, entry["methods"]["b"])
            self.assertAlmostEqual(a["score_drop"], float(effects[a["selected_indices"]].sum()), places=5)
            self.assertGreaterEqual(a["random_midrank_percentile"], 0)
            self.assertLessEqual(a["random_midrank_percentile"], 1)


RESULTS = ROOT / "experiments/08_som2img_attributions/results.json"


@unittest.skipUnless(RESULTS.exists(), "Run Experiment 08 to verify its saved evidence")
class SavedAttributionEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(RESULTS.read_text())

    def test_protocol_and_source_hashes(self):
        self.assertEqual(self.report["protocol_sha256"], sha256(PROTOCOL))
        for path, digest in self.report["source_hashes"].items():
            self.assertEqual(sha256(ROOT / path), digest)

    def test_complete_case_model_reference_inventory(self):
        self.assertEqual(len(self.report["runs"]), 21)
        self.assertEqual(len(self.report["selected_evaluation_positions"]), 12)
        for run in self.report["runs"]:
            self.assertEqual([c["position"] for c in run["cases"]], self.report["selected_evaluation_positions"])
            self.assertEqual(sum(c["label"] for c in run["cases"]), 6)
            for case in run["cases"]:
                self.assertEqual(case["target_sign"], 2 * case["label"] - 1)
                self.assertEqual(set(case["references"]), {"fit_median", "fit_minimum"})
                for ref in case["references"].values():
                    self.assertTrue(all(len(v) == 30 for v in ref["vectors"].values()))

    def test_pairwise_comparisons_recompute_exactly(self):
        self.assertEqual(pairwise_stability(self.report["runs"]), self.report["pairwise_stability"])
        pairs = self.report["pairwise_stability"]
        self.assertEqual(len(pairs), 1872)
        for axis in ("cnn_seed", "layout_seed"):
            for method in ("Fixed", "SnakeRandom", "SOM2IMG"):
                chosen = [p for p in pairs if p["axis"] == axis and p["method"] == method
                          and p["reference"] == "fit_median" and p["explainer"] == "integrated_gradients"]
                expected = (3 if method == "Fixed" else 9) if axis == "cnn_seed" else (0 if method == "Fixed" else 9)
                self.assertEqual(len({(p["left"], p["right"]) for p in chosen}), expected)
                self.assertEqual(len(chosen), expected * 12)

    def test_summary_recomputes_exactly(self):
        self.assertEqual(summarize(self.report["runs"], self.report["pairwise_stability"]), self.report["summary"])

    def test_random_percentiles_recompute_from_raw_drops(self):
        for run in self.report["runs"]:
            for case in run["cases"]:
                for reference in case["references"].values():
                    for entry in reference["replacement"].values():
                        drops = np.array(entry["random_drops"])
                        self.assertEqual(len(drops), 100)
                        for method in entry["methods"].values():
                            if method["eligible"]:
                                value = method["score_drop"]
                                expected = float(np.mean(drops < value) + 0.5 * np.mean(drops == value))
                                self.assertEqual(method["random_midrank_percentile"], expected)
                                self.assertEqual(method["excess_over_random_mean"], value - float(drops.mean()))

    @unittest.skipUnless((ROOT / "artifacts/07_som2img_layout").exists(), "Local checkpoints not present")
    def test_frozen_models_and_excluded_test_partition(self):
        _, manifest, frozen = load_frozen(json.loads(PROTOCOL.read_text()))
        self.assertEqual(len(frozen), 21)
        self.assertEqual(manifest["excluded_learning_test_count"], 86)


if __name__ == "__main__":
    unittest.main()
