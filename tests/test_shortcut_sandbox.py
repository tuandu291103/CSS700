"""Check the controlled intervention itself, not only attractive model results."""

import json
import unittest

import numpy as np

from scripts.shortcut_sandbox import (
    PROTOCOL,
    array_digest,
    attribution_mass,
    generate_partition,
    paired_diagnostic,
    summarize,
)
from scripts.wisconsin_learning import ROOT
from scripts.wisconsin_models import metrics, sha256


class ShortcutTests(unittest.TestCase):
    def test_generator_determinism_and_distinct_partition_seeds(self):
        first, repeated, other = (generate_partition(32, seed) for seed in (7, 7, 8))
        np.testing.assert_array_equal(first["images"]["correlated"], repeated["images"]["correlated"])
        self.assertNotEqual(array_digest(first["images"]["correlated"]), array_digest(other["images"]["correlated"]))

    def test_paired_views_only_change_background(self):
        part = generate_partition(32, 9)
        masks = part["masks"]
        for name, image in part["images"].items():
            np.testing.assert_array_equal(image[masks], part["foreground"][masks])
            np.testing.assert_array_equal(image[~masks], part["backgrounds"][name][~masks])
        self.assertTrue(np.all(part["images"]["correlated"][~masks] != part["images"]["reversed"][~masks]))
        np.testing.assert_array_equal(masks.sum(axis=(1, 2, 3)), np.full(32, 12))

    def test_shape_orientation_matches_label(self):
        part = generate_partition(32, 10)
        for mask, label in zip(part["masks"], part["labels"], strict=True):
            rows, columns = np.nonzero(mask[0])
            extent = (np.ptp(rows) + 1, np.ptp(columns) + 1)
            self.assertEqual(extent, (2, 6) if label == 0 else (6, 2))

    def test_independent_backgrounds_exactly_balanced_within_class(self):
        part = generate_partition(32, 11)
        for label in (0, 1):
            colors = part["background_labels"]["independent"][part["labels"] == label]
            np.testing.assert_array_equal(np.bincount(colors), [8, 8])
        np.testing.assert_array_equal(part["background_labels"]["correlated"], part["labels"])

    def test_generator_rejects_invalid_size(self):
        for count in (0, 3, 6):
            with self.assertRaises(ValueError):
                generate_partition(count, 1)

    def test_paired_metric_uses_same_rows_and_true_class(self):
        result = paired_diagnostic(np.array([0, 1]), np.array([0.1, 0.9]), np.array([0.9, 0.1]))
        self.assertEqual(result["prediction_flip_rate"], 1)
        self.assertAlmostEqual(result["mean_true_class_probability_drop"], 0.8)

    def test_mass_zero_map_and_area_accounting(self):
        mask = np.array([True, False, False])
        self.assertIsNone(attribution_mass(np.zeros(3), mask)["foreground_fraction"])
        result = attribution_mass(np.array([-2, 1, -1]), mask)
        self.assertEqual(result["foreground_fraction"], 0.5)
        self.assertEqual(result["foreground_mean_absolute_per_pixel"], 2)
        self.assertEqual(result["background_mean_absolute_per_pixel"], 1)

    def test_saved_metrics_hashes_and_seed_coverage(self):
        report = json.loads((PROTOCOL.parent / "results.json").read_text())
        self.assertEqual(report["protocol_sha256"], sha256(PROTOCOL))
        self.assertEqual(report["source_code_sha256"], sha256(ROOT / "scripts/shortcut_sandbox.py"))
        self.assertEqual(report["attribution_helper_sha256"], sha256(ROOT / "scripts/wisconsin_explanations.py"))
        self.assertEqual(report["summary"], summarize(report["runs"]))
        labels = np.array(report["test_labels"])
        self.assertEqual(len(report["runs"]), 6)
        for regime in ("correlated", "independent"):
            self.assertEqual([r["seed"] for r in report["runs"] if r["training_regime"] == regime], [7201, 7202, 7203])
        for run in report["runs"]:
            probabilities = {name: np.array(values) for name, values in run["probability_vertical"].items()}
            for view, values in probabilities.items():
                self.assertEqual(metrics(labels, values), run["test_metrics"][view])
            self.assertEqual(paired_diagnostic(labels, probabilities["correlated"], probabilities["reversed"]),
                             run["paired_background_change"])
            losses = [entry["validation_loss"] for entry in run["diagnostics"]["history"]]
            self.assertEqual(min(losses), run["diagnostics"]["selected_validation_loss"])
            for case in run.get("explanations", []):
                # IG's input-minus-reference factor is zero outside the mask for this baseline.
                values = np.array(case["references"]["background_only"]["attribution"])
                mask = np.array(case["foreground_mask"])
                np.testing.assert_array_equal(values[~mask], np.zeros((~mask).sum()))
                self.assertEqual(case["references"]["background_only"]["mass"]["foreground_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
