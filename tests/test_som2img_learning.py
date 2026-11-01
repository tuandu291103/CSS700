"""Offline controls for the layout exercise; no models fitted to research test data."""

import json
import unittest

import numpy as np

from scripts.som2img_learning import (
    DEFAULT_UPSTREAM,
    PROTOCOL_PATH,
    decode,
    edges,
    encoding_audit,
    import_upstream,
    layout_statistics,
    positions,
    prepare_data,
)
from scripts.wisconsin_models import metrics, sha256


class LayoutAnalyticTests(unittest.TestCase):
    def test_inverse_round_trip(self):
        fm = np.array([[2, 0], [1, -1]])
        values = np.array([[0.0, -0.2, 1.2], [0.9, 0.5, 0.1]])
        images = np.stack([values[:, 2], values[:, 0], values[:, 1], np.zeros(2)], axis=1).reshape(2, 2, 2)
        np.testing.assert_array_equal(decode(images, fm, 3), values)

    def test_reject_duplicate_missing_or_extra_features(self):
        for fm in (np.array([[0, 0], [1, 2]]), np.array([[0, 1], [-1, -1]]), np.array([[0, 1], [2, 3]])):
            with self.subTest(fm=fm), self.assertRaises(ValueError):
                positions(fm, 3)

    def test_neighbor_edges_are_unique_and_orthogonal(self):
        self.assertEqual(edges(np.array([[0, 1], [2, 3]])), {(0, 1), (0, 2), (1, 3), (2, 3)})
        self.assertEqual(len(edges(np.arange(30).reshape(5, 6))), 49)

    def test_identical_profiles_have_zero_neighbor_difference(self):
        x = np.tile(np.linspace(0, 1, 20)[:, None], (1, 30))
        result = layout_statistics(np.arange(30).reshape(5, 6), x)
        self.assertEqual(result["adjacent_profile_rmse"], 0)
        self.assertAlmostEqual(result["adjacent_absolute_pearson"], 1)
        self.assertEqual(result["same_measurement_pair_count"], 30)

    def test_partition_and_scaler_boundary(self):
        protocol = json.loads(PROTOCOL_PATH.read_text())
        data, manifest, scaler = prepare_data(protocol)
        self.assertEqual([len(data[p]["y"]) for p in ("fit", "inner_stop", "evaluation")], [318, 79, 86])
        groups = [set(p["row_indices_zero_based"]) for p in manifest["partitions"].values()]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        np.testing.assert_array_equal(scaler.data_min_, data["fit"]["raw"].min(axis=0))
        np.testing.assert_array_equal(scaler.data_max_, data["fit"]["raw"].max(axis=0))
        self.assertEqual(manifest["excluded_learning_test_count"], 86)


@unittest.skipUnless((DEFAULT_UPSTREAM / ".git").exists(), "Pinned author checkout not present; see Experiment 07 instructions")
class AuthorIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        protocol = json.loads(PROTOCOL_PATH.read_text())
        cls.t2i, _ = import_upstream(DEFAULT_UPSTREAM, protocol["upstream_commit"])

    def test_author_layout_preserves_features_and_frozen_scaler(self):
        rng = np.random.default_rng(7)
        fit = rng.normal(size=(40, 6))
        fitted = self.t2i.create("SOM2IMG", image_shape=(2, 3), n_iterations=100).fit(fit)
        lo = fitted.value_scaler_.data_min_.copy()
        fm = fitted.feature_map_.copy()
        probe = fit[:3] + 100
        recovered = decode(fitted.transform(probe), fm, 6)
        np.testing.assert_array_equal(recovered, fitted.value_scaler_.transform(probe))
        np.testing.assert_array_equal(fitted.feature_map_, fm)
        np.testing.assert_array_equal(fitted.value_scaler_.data_min_, lo)

    def test_fit_does_not_use_labels(self):
        x = np.random.default_rng(8).random((40, 6))
        a = self.t2i.create("SOM2IMG", image_shape=(2, 3), n_iterations=100).fit(x)
        b = self.t2i.create("SOM2IMG", image_shape=(2, 3), n_iterations=100).fit(x, np.arange(40) % 2)
        np.testing.assert_array_equal(a.feature_map_, b.feature_map_)

    def test_default_wrap_clipping_and_float_modes(self):
        values = np.tile(np.linspace(0, 1, 30), (2, 1))
        values[0, 0], values[1, 1] = 1.02, -0.02
        audit = encoding_audit(self.t2i, values, ["synthetic_a", "synthetic_b"])
        self.assertEqual(audit["outside_training_range"], 2)
        self.assertEqual(audit["default_vs_clipped_difference_over_half_range"], 2)
        self.assertAlmostEqual(audit["synthetic_1_02_default"], 4 / 255)
        self.assertEqual(audit["synthetic_1_02_clipped"], 1)
        self.assertLess(audit["float32_max_absolute_error"], 1e-6)


@unittest.skipUnless((PROTOCOL_PATH.parent / "results.json").exists(), "Experiment not yet run")
class SavedLayoutEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads((PROTOCOL_PATH.parent / "results.json").read_text())

    def test_protocol_hash_and_all_crossed_seeds(self):
        r = self.result
        self.assertEqual(r["protocol_sha256"], sha256(PROTOCOL_PATH))
        self.assertEqual(len(r["layouts"]), 7)
        self.assertEqual(len(r["runs"]), 21)
        for layout in r["layouts"]:
            self.assertEqual(sorted(run["cnn_seed"] for run in r["runs"] if run["layout_key"] == layout["key"]),
                             r["protocol"]["cnn_seeds"])

    def test_coverage_and_round_trip(self):
        for layout in self.result["layouts"]:
            positions(np.array(layout["feature_map"]), 30)
            self.assertEqual(layout["round_trip_max_abs"], {"fit": 0.0, "inner_stop": 0.0, "evaluation": 0.0})

    def test_metrics_recompute_and_checkpoint_selection(self):
        y = self.result["data"]["partitions"]["evaluation"]["labels"]
        for run in self.result["runs"]:
            self.assertEqual(metrics(y, run["probability_m"]), run["evaluation"])
            diag = run["diagnostics"]
            losses = [h["validation_loss"] for h in diag["history"]]
            self.assertEqual(diag["best_epoch"], losses.index(min(losses)) + 1)
            self.assertEqual(diag["parameters"], 1282)

    def test_every_run_in_summary(self):
        for method, summary in self.result["summary"].items():
            selected = [r for r in self.result["runs"] if r["method"] == method]
            self.assertEqual(summary["n_models"], len(selected))
            for metric, scores in summary["metrics"].items():
                self.assertEqual(scores["mean"], float(np.mean([r["evaluation"][metric] for r in selected])))


if __name__ == "__main__":
    unittest.main()
