"""Genomic pilot controls on synthetic inputs and saved aggregate evidence."""

import json
import unittest
from pathlib import Path

import numpy as np

from scripts.audit_tcga_brca import sha256
from scripts.prepare_tcga_pilot import make_splits, select_scale, validate_splits
from scripts.tcga_pilot_models import class_weights, fit_cnn, prediction_metrics

ROOT = Path(__file__).resolve().parents[1]


def synthetic_rows():
    rows = []
    for i in range(40):
        site = "BH" if i % 4 == 0 else "AA"
        participant = f"TCGA-{site}-{i:04d}"
        for code in (["01", "11"] if i < 20 else ["01"]):
            rows.append({"participant": participant, "case_id": participant + "uuid",
                         "source_site": site, "code": code})
    return rows


class TcgaPilotTests(unittest.TestCase):
    def test_patient_split_is_complete_and_grouped(self):
        rows = synthetic_rows()
        splits = make_splits(rows, 12)
        validate_splits(rows, splits)
        self.assertEqual(sum(map(len, splits.values())), len(rows))
        assignments = {}
        for part, ix in splits.items():
            for i in ix:
                patient = rows[i]["participant"]
                self.assertEqual(assignments.setdefault(patient, part), part)

    def test_seed_reproduces_partitions(self):
        rows = synthetic_rows()
        a, b = make_splits(rows, 12), make_splits(rows, 12)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        self.assertFalse(np.array_equal(a["test"], make_splits(rows, 13)["test"]))

    def test_site_holdout_excludes_entire_site(self):
        rows = synthetic_rows()
        splits = make_splits(rows, 12, "BH")
        self.assertEqual({rows[i]["source_site"] for i in splits["test"]}, {"BH"})
        self.assertNotIn("BH", {rows[i]["source_site"] for i in splits["train"]})

    def test_case_uuid_overlap_rejected(self):
        rows = synthetic_rows()
        splits = make_splits(rows, 12)
        rows[splits["test"][0]]["case_id"] = rows[splits["train"][0]]["case_id"]
        with self.assertRaisesRegex(ValueError, "case_id overlap"):
            validate_splits(rows, splits)

    def test_source_site_mismatch_rejected(self):
        rows = synthetic_rows()
        rows[0]["source_site"] = "XX"
        with self.assertRaisesRegex(ValueError, "site metadata conflict"):
            make_splits(rows, 12)

    def test_selection_and_scaler_unchanged_by_heldout_extremes(self):
        raw = np.array([[0, 0, 1], [2, 1, 1], [4, 2, 1], [1, 3, 1.0]])
        a = select_scale(raw, ["g1", "g2", "g3"], [0, 1, 2], 2)
        raw[3] = [1e9, -1e9, 1e12]
        b = select_scale(raw, ["g1", "g2", "g3"], [0, 1, 2], 2)
        for i in (0, 2, 3, 4):
            np.testing.assert_array_equal(a[i], b[i])
        np.testing.assert_array_equal(a[1][:3], b[1][:3])
        np.testing.assert_array_equal(a[0], [0, 1])

    def test_variance_ties_resolved_by_identifier(self):
        raw = np.array([[0, 0, 0], [1, 1, 1.0]])
        chosen, *_ = select_scale(raw, ["g3", "g1", "g2"], [0, 1], 2)
        np.testing.assert_array_equal(chosen, [1, 2])

    def test_scaled_float32_inputs_with_float64_diagnostic_reductions(self):
        raw = np.random.default_rng(12).normal(7, 3, size=(1000, 256))
        _, x, *_ = select_scale(raw, [f"g{i}" for i in range(256)], np.arange(800), 256)
        checked = x[:800].astype(np.float64)
        np.testing.assert_allclose(checked.mean(axis=0), 0, atol=1e-7)
        np.testing.assert_allclose(checked.std(axis=0), 1, atol=1e-7)

    def test_constant_training_genes_do_not_fill_budget(self):
        with self.assertRaisesRegex(ValueError, "Too few"):
            select_scale(np.ones((4, 5)), list("abcde"), [0, 1], 2)

    def test_float32_grid_roundtrip_exact(self):
        x = np.random.default_rng(12).normal(size=(7, 256)).astype(np.float32)
        np.testing.assert_array_equal(x.reshape(7, 1, 16, 16).reshape(7, 256), x)

    def test_class_weights_training_only_and_balanced(self):
        weights = class_weights(np.array([0, 1, 1, 1]))
        np.testing.assert_allclose(weights, [2, 2 / 3])
        with self.assertRaises(ValueError):
            class_weights(np.array([1, 1]))

    def test_majority_accuracy_does_not_hide_normal_recall(self):
        scores = prediction_metrics(np.array([0, 1, 1, 1]), np.full(4, 0.75))
        self.assertEqual(scores["accuracy"], 0.75)
        self.assertEqual(scores["balanced_accuracy"], 0.5)
        self.assertEqual(scores["recall_normal"], 0)
        self.assertEqual(scores["confusion_matrix"], [[0, 1], [0, 3]])

    def test_cnn_determinism_and_weighted_checkpoint_selection(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(8, 1, 4, 4)).astype(np.float32)
        y = np.array([0, 1, 1, 1, 0, 1, 1, 1], dtype=np.int64)
        config = {"learning_rate": 0.003, "weight_decay": 0.0001,
                  "max_epochs": 3, "patience": 2, "batch_size": 4}
        _, a = fit_cnn(x, y, x, y, config, 12)
        _, b = fit_cnn(x, y, x, y, config, 12)
        self.assertEqual(a, b)
        self.assertEqual(a["parameters"], 1282)
        self.assertEqual(a["selected_validation_weighted_loss"], min(h["validation_weighted_loss"] for h in a["history"]))

    def test_saved_results_integrity(self):
        directory = ROOT / "experiments/06_tcga_brca_pilot"
        if not (directory / "results.json").exists():
            self.skipTest("Pilot has not been run yet")
        results = json.loads((directory / "results.json").read_text())
        prepared = json.loads((directory / "preparation.json").read_text())
        self.assertEqual(results["protocol_sha256"], sha256(directory / "protocol.json"))
        self.assertEqual(results["preparation_sha256"], sha256(directory / "preparation.json"))
        self.assertEqual(results["model_code_sha256"], sha256(ROOT / "scripts/tcga_pilot_models.py"))
        self.assertEqual(prepared["preparation_code_sha256"], sha256(ROOT / "scripts/prepare_tcga_pilot.py"))
        for scenario, description in results["scenarios"].items():
            self.assertEqual(len(description["runs"]), 8)
            for run in description["runs"]:
                for part, result in run["metrics"].items():
                    matrix = np.array(result["confusion_matrix"])
                    self.assertEqual(result["total"], prepared["scenarios"][scenario]["partitions"][part]["samples"])
                    self.assertEqual(result["correct"], int(matrix.trace()))
                    self.assertEqual(result["total"], int(matrix.sum()))
                    self.assertAlmostEqual(result["balanced_accuracy"], np.mean(matrix.diagonal() / matrix.sum(axis=1)))


if __name__ == "__main__":
    unittest.main()
