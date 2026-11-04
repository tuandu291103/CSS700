"""Analytic and evidence checks for the model-free/reference follow-up."""

import json
import unittest

import numpy as np

from scripts.reference_controls import (
    PROTOCOL,
    midrank,
    outcome,
    prepare,
    random_subsets,
    ranking_control,
    reference_inventory,
    select_reference_positions,
    sensitivity_comparisons,
    stability_comparisons,
    summarize,
)
from scripts.wisconsin_learning import ROOT
from scripts.wisconsin_models import sha256


class ReferenceControlTests(unittest.TestCase):
    def test_reference_sampling_balanced_unique_and_deterministic(self):
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        selected = select_reference_positions(labels, 2, 91)
        self.assertEqual(selected, select_reference_positions(labels, 2, 91))
        self.assertEqual(len(set(selected)), 4)
        np.testing.assert_array_equal(labels[selected], [0, 0, 1, 1])

    def test_reference_sampling_rejects_insufficient_classes(self):
        for labels in ([0, 0, 0], [0, 1, 1]):
            with self.assertRaises(ValueError):
                select_reference_positions(labels, 2, 1)

    def test_reference_values_and_identity_from_fit_only(self):
        fit = np.arange(24, dtype=float).reshape(8, 3)
        data = {"fit": {"scaled": fit, "y": np.array([0, 1] * 4)}, "evaluation": {"scaled": np.ones((2, 3))}}
        manifest = {"partitions": {"fit": {"row_indices_zero_based": list(range(100, 108)), "sample_ids": list(range(200, 208))}}}
        config = {"per_class": 2, "seed": 91}
        refs = reference_inventory(data, manifest, config)
        data["evaluation"]["scaled"][:] = 1e10
        self.assertEqual(refs, reference_inventory(data, manifest, config))
        for ref in refs.values():
            if ref["kind"] == "real":
                p = ref["fit_position"]
                self.assertEqual(ref["original_row_index"], 100 + p)
                self.assertEqual(ref["sample_id"], 200 + p)
                np.testing.assert_array_equal(ref["values"], fit[p])

    def test_random_bank_is_deterministic_and_without_replacement(self):
        bank = random_subsets(30, 5, 100, 55, 4)
        np.testing.assert_array_equal(bank, random_subsets(30, 5, 100, 55, 4))
        self.assertTrue(all(len(set(row)) == 5 for row in bank))
        self.assertFalse(np.array_equal(bank, random_subsets(30, 5, 100, 55, 5)))

    def test_midrank_counts_ties(self):
        self.assertEqual(midrank(2, [1, 2, 2, 3]), 0.5)

    def test_outcome_uses_declared_numerical_tolerance(self):
        self.assertEqual(outcome(1, 1 + 5e-7), "tie")
        self.assertEqual(outcome(2, 1), "win")
        self.assertEqual(outcome(1, 2), "loss")

    def test_weighted_linear_model_outperforms_distance_ranking(self):
        row = np.array([10, 9, 2, 1, 0.5, 0.1])
        weights = np.array([0, 0, 10, 20, 0, 0])
        ref = np.zeros(6)
        result = ranking_control(lambda x: x @ weights, row, ref, row * weights, random_subsets(6, 2, 100, 1, 0))
        self.assertEqual(result["primary"]["outcome"], "win")
        self.assertEqual(result["rankings"]["ig_magnitude"]["absolute_score_change"], 40)
        self.assertEqual(result["rankings"]["distance"]["absolute_score_change"], 0)

    def test_absolute_joint_change_can_cancel_signed_effects(self):
        row = np.array([5, 5, 10, 9, 1, 2], dtype=float)
        weights = np.array([1, -1, 0.2, 0.2, 0, 0])
        result = ranking_control(lambda x: x @ weights, row, np.zeros(6), row * weights, random_subsets(6, 2, 100, 1, 0))
        self.assertEqual(result["primary"]["outcome"], "loss")
        self.assertEqual(result["secondary"]["outcome"], "win")
        self.assertAlmostEqual(result["rankings"]["ig_magnitude"]["absolute_score_change"], 0)

    def test_distance_selection_is_independent_of_model_weights(self):
        row = np.array([1, 4, 2, 5, 3, 6], dtype=float)
        bank = random_subsets(6, 2, 100, 1, 0)
        outputs = []
        for weights in (np.arange(6), -np.arange(6)):
            r = ranking_control(lambda x, w=weights: x @ w, row, np.zeros(6), row * weights, bank)
            outputs.append(r["rankings"]["distance"]["selected_indices"])
        self.assertEqual(outputs[0], outputs[1])

    def test_constant_ig_is_ineligible_not_a_win(self):
        row = np.arange(6, dtype=float)
        r = ranking_control(lambda x: x.sum(axis=1), row, np.zeros(6), np.zeros(6), random_subsets(6, 2, 100, 1, 0))
        self.assertFalse(r["primary"]["eligible"])
        self.assertNotIn("outcome", r["primary"])

    def test_rank_ties_use_original_feature_order(self):
        row = np.array([4, 4, 3, 2, 1, 0], dtype=float)
        r = ranking_control(lambda x: x.sum(axis=1), row, np.zeros(6), row, random_subsets(6, 1, 10, 1, 0))
        self.assertEqual(r["rankings"]["distance"]["selected_indices"], [0])

    def test_control_rejects_nonfinite_vectors(self):
        with self.assertRaises(ValueError):
            ranking_control(lambda x: x.sum(axis=1), np.array([1, 2, np.nan]), np.zeros(3), np.ones(3), random_subsets(3, 1, 10, 1, 0))


RESULTS = ROOT / "experiments/09_reference_controls/results.json"


@unittest.skipUnless(RESULTS.exists(), "Run Experiment 09 for saved-evidence checks")
class ReferenceEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(RESULTS.read_text())

    def test_protocol_and_source_hashes(self):
        self.assertEqual(sha256(PROTOCOL), self.report["protocol_sha256"])
        for path, digest in self.report["source_hashes"].items():
            self.assertEqual(sha256(ROOT / path), digest)

    def test_full_inventory(self):
        r = self.report
        self.assertEqual(len(r["runs"]), 21)
        self.assertEqual(len(r["references"]), 6)
        self.assertEqual(sorted(v["label"] for v in r["references"].values() if v["kind"] == "real"), [0, 0, 1, 1])
        for run in r["runs"]:
            self.assertEqual([c["position"] for c in run["cases"]], r["selected_evaluation_positions"])
            for case in run["cases"]:
                self.assertEqual(case["target_sign"], 2 * case["label"] - 1)
                self.assertEqual(set(case["references"]), set(r["references"]))

    def test_old_vectors_are_reused_unchanged(self):
        old = json.loads((ROOT / self.report["protocol"]["attribution_results"]).read_text())
        for run, previous in zip(self.report["runs"], old["runs"], strict=True):
            self.assertEqual(run["key"], previous["key"])
            for case, prior in zip(run["cases"], previous["cases"], strict=True):
                for name in ("fit_median", "fit_minimum"):
                    self.assertEqual(case["references"][name]["ig"], prior["references"][name]["vectors"]["integrated_gradients"])

    def test_all_comparisons_and_summaries_recompute(self):
        r = self.report
        self.assertEqual(stability_comparisons(r["runs"]), r["seed_stability"])
        self.assertEqual(sensitivity_comparisons(r["runs"], r["references"]), r["reference_sensitivity"])
        self.assertEqual(len(r["seed_stability"]), 2808)
        self.assertEqual(len(r["reference_sensitivity"]), 2520)
        self.assertEqual(summarize(r["runs"], r["references"], r["seed_stability"], r["reference_sensitivity"]), r["summary"])

    def test_raw_drops_reproduce_percentiles_and_paired_outcomes(self):
        for run in self.report["runs"]:
            for case in run["cases"]:
                for ref in case["references"].values():
                    control = ref["control"]
                    random = np.array(control["random_signed_drops"])
                    self.assertEqual(len(random), 100)
                    for ranking in control["rankings"].values():
                        if ranking["eligible"]:
                            self.assertEqual(ranking["absolute_random_percentile"], midrank(abs(ranking["signed_score_drop"]), np.abs(random)))
                            self.assertEqual(ranking["signed_random_percentile"], midrank(ranking["signed_score_drop"], random))
                    if control["primary"]["eligible"]:
                        a, b = (control["rankings"][k]["absolute_score_change"] for k in ("ig_magnitude", "distance"))
                        self.assertEqual(control["primary"]["outcome"], outcome(a, b))

    @unittest.skipUnless((ROOT / "artifacts/07_som2img_layout").exists(), "Local checkpoint verification requires artifacts")
    def test_reference_selection_is_disjoint_from_evaluation(self):
        _, _, manifest, frozen, references = prepare(json.loads(PROTOCOL.read_text()))
        self.assertEqual(references, self.report["references"])
        self.assertEqual(len(frozen), 21)
        rows = {r["original_row_index"] for r in references.values() if r["kind"] == "real"}
        fit = set(manifest["partitions"]["fit"]["row_indices_zero_based"])
        evaluation = set(manifest["partitions"]["evaluation"]["row_indices_zero_based"])
        self.assertTrue(rows <= fit)
        self.assertFalse(rows & evaluation)


if __name__ == "__main__":
    unittest.main()
