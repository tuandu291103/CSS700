"""Synthetic integrity checks; no model fitting and no held-out outcome use."""

import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import download_tcga_expression as downloader
from scripts.audit_tcga_brca import (
    barcode,
    join_clinical,
    missing,
    read_table,
    review_cohort_options,
    scan_expression,
    select_vials,
    sha256,
    source_count_reconciliation,
)


def clinical_row(sample, **changes):
    participant, code = barcode(sample)
    result = {"submitter_id": participant, "case_id": participant + "-uuid",
              "sample_type_id.samples": code,
              "sample_type.samples": {"01": "Primary Tumor", "11": "Solid Tissue Normal"}[code]}
    for field in ("code.tissue_source_site", "preservation_method.samples", "is_ffpe.samples",
                  "prior_treatment.diagnoses", "category.annotations", "annotations.samples",
                  "notes.annotations"):
        result[field] = ""
    return result | changes


class TcgaAuditTests(unittest.TestCase):
    def scan(self, content, block_size=256):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expression.tsv.gz"
            with gzip.open(path, "wt") as output:
                output.write(content)
            return scan_expression(path, block_size=block_size)

    def test_barcode_levels_and_rejection(self):
        self.assertEqual(barcode("TCGA-AB-1234-11A"), ("TCGA-AB-1234", "11"))
        for value in ("TCGA-AB-1234", "TCGA-AB-1234-11", "TCGA-AB-1234-11A-extra", "bad"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                barcode(value)

    def test_missing_is_not_zero(self):
        self.assertTrue(missing(" Not Reported "))
        self.assertTrue(missing(""))
        self.assertFalse(missing("0"))
        self.assertFalse(missing("False"))

    def test_full_numeric_scan_and_constants(self):
        _, genes, stats, _ = self.scan(
            "id\tTCGA-AB-1234-01A\tTCGA-CD-5678-11A\ng1\t0\t0\ng2\t2\t3\n"
        )
        self.assertEqual(genes, ["g1", "g2"])
        self.assertEqual(stats["values_checked"], 4)
        self.assertEqual(stats["all_zero_features"], 1)
        self.assertEqual(stats["finite_maximum"], 3)

    def test_missing_infinity_and_negative_are_visible(self):
        _, _, stats, _ = self.scan(
            "id\tTCGA-AB-1234-01A\tTCGA-CD-5678-11A\ng1\tNA\tinf\ng2\t-1\t0\n"
        )
        self.assertEqual(stats["missing_values"], 1)
        self.assertEqual(stats["infinite_values"], 1)
        self.assertEqual(stats["negative_values"], 1)
        self.assertEqual(stats["samples_with_nonfinite_values"], 2)

    def test_bad_expression_shape_and_ids_fail(self):
        examples = [
            "id\tTCGA-AB-1234-01A\ng1\t2\t3\n",
            "id\tTCGA-AB-1234-01A\ng1\t2\ng1\t3\n",
            "id\tTCGA-AB-1234-01A\tTCGA-AB-1234-01A\ng1\t2\t3\n",
            "id\tTCGA-AB-1234-01A\ng1\tnot-a-number\n",
        ]
        for text in examples:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.scan(text)

    def test_profile_duplicates_independent_of_block_size(self):
        text = "id\tTCGA-AB-1234-01A\tTCGA-CD-5678-11A\ng1\t2\t2\ng2\t3\t3\n"
        one, two = self.scan(text, 1), self.scan(text, 3)
        self.assertEqual(one, two)
        self.assertEqual(one[2]["cross_participant_duplicate_profile_groups"], 1)

    def test_exact_join_not_row_order_and_no_silent_missing(self):
        a, b = "TCGA-AB-1234-01A", "TCGA-CD-5678-11A"
        manifest, conflicts, _ = join_clinical([a, b], {b: clinical_row(b), a: clinical_row(a)})
        self.assertEqual([r["sample"] for r in manifest], [a, b])
        self.assertFalse(conflicts)
        _, conflicts, _ = join_clinical([a, b], {a: clinical_row(a)})
        self.assertEqual(conflicts[0]["reason"], "missing_clinical")

    def test_mismatched_identity_or_label_is_flagged(self):
        a = "TCGA-AB-1234-01A"
        bad = clinical_row(a, submitter_id="TCGA-CD-5678")
        bad["sample_type.samples"] = "Solid Tissue Normal"
        manifest, conflicts, _ = join_clinical([a], {a: bad})
        self.assertFalse(manifest[0]["identity_valid"])
        self.assertEqual(len(conflicts[0]["reason"]), 2)

    def test_case_participant_bijection(self):
        a, b = "TCGA-AB-1234-01A", "TCGA-CD-5678-11A"
        _, _, result = join_clinical([a, b], {
            a: clinical_row(a, case_id="same-case"), b: clinical_row(b, case_id="same-case")})
        self.assertEqual(result["case_ids_with_multiple_participants"], 1)

    def test_vial_selection_preserves_tumor_normal_pair(self):
        ids = ["TCGA-AB-1234-01B", "TCGA-AB-1234-11A", "TCGA-AB-1234-01A"]
        manifest, _, _ = join_clinical(ids, {s: clinical_row(s) for s in ids})
        selected, excluded = select_vials(manifest)
        self.assertEqual([r["sample"] for r in selected], [ids[2], ids[1]])
        self.assertEqual(excluded, [ids[0]])

    def test_duplicate_or_ragged_metadata_rejected(self):
        for text in ("id\tgene\ng1\ta\ng1\tb\n", "id\tgene\ng1\n"):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "map.tsv"
                path.write_text(text)
                with self.assertRaises(ValueError):
                    read_table(path, "id", ["id", "gene"])

    def test_count_reconciliation_requires_exact_identifier_digest(self):
        review = {"listed_unique_ensembl_identifiers": 2, "listed_total_identifiers": 3,
                  "listed_non_expression_identifiers": ["sampleID"],
                  "sorted_ensembl_identifiers_sha256": hashlib.sha256(b"g1\ng2\n").hexdigest()}
        self.assertTrue(source_count_reconciliation(["g2", "g1"], review)["matches_full_browser_inventory"])
        self.assertFalse(source_count_reconciliation(["g1", "wrong"], review)["matches_full_browser_inventory"])

    def test_case_annotation_propagates_from_clinical_only_row(self):
        tumor, normal = "TCGA-AB-1234-01A", "TCGA-AB-1234-11A"
        clinical = {s: clinical_row(s, **{"prior_treatment.diagnoses": "No"}) for s in (tumor, normal)}
        clinical[normal]["notes.annotations"] = "case-level warning"
        manifest, _, _ = join_clinical([tumor], clinical)
        summary, private = review_cohort_options(manifest, clinical)
        self.assertEqual(summary["sample_count"], 0)
        self.assertEqual(private["metadata_exclusion_reasons"][tumor], ["case_annotation"])

    def test_sample_flags_applied_before_vial_choice(self):
        ids = ["TCGA-AB-1234-01A", "TCGA-AB-1234-01B", "TCGA-AB-1234-11A"]
        clinical = {s: clinical_row(s, **{"prior_treatment.diagnoses": "No"}) for s in ids}
        clinical[ids[0]].update({"preservation_method.samples": "FFPE", "is_ffpe.samples": "False"})
        manifest, _, _ = join_clinical(ids, clinical)
        summary, private = review_cohort_options(manifest, clinical)
        self.assertEqual([r["sample"] for r in private["samples"]], ids[1:])
        self.assertEqual(summary["participant_type_patterns"], {"01+11": 1})
        self.assertEqual(summary["initial_candidate_diagnostics"]["preservation_FFPE_but_ffpe_flag_False"], 1)

    def test_unknown_treatment_excluded_but_unknown_preservation_retained(self):
        a, b = "TCGA-AB-1234-01A", "TCGA-CD-5678-11A"
        clinical = {s: clinical_row(s, **{"preservation_method.samples": "Unknown"}) for s in (a, b)}
        clinical[b]["prior_treatment.diagnoses"] = "No"
        manifest, _, _ = join_clinical([a, b], clinical)
        summary, private = review_cohort_options(manifest, clinical)
        self.assertEqual(summary["sample_count"], 1)
        self.assertEqual(private["samples"][0]["sample"], b)
        self.assertEqual(private["metadata_exclusion_reasons"][a], ["prior_treatment_not_documented_No"])

    def test_saved_audit_integrity_and_no_training_claim(self):
        root = Path(__file__).resolve().parents[1]
        result = json.loads((root / "experiments/05_tcga_brca_audit/results.json").read_text())
        self.assertEqual(result["audit_script_sha256"], sha256(root / "scripts/audit_tcga_brca.py"))
        self.assertEqual(result["protocol_sha256"], sha256(root / "experiments/05_tcga_brca_audit/protocol.json"))
        self.assertEqual(result["source_count_review_sha256"], sha256(root / "experiments/05_tcga_brca_audit/source_count_review.json"))
        self.assertEqual(result["expression"]["values_checked"], result["expression"]["features"] * result["expression"]["samples"])
        self.assertTrue(result["source_count_reconciliation"]["matches_full_browser_inventory"])
        self.assertFalse(result["models_trained"])
        self.assertFalse(result["training_ready_without_further_review"])
        for name in ("candidate", "conservative_cohort_option"):
            cohort = result[name]
            self.assertEqual(cohort["sample_count"], sum(cohort["class_counts"].values()))
            self.assertEqual(cohort["participant_count"], sum(cohort["participant_type_patterns"].values()))

    def test_download_ranges_have_distinct_headers_and_preserve_existing_bytes(self):
        def curl(command, check):
            self.assertTrue(check)
            start, end = map(int, command[command.index("--range") + 1].split("-"))
            headers = Path(command[command.index("--dump-header") + 1])
            headers.write_text(f"HTTP/1.1 206 Partial Content\nContent-Range: bytes {start}-{end}/{downloader.SIZE}\n")
            if "--head" not in command:
                Path(command[command.index("--output") + 1]).write_bytes(b"x" * (end - start + 1))

        with tempfile.TemporaryDirectory() as directory, patch.object(downloader.subprocess, "run", curl):
            root = Path(directory)
            a = downloader.download_range(root, 0, 2)
            b = downloader.download_range(root, 3, 5)
            self.assertTrue(Path(str(a) + ".headers").exists())
            self.assertTrue(Path(str(b) + ".headers").exists())
            a.write_bytes(b"abc")
            self.assertEqual(downloader.download_range(root, 0, 2).read_bytes(), b"abc")

    def test_download_wrong_range_header_fails(self):
        def curl(command, check):
            Path(command[command.index("--dump-header") + 1]).write_text("HTTP/1.1 200 OK\n")
            Path(command[command.index("--output") + 1]).write_bytes(b"abc")

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(downloader.subprocess, "run", curl),
            self.assertRaises(ValueError),
        ):
            downloader.download_range(Path(directory), 0, 2)


if __name__ == "__main__":
    unittest.main()
