"""Synthetic tests for the audit only; these do not test a training pipeline."""

import unittest

from scripts.audit_mile_metadata import (
    audit_annotation, audit_profile, audit_samples, gene_ids, parse_samples, single,
)


def synthetic_samples():
    series = ["^SERIES = GSE13159"]
    samples = []
    for index in range(1, 2097):
        accession = f"GSM{index}"
        series.append(f"!Series_sample_id = {accession}")
        samples.extend([
            f"^SAMPLE = {accession}",
            f"!Sample_geo_accession = {accession}",
            f"!Sample_title = MILES stage 1 data N1_{index:04d}",
            "!Sample_organism_ch1 = Homo sapiens",
            "!Sample_characteristics_ch1 = leukemia class: synthetic class",
            "!Sample_characteristics_ch1 = sample type: bone marrow",
            "!Sample_platform_id = GPL570",
            "!Sample_data_row_count = 54675",
            "!Sample_source_name_ch1 = synthetic source",
            "!Sample_data_processing = synthetic processing",
            "!Sample_treatment_protocol_ch1 = synthetic treatment",
            "!Sample_series_id = GSE13159",
        ])
    return "\n".join(series), "\n".join(samples)


class MetadataAuditTests(unittest.TestCase):
    def profile_fixture(self):
        text = "\n".join([
            "^SAMPLE = GSM1", "!Sample_geo_accession = GSM1",
            "!Sample_title = synthetic", "!Sample_platform_id = GPL570",
            "!Sample_characteristics_ch1 = leukemia class: synthetic",
            "!Sample_data_row_count = 2", "#VALUE = DQN3 synthetic description",
            "!sample_table_begin", "ID_REF\tVALUE_DS\tVALUE",
            "AFFX-control\t100\t", "probe1\t500\t0.25", "!sample_table_end",
        ])
        return text, parse_samples(text), {"AFFX-control", "probe1"}

    def test_profile_missing_values_are_not_zero_imputed(self):
        result = audit_profile(*self.profile_fixture())
        stats = result["column_statistics"]["VALUE"]
        self.assertEqual((stats["finite_count"], stats["missing_count"]), (1, 1))
        self.assertEqual(stats["minimum"], 0.25)
        self.assertEqual(stats["missing_non_AFFX_count"], 0)
        self.assertTrue(result["annotation_probe_set_exact_match"])

    def test_profile_rejects_nonfinite_and_malformed_data(self):
        text, expected, ids = self.profile_fixture()
        for changed in (text.replace("0.25", "nan"), text.replace("0.25", "inf"),
                        text.replace("probe1\t500", "AFFX-control\t500"),
                        text.replace("!sample_table_end", ""),
                        text.replace("probe1\t500\t0.25", "probe1\t500")):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                audit_profile(changed, expected, ids)

    def test_profile_rejects_wrong_identity_and_annotation(self):
        text, expected, ids = self.profile_fixture()
        with self.assertRaises(ValueError):
            audit_profile(text.replace("GSM1", "GSM2"), expected, ids)
        with self.assertRaises(ValueError):
            audit_profile(text.replace("!Sample_title = synthetic", "!Sample_title = changed"), expected, ids)
        with self.assertRaises(ValueError):
            audit_profile(text, expected, {"AFFX-control", "different_probe"})

    def test_gene_ids_preserve_ambiguity_and_deduplicate(self):
        self.assertEqual(gene_ids(" 780 /// 780 /// 100616237 "), {"780", "100616237"})
        self.assertEqual(gene_ids("---"), set())
        self.assertEqual(gene_ids(""), set())

    def test_invalid_gene_id_rejected(self):
        with self.assertRaises(ValueError):
            gene_ids("DDR1")

    def test_html_response_rejected(self):
        with self.assertRaises(ValueError):
            parse_samples("<html>Check your browser</html>")

    def test_duplicate_accession_rejected(self):
        with self.assertRaises(ValueError):
            parse_samples("^SAMPLE = GSM1\n^SAMPLE = GSM1")

    def test_missing_and_repeated_single_fields_rejected(self):
        for record in ({}, {"field": [""]}, {"field": ["a", "b"]}):
            with self.subTest(record=record), self.assertRaises(ValueError):
                single(record, "field")

    def test_complete_sample_inventory(self):
        series, samples = synthetic_samples()
        result = audit_samples(series, samples)
        self.assertEqual(result["labels"], {"synthetic class": 2096})
        self.assertEqual(result["specimen_types"], {"bone marrow": 2096})
        self.assertTrue(result["titles_match_N1_0001_through_N1_2096"])
        self.assertFalse(result["patient_independence_verified"])

    def test_truncated_sample_record_rejected(self):
        series, samples = synthetic_samples()
        with self.assertRaises(ValueError):
            audit_samples(series, samples.rsplit("!Sample_series_id", 1)[0])

    def test_wrong_inventory_rejected(self):
        series, samples = synthetic_samples()
        with self.assertRaises(ValueError):
            audit_samples(series, samples.replace("GSM2096", "GSM9999"))

    def test_missing_label_rejected(self):
        series, samples = synthetic_samples()
        samples = samples.replace("!Sample_characteristics_ch1 = leukemia class: synthetic class\n", "", 1)
        with self.assertRaises(ValueError):
            audit_samples(series, samples)

    def test_annotation_counts_and_truncation(self):
        lines = [
            "!Annotation_platform = GPL570", "!Annotation_date = synthetic",
            "!platform_table_begin", "ID\tGene ID\tGene symbol",
            "probeA\t1///2\tA///B", "probeB\t1\tA", "probeC\t1\tA",
            "AFFX-control\t1\tA",
        ]
        lines.extend(f"unmapped{i}\t\t" for i in range(54671))
        lines.append("!platform_table_end")
        result = audit_annotation("\n".join(lines))
        self.assertEqual(result["mapping_counts_by_unique_Entrez_IDs_per_probe"],
                         {"single_gene": 3, "multiple_genes": 1, "unmapped": 54671})
        self.assertEqual(result["genes_with_multiple_single_gene_probes"], 1)
        self.assertEqual(result["AFFX_control_mapping_counts"]["single_gene"], 1)
        with self.assertRaises(ValueError):
            audit_annotation("\n".join(lines[:-1]))
        with self.assertRaises(ValueError):
            audit_annotation("\n".join(lines[:-1] + ["probeA\t1\tA", lines[-1]]))


if __name__ == "__main__":
    unittest.main()
