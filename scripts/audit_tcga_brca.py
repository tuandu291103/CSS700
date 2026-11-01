"""Offline TCGA-BRCA acquisition audit; original inputs are never modified.

Streaming expression checks avoid loading the complete matrix into a workbook.
Outputs are aggregate JSON plus an ignored sample manifest for later review.
"""

import argparse
import csv
import gzip
import hashlib
import json
import platform
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/tcga_brca_xena_2026-10-26"
PROTOCOL = ROOT / "experiments/05_tcga_brca_audit/protocol.json"
COUNT_REVIEW = ROOT / "experiments/05_tcga_brca_audit/source_count_review.json"
BARCODE = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-([0-9]{2})([A-Z])$")
MISSING = {"", "na", "n/a", "nan", "unknown", "not reported", "not available", "--"}
LABELS = {"01": "Primary Tumor", "11": "Solid Tissue Normal"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def counts(values):
    return dict(sorted(Counter(values).items()))


def missing(value):
    return value.strip().lower() in MISSING


def barcode(sample):
    match = BARCODE.fullmatch(sample)
    if match is None:
        raise ValueError(f"Expected TCGA sample-vial barcode: {sample!r}")
    return match.group(1), match.group(2)


def read_table(path, key, required):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        fields = reader.fieldnames
        if not fields or len(set(fields)) != len(fields) or not set(required) <= set(fields):
            raise ValueError("Missing or duplicate source columns")
        records = {}
        for line, row in enumerate(reader, start=2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"Ragged table at line {line}")
            identifier = row[key]
            if not identifier or identifier in records:
                raise ValueError(f"Missing or duplicate {key}: {identifier!r}")
            records[identifier] = row
    return records, fields


def scan_expression(path, block_size=256):
    """Audit all values; profile digests are over canonical float64 row order."""
    features, feature_set, rows, row_hashes = [], set(), [], Counter()
    n_missing = n_infinite = n_negative = n_zero = constant_rows = all_zero_rows = 0
    minimum, maximum = float("inf"), float("-inf")
    with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
        reader = csv.reader(source, delimiter="\t")
        header = next(reader)
        samples = header[1:]
        if not samples or len(set(samples)) != len(samples) or any(not s for s in samples):
            raise ValueError("Empty/duplicate expression sample IDs")
        for sample in samples:
            barcode(sample)
        sample_hashes = [hashlib.sha256() for _ in samples]
        per_sample_nonfinite = np.zeros(len(samples), dtype=np.int64)

        def flush():
            if rows:
                block = np.vstack(rows)
                for i, digest in enumerate(sample_hashes):
                    digest.update(block[:, i].astype("<f8").tobytes())
                rows.clear()

        for line, fields in enumerate(reader, start=2):
            if len(fields) != len(header):
                raise ValueError(f"Ragged expression row {line}")
            feature = fields[0]
            if not feature or feature in feature_set:
                raise ValueError(f"Missing/duplicate expression feature: {feature!r}")
            feature_set.add(feature)
            features.append(feature)
            try:
                values = np.asarray(
                    ["nan" if missing(v) else v for v in fields[1:]], dtype=np.float64
                )
            except ValueError as exc:
                raise ValueError(f"Nonnumeric expression row {line}") from exc
            finite = np.isfinite(values)
            per_sample_nonfinite += ~finite
            n_missing += int(np.isnan(values).sum())
            n_infinite += int(np.isinf(values).sum())
            n_negative += int((values[finite] < 0).sum())
            n_zero += int((values == 0).sum())
            if finite.any():
                minimum = min(minimum, float(values[finite].min()))
                maximum = max(maximum, float(values[finite].max()))
            if finite.all() and np.ptp(values) == 0:
                constant_rows += 1
                all_zero_rows += int(values[0] == 0)
            row_hashes[hashlib.sha256(values.astype("<f8").tobytes()).hexdigest()] += 1
            rows.append(values)
            if len(rows) >= block_size:
                flush()
        flush()
    if not features:
        raise ValueError("Expression table has no features")
    duplicate_profiles = defaultdict(list)
    for sample, digest in zip(samples, sample_hashes, strict=True):
        duplicate_profiles[digest.hexdigest()].append(sample)
    duplicate_groups = sorted(v for v in duplicate_profiles.values() if len(v) > 1)
    stats = {
        "features": len(features), "samples": len(samples),
        "values_checked": len(features) * len(samples),
        "missing_values": n_missing, "infinite_values": n_infinite,
        "negative_values": n_negative, "zero_values": n_zero,
        "finite_minimum": minimum if np.isfinite(minimum) else None,
        "finite_maximum": maximum if np.isfinite(maximum) else None,
        "all_cohort_constant_features_diagnostic_only": constant_rows,
        "all_zero_features": all_zero_rows,
        "duplicate_feature_value_groups": sum(v > 1 for v in row_hashes.values()),
        "duplicate_feature_value_extra_rows": sum(v - 1 for v in row_hashes.values()),
        "exact_duplicate_profile_groups": len(duplicate_groups),
        "cross_participant_duplicate_profile_groups": sum(
            len({barcode(s)[0] for s in group}) > 1 for group in duplicate_groups
        ),
        "samples_with_nonfinite_values": int((per_sample_nonfinite > 0).sum()),
    }
    return samples, features, stats, duplicate_groups


def join_clinical(samples, clinical):
    """Exact-ID join and participant identity checks, never a positional join."""
    manifest, conflicts = [], []
    patient_cases, case_patients = defaultdict(set), defaultdict(set)
    for sample in samples:
        participant, code = barcode(sample)
        row = clinical.get(sample)
        if row is None:
            conflicts.append({"sample": sample, "reason": "missing_clinical"})
            continue
        reason = []
        if row["submitter_id"] != participant:
            reason.append("participant_prefix_mismatch")
        if missing(row["case_id"]):
            reason.append("missing_case_id")
        actual_code = row["sample_type_id.samples"]
        if not actual_code.isdigit() or actual_code.zfill(2) != code:
            reason.append("sample_type_code_mismatch")
        if code in LABELS and row["sample_type.samples"] != LABELS[code]:
            reason.append("sample_type_label_mismatch")
        if reason:
            conflicts.append({"sample": sample, "reason": reason})
        patient_cases[participant].add(row["case_id"])
        case_patients[row["case_id"]].add(participant)
        manifest.append({
            "sample": sample, "participant": participant, "code": code,
            "label": row["sample_type.samples"], "case_id": row["case_id"],
            "eligible_type": code in LABELS, "identity_valid": not reason,
            "source_site": row["code.tissue_source_site"],
            "preservation": row["preservation_method.samples"],
            "ffpe": row["is_ffpe.samples"],
            "prior_treatment": row["prior_treatment.diagnoses"],
            "annotation_category": row["category.annotations"],
            "sample_annotations": row["annotations.samples"],
            "annotation_notes": row["notes.annotations"],
        })
    non_bijective = {
        "participants_with_multiple_case_ids": sum(len(v) != 1 for v in patient_cases.values()),
        "case_ids_with_multiple_participants": sum(len(v) != 1 for v in case_patients.values()),
    }
    return manifest, conflicts, non_bijective


def select_vials(manifest):
    retained, excluded, seen = [], [], set()
    for row in sorted(manifest, key=lambda r: r["sample"]):
        if not row["identity_valid"] or not row["eligible_type"]:
            continue
        key = (row["participant"], row["code"])
        if key in seen:
            excluded.append(row["sample"])
        else:
            seen.add(key)
            retained.append(row)
    return retained, excluded


def field_by_type(rows, field):
    result = defaultdict(Counter)
    for row in rows:
        result[row[field] or "<blank>"][row["code"]] += 1
    return {value: dict(sorted(ct.items())) for value, ct in sorted(result.items())}


def source_count_reconciliation(features, review):
    digest = hashlib.sha256(("\n".join(sorted(features)) + "\n").encode()).hexdigest()
    matches = (
        digest == review["sorted_ensembl_identifiers_sha256"]
        and len(features) == review["listed_unique_ensembl_identifiers"]
        and review["listed_non_expression_identifiers"] == ["sampleID"]
        and len(features) + 1 == review["listed_total_identifiers"]
    )
    return {"downloaded_sorted_ids_sha256": digest, "matches_full_browser_inventory": matches,
            "listed_non_expression_identifiers": review["listed_non_expression_identifiers"]}


def review_cohort_options(manifest, clinical):
    """Metadata-only feasibility scenario, not an adopted thesis cohort or split.

    Case flags propagate across the entire clinical table; sample flags apply to
    the affected sample. Exclude before deterministic vial selection. Never use
    expression values or prediction outcomes to decide inclusion.
    """
    case_flags = defaultdict(set)
    for row in clinical.values():
        patient = row["submitter_id"]
        if not missing(row["category.annotations"]) or not missing(row["notes.annotations"]):
            case_flags[patient].add("case_annotation")
        if row["prior_treatment.diagnoses"] != "No":
            case_flags[patient].add("prior_treatment_not_documented_No")
    reasons, passed = {}, []
    for row in manifest:
        if not row["eligible_type"] or not row["identity_valid"]:
            continue
        flags = set(case_flags[row["participant"]])
        if not missing(row["sample_annotations"]):
            flags.add("sample_annotation")
        if row["preservation"] == "FFPE" or row["ffpe"] == "True":
            flags.add("FFPE_in_either_source_field")
        if flags:
            reasons[row["sample"]] = sorted(flags)
        else:
            passed.append(row)
    conservative, repeated = select_vials(passed)
    participants = defaultdict(set)
    for row in conservative:
        participants[row["participant"]].add(row["code"])
    baseline, _ = select_vials(manifest)
    sites = field_by_type(baseline, "source_site")
    clinical_sample_flags = [r for r in clinical.values() if not missing(r["annotations.samples"])]
    joined_ids = {r["sample"] for r in manifest}
    summary = {
        "status": "Metadata-only conservative option; not yet adopted for a model experiment",
        "rule": "Exclude case-annotated participants and participants without consistently documented No prior treatment across the clinical source; exclude sample-annotated or FFPE-flagged samples; then choose one vial per participant/type. Unknown preservation is retained as unknown, not certified fresh tissue.",
        "sample_count": len(conservative), "participant_count": len(participants),
        "class_counts": counts(r["label"] for r in conservative),
        "participant_type_patterns": counts("+".join(sorted(v)) for v in participants.values()),
        "metadata_excluded_expression_samples_before_vial_selection": len(reasons),
        "exclusion_reason_counts_overlapping": counts(reason for v in reasons.values() for reason in v),
        "repeated_vials_excluded_after_metadata_filter": len(repeated),
        "metadata_by_sample_type": {field: field_by_type(conservative, field) for field in (
            "source_site", "preservation", "ffpe", "prior_treatment")},
        "initial_candidate_diagnostics": {
            "source_sites": len(sites),
            "sites_with_both_classes": sum(set(ct) == {"01", "11"} for ct in sites.values()),
            "tumor_samples_at_sites_without_normal_samples": sum(
                ct.get("01", 0) for ct in sites.values() if "11" not in ct
            ),
            "preservation_FFPE_but_ffpe_flag_False": sum(
                r["preservation"] == "FFPE" and r["ffpe"] == "False" for r in baseline
            ),
            "clinical_sample_annotation_records": len(clinical_sample_flags),
            "sample_annotated_records_present_in_expression": sum(
                r["sample"] in joined_ids for r in clinical_sample_flags
            ),
        },
    }
    return summary, {"samples": conservative, "metadata_exclusion_reasons": reasons,
                     "excluded_repeated_vials": repeated}


def audit(raw_dir):
    protocol = json.loads(PROTOCOL.read_text())
    count_review = json.loads(COUNT_REVIEW.read_text())
    clinical, clinical_fields = read_table(raw_dir / "TCGA-BRCA.clinical.tsv.gz", "sample", [
        "sample", "submitter_id", "case_id", "sample_type_id.samples", "sample_type.samples",
        "code.tissue_source_site", "preservation_method.samples", "is_ffpe.samples",
        "prior_treatment.diagnoses", "category.annotations", "annotations.samples", "notes.annotations",
    ])
    mapping, mapping_fields = read_table(
        raw_dir / "gencode.v36.annotation.gtf.gene.probemap", "id", ["id", "gene", "chrom"]
    )
    samples, features, expression, duplicates = scan_expression(raw_dir / "TCGA-BRCA.star_tpm.tsv.gz")
    print("Scanned all expression rows and values", flush=True)
    manifest, conflicts, bijection = join_clinical(samples, clinical)
    selected, repeated = select_vials(manifest)
    patients = defaultdict(set)
    for row in selected:
        patients[row["participant"]].add(row["code"])
    unmapped = sorted(set(features) - set(mapping))
    symbols = [mapping[f]["gene"] for f in features if f in mapping]
    symbol_counts = Counter(symbols)
    annotated = [row for row in selected if (
        not missing(row["annotation_category"]) or not missing(row["sample_annotations"])
        or not missing(row["annotation_notes"])
    )]
    count_check = source_count_reconciliation(features, count_review)
    cohort_options, private_option = review_cohort_options(manifest, clinical)
    expected = protocol["expected_source_dimensions"]
    sample_dimensions_match = (
        len(samples) == expected["expression_samples"] and len(clinical) == expected["clinical_samples"]
    )
    gates = {
        "expression_complete_finite_nonnegative": not any(
            expression[k] for k in ("missing_values", "infinite_values", "negative_values")
        ),
        "source_dimensions_reconciled": sample_dimensions_match and count_check["matches_full_browser_inventory"],
        "all_expression_samples_have_consistent_clinical_identity": not conflicts and not any(bijection.values()),
        "every_expression_id_has_exact_mapping": not unmapped,
        "no_cross_participant_exact_duplicate_profiles": not expression["cross_participant_duplicate_profile_groups"],
        "no_unreviewed_annotations_in_candidate": not annotated,
        "both_learning_classes_present": {r["code"] for r in selected} == {"01", "11"},
    }
    filenames = ["TCGA-BRCA.star_tpm.tsv.gz", "TCGA-BRCA.clinical.tsv.gz",
                 "gencode.v36.annotation.gtf.gene.probemap"]
    result = {
        "experiment": "05_tcga_brca_audit", "retrieval_date_local": "2026-10-26",
        "sources": {name: {"url": f"https://gdc-hub.s3.us-east-1.amazonaws.com/download/{name}",
                            "bytes": (raw_dir / name).stat().st_size, "sha256": sha256(raw_dir / name)}
                    for name in filenames},
        "protocol_sha256": sha256(PROTOCOL), "audit_script_sha256": sha256(Path(__file__)),
        "source_count_review_sha256": sha256(COUNT_REVIEW),
        "source_count_reconciliation": {**count_check,
            "unadjusted_dimensions_match_initial_listing": sample_dimensions_match and len(features) == expected["expression_features"]},
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "expression": expression,
        "clinical": {"samples": len(clinical), "file_columns_including_sample_key": len(clinical_fields),
                     "participants": len({r["submitter_id"] for r in clinical.values()}),
                     "sample_type_counts": counts(r["sample_type.samples"] for r in clinical.values()),
                     "pam50_named_fields": [f for f in clinical_fields if "pam50" in f.lower()]},
        "join": {"matched": len(manifest), "expression_only": len(set(samples) - set(clinical)),
                 "clinical_only": len(set(clinical) - set(samples)),
                 "identity_conflicts": len(conflicts), **bijection,
                 "matched_sample_type_counts": counts(r["label"] for r in manifest),
                 "matched_participants": len({r["participant"] for r in manifest})},
        "mapping": {"rows": len(mapping), "fields": mapping_fields,
                    "exact_mapped_expression_ids": len(features) - len(unmapped),
                    "unmapped_expression_ids": unmapped,
                    "unused_map_ids": len(set(mapping) - set(features)),
                    "distinct_symbols": len(symbol_counts),
                    "blank_symbols": sum(missing(s) for s in symbols),
                    "symbols_shared_by_multiple_expression_ids": sum(n > 1 for n in symbol_counts.values()),
                    "extra_expression_ids_sharing_symbols": sum(n - 1 for n in symbol_counts.values()),
                    "most_repeated_symbols": dict(symbol_counts.most_common(10)),
                    "symbol_collapse_performed": False},
        "candidate": {"sample_count": len(selected), "participant_count": len(patients),
                      "class_counts": counts(r["label"] for r in selected),
                      "participant_type_patterns": counts("+".join(sorted(v)) for v in patients.values()),
                      "repeated_participant_type_vials_excluded": len(repeated),
                      "non_target_samples_excluded": sum(not r["eligible_type"] for r in manifest),
                      "annotated_samples_requiring_review": len(annotated),
                      "annotated_participants_requiring_review": len({r["participant"] for r in annotated}),
                      "metadata_by_sample_type": {field: field_by_type(selected, field) for field in (
                          "source_site", "preservation", "ffpe", "prior_treatment", "annotation_category",
                          "sample_annotations")}},
        "conservative_cohort_option": cohort_options,
        "gates": gates, "training_ready_without_further_review": all(gates.values()),
        "models_trained": False,
    }
    private = {"all_joined_samples": manifest, "candidate_samples": selected,
               "excluded_repeated_vials": repeated, "identity_conflicts": conflicts,
               "duplicate_profile_groups": duplicates, "conservative_cohort_option": private_option}
    return result, private


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/05_tcga_brca_audit/results.json")
    parser.add_argument("--private-output", type=Path, default=ROOT / "data/interim/tcga_brca_audit_samples.json")
    args = parser.parse_args()
    result, private = audit(args.raw_dir)
    for path, data in ((args.output, result), (args.private_output, private)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("expression", "join", "mapping", "gates")}, indent=2))


if __name__ == "__main__":
    main()
