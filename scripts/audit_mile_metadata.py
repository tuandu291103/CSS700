"""Offline, standard-library audit of GEO metadata; no model training or downloads.

See week02/GSE13159_FEASIBILITY.md for acquisition and reproduction commands.
The JSON is an audit of deposited metadata, not proof of patient independence.
"""

import argparse
from collections import Counter, defaultdict
from datetime import date
import gzip
import hashlib
import json
from pathlib import Path
import re


SOURCES = {
    "series": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE13159&targ=self&form=text&view=brief",
    "samples": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE13159&targ=gsm&form=text&view=brief",
    "annotation": "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_samples(text):
    records = {}
    current = None
    for line in text.splitlines():
        if line.startswith("^SAMPLE = "):
            accession = line.split(" = ", 1)[1]
            require(re.fullmatch(r"GSM\d+", accession), "Invalid sample accession")
            require(accession not in records, f"Duplicate accession: {accession}")
            current = defaultdict(list)
            records[accession] = current
        elif line.startswith("!Sample_"):
            require(current is not None, "Sample attribute outside a sample record")
            key, value = line.split(" = ", 1)
            current[key.removeprefix("!Sample_")].append(value)
    require(records, "No sample records: possible empty/HTML response")
    return records


def single(record, field):
    values = record.get(field, [])
    require(len(values) == 1 and values[0].strip(), f"Missing/repeated field: {field}")
    return values[0]


def audit_samples(series_text, sample_text):
    require("^SERIES = GSE13159" in series_text.splitlines(), "Wrong series")
    expected = re.findall(r"^!Series_sample_id = (GSM\d+)\s*$", series_text, re.M)
    require(len(expected) == len(set(expected)) == 2096, "Expected 2,096 unique series samples")
    records = parse_samples(sample_text)
    require(set(records) == set(expected), "Sample accessions differ from series inventory")
    counts = {key: Counter() for key in (
        "labels", "specimen_types", "platforms", "reported_probe_rows",
        "series_memberships", "characteristic_fields", "metadata_fields",
        "source_names", "processing_descriptions", "treatment_descriptions",
    )}
    titles = []
    by_label = defaultdict(Counter)
    for accession, record in records.items():
        require(single(record, "geo_accession") == accession, "Accession mismatch")
        titles.append(single(record, "title"))
        require(single(record, "organism_ch1") == "Homo sapiens", "Unexpected organism")
        characteristics = defaultdict(list)
        for entry in record.get("characteristics_ch1", []):
            key, separator, value = entry.partition(": ")
            require(separator and value.strip(), "Invalid characteristic")
            characteristics[key].append(value)
        label = single(characteristics, "leukemia class")
        specimen = single(characteristics, "sample type")
        counts["labels"][label] += 1
        counts["specimen_types"][specimen] += 1
        by_label[label][specimen] += 1
        counts["characteristic_fields"].update(characteristics.keys())
        counts["metadata_fields"].update(record.keys())
        for target, field in (
            ("platforms", "platform_id"), ("reported_probe_rows", "data_row_count"),
            ("source_names", "source_name_ch1"), ("processing_descriptions", "data_processing"),
            ("treatment_descriptions", "treatment_protocol_ch1"),
        ):
            counts[target][single(record, field)] += 1
        require("GSE13159" in record.get("series_id", []), "Sample missing target series")
        counts["series_memberships"].update(set(record["series_id"]))
    require(counts["platforms"] == {"GPL570": 2096}, "Unexpected platform distribution")
    require(counts["reported_probe_rows"] == {"54675": 2096}, "Unexpected declared row counts")
    return {
        "record_count": len(records),
        "series_inventory_exact_match": True,
        "unique_titles": len(set(titles)),
        "titles_match_N1_0001_through_N1_2096": set(titles) == {
            f"MILES stage 1 data N1_{i:04d}" for i in range(1, 2097)
        },
        **{key: dict(sorted(value.items())) for key, value in counts.items()},
        "label_by_specimen_type": {key: dict(sorted(value.items())) for key, value in sorted(by_label.items())},
        "largest_to_smallest_class_ratio": max(counts["labels"].values()) / min(counts["labels"].values()),
        "patient_independence_verified": False,
        "expression_values_audited": False,
    }


def gene_ids(value):
    parts = {part.strip() for part in value.split("///")}
    parts -= {"", "---"}
    require(all(re.fullmatch(r"[1-9]\d*", part) for part in parts), "Invalid Gene ID token")
    return parts


def audit_annotation(text):
    lines = text.splitlines()
    metadata = {}
    for line in lines:
        if line.startswith("!Annotation_"):
            key, value = line.split(" = ", 1)
            metadata[key.removeprefix("!Annotation_")] = value
    require(metadata.get("platform") == "GPL570", "Wrong annotation platform")
    require(lines.count("!platform_table_begin") == lines.count("!platform_table_end") == 1,
            "Missing/duplicate annotation table markers: possible truncated input")
    start = lines.index("!platform_table_begin")
    end = lines.index("!platform_table_end")
    require(end > start + 1, "Empty/reversed annotation table")
    header = lines[start + 1].split("\t")
    require({"ID", "Gene ID", "Gene symbol"} <= set(header), "Missing annotation columns")
    probes, all_genes = set(), set()
    categories = Counter({"unmapped": 0, "single_gene": 0, "multiple_genes": 0})
    control_categories = categories.copy()
    single_gene_probes = Counter()
    examples = {}
    for line in lines[start + 2:end]:
        values = line.split("\t")
        require(len(values) == len(header), "Malformed annotation row")
        row = dict(zip(header, values))
        probe = row["ID"]
        require(probe and probe not in probes, "Empty/duplicate probe ID")
        probes.add(probe)
        genes = gene_ids(row["Gene ID"])
        category = "unmapped" if not genes else "single_gene" if len(genes) == 1 else "multiple_genes"
        categories[category] += 1
        if probe.startswith("AFFX-"):
            control_categories[category] += 1
        all_genes.update(genes)
        if len(genes) == 1:
            single_gene_probes.update(genes)
        if probe in {"1007_s_at", "1053_at", "1294_at"}:
            examples[probe] = {"gene_ids": sorted(genes), "symbol_field": row["Gene symbol"]}
    require(len(probes) == 54675, "Expected 54,675 unique GPL570 annotation rows")
    return {
        "metadata": metadata,
        "row_count": len(probes),
        "mapping_counts_by_unique_Entrez_IDs_per_probe": dict(categories),
        "AFFX_control_mapping_counts": dict(control_categories),
        "unique_Entrez_IDs_including_ambiguous_probes": len(all_genes),
        "unique_Entrez_IDs_from_single_gene_probes": len(single_gene_probes),
        "genes_with_multiple_single_gene_probes": sum(n > 1 for n in single_gene_probes.values()),
        "illustrative_probe_mappings": examples,
        "expression_probe_ID_join_verified": False,
        "current_gene_ID_validity_verified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in SOURCES:
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--retrieved-on", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {}
    texts = {}
    for key, url in SOURCES.items():
        path = getattr(args, key)
        payload = path.read_bytes()
        inputs[key] = {
            "filename": path.name, "url": url, "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        texts[key] = (gzip.decompress(payload) if key == "annotation" else payload).decode("utf-8")
    report = {
        "schema_version": 1,
        "retrieved_on": args.retrieved_on.isoformat(),
        "scope": "Metadata and platform annotation only; no expression matrix or patient-linkage audit",
        "inputs": inputs,
        "samples": audit_samples(texts["series"], texts["samples"]),
        "annotation": audit_annotation(texts["annotation"]),
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Validated {report['samples']['record_count']} sample records and "
          f"{report['annotation']['row_count']} annotation rows; wrote {args.output}")


if __name__ == "__main__":
    main()
