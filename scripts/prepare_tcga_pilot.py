"""Freeze metadata-only splits and training-only genomic features; no model fitting."""

import argparse
import csv
import gzip
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.audit_tcga_brca import (
    RAW,
    ROOT,
    counts,
    join_clinical,
    read_table,
    review_cohort_options,
    sha256,
)

PROTOCOL = ROOT / "experiments/06_tcga_brca_pilot/protocol.json"
PUBLIC = ROOT / "experiments/06_tcga_brca_pilot/preparation.json"
PRIVATE = ROOT / "data/processed/tcga_brca_pilot"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def make_splits(rows, seed, heldout_site=None):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["participant"]].append(i)
    strata = defaultdict(list)
    partitions = {"train": [], "validation": [], "test": []}
    for patient, indices in sorted(groups.items()):
        sites = {rows[i]["source_site"] for i in indices}
        if len(sites) != 1 or sites != {patient.split("-")[1]}:
            raise ValueError("Participant/site metadata conflict")
        if {rows[i]["code"] for i in indices} not in ({"01"}, {"01", "11"}):
            raise ValueError("Unexpected participant sample-type pattern")
        if heldout_site in sites:
            partitions["test"].extend(indices)
        else:
            strata[len(indices)].append(patient)
    rng = np.random.default_rng(seed)
    for pattern in sorted(strata):
        patients = rng.permutation(sorted(strata[pattern])).tolist()
        n = int(np.ceil(len(patients) * 0.15))
        n_test = n if heldout_site is None else 0
        if len(patients) <= n + n_test:
            raise ValueError("Insufficient participants for fixed stratified policy")
        for j, patient in enumerate(patients):
            part = "test" if j < n_test else "validation" if j < n_test + n else "train"
            partitions[part].extend(groups[patient])
    splits = {name: np.array(sorted(ix), dtype=int) for name, ix in partitions.items()}
    validate_splits(rows, splits, heldout_site)
    return splits


def validate_splits(rows, splits, heldout_site=None):
    if not np.array_equal(np.sort(np.concatenate(list(splits.values()))), np.arange(len(rows))):
        raise ValueError("Splits must cover samples exactly once")
    for field in ("participant", "case_id"):
        seen = set()
        for ix in splits.values():
            values = {rows[i][field] for i in ix}
            if values & seen:
                raise ValueError(f"Cross-partition {field} overlap")
            seen |= values
    for name, ix in splits.items():
        if {rows[i]["code"] for i in ix} != {"01", "11"}:
            raise ValueError(f"Both classes required in {name}")
        if heldout_site is not None:
            sites = {rows[i]["source_site"] for i in ix}
            if (name == "test" and sites != {heldout_site}) or (name != "test" and heldout_site in sites):
                raise ValueError("Held-out collection site leaked into development")


def select_scale(raw, features, train_indices, budget):
    """All fitted statistics depend only on the explicit training rows."""
    training = raw[train_indices]
    variance = np.var(training, axis=0, ddof=0)
    order = np.lexsort((np.asarray(features), -variance))
    eligible = order[variance[order] > 0]
    if len(eligible) < budget:
        raise ValueError("Too few nonconstant training features")
    selected = np.array(sorted(eligible[:budget], key=lambda i: features[i]))
    center = training[:, selected].mean(axis=0)
    scale = np.sqrt(variance[selected])
    transformed = (raw[:, selected] - center) / scale
    if not np.isfinite(transformed).all():
        raise ValueError("Nonfinite standardized data")
    return selected, transformed.astype(np.float32), center, scale, variance[selected]


def load_sources(protocol):
    audit_path = ROOT / protocol["input_audit"]
    if sha256(audit_path) != protocol["input_audit_sha256"]:
        raise ValueError("Audit snapshot changed")
    if sha256(ROOT / "scripts/audit_tcga_brca.py") != protocol["cohort_implementation_sha256"]:
        raise ValueError("Cohort implementation changed")
    audit = json.loads(audit_path.read_text())
    for filename, item in audit["sources"].items():
        if sha256(RAW / filename) != item["sha256"]:
            raise ValueError(f"Raw input changed: {filename}")
    with gzip.open(RAW / "TCGA-BRCA.star_tpm.tsv.gz", "rt") as source:
        expression_samples = next(csv.reader(source, delimiter="\t"))[1:]
    clinical, _ = read_table(RAW / "TCGA-BRCA.clinical.tsv.gz", "sample", ["sample", "case_id"])
    manifest, conflicts, bijection = join_clinical(expression_samples, clinical)
    if conflicts or any(bijection.values()):
        raise ValueError("Clinical identity conflicts")
    _, option = review_cohort_options(manifest, clinical)
    rows = option["samples"]
    actual = {"samples": len(rows), "participants": len({r["participant"] for r in rows}),
              "tumors": sum(r["code"] == "01" for r in rows), "normals": sum(r["code"] == "11" for r in rows)}
    if actual != protocol["expected_cohort"]:
        raise ValueError("Frozen conservative cohort changed")
    mapping, _ = read_table(RAW / "gencode.v36.annotation.gtf.gene.probemap", "id", ["id", "gene"])
    return audit, rows, mapping


def prepare(output_dir, public_output):
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Use a new directory; do not overwrite frozen data")
    if public_output.exists():
        raise FileExistsError("Use a new preparation report path")
    protocol = json.loads(PROTOCOL.read_text())
    audit, rows, mapping = load_sources(protocol)
    scenarios = {name: make_splits(rows, protocol["split_seed"], site) for name, site in (
        ("patient_grouped", None), ("site_BH_holdout", "BH"))}
    private = {"protocol_sha256": sha256(PROTOCOL), "samples": rows,
               "scenarios": {name: {k: ix.tolist() for k, ix in splits.items()} for name, splits in scenarios.items()}}
    # Persist metadata-only partitions BEFORE reading expression values for selection.
    write_json(output_dir / "partitions.json", private)
    print("Frozen both patient-grouped partitions before feature selection", flush=True)
    raw = np.empty((len(rows), audit["expression"]["features"]), dtype=np.float64)
    features = []
    with gzip.open(RAW / "TCGA-BRCA.star_tpm.tsv.gz", "rt") as source:
        header = next(source).rstrip("\n").split("\t")[1:]
        indices = [header.index(r["sample"]) for r in rows]
        for j, line in enumerate(source):
            feature, payload = line.rstrip("\n").split("\t", 1)
            values = np.fromstring(payload, sep="\t", dtype=np.float64)
            if len(values) != len(header) or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("Malformed expression row")
            features.append(feature)
            raw[:, j] = values[indices]
    if len(features) != raw.shape[1] or len(set(features)) != len(features):
        raise ValueError("Expression feature count/identity mismatch")
    report = {"experiment": protocol["experiment"], "protocol_sha256": sha256(PROTOCOL),
              "preparation_code_sha256": sha256(Path(__file__)),
              "cohort_implementation_sha256": protocol["cohort_implementation_sha256"],
              "input_audit_sha256": protocol["input_audit_sha256"],
              "source_hashes": {k: v["sha256"] for k, v in audit["sources"].items()},
              "private_partitions_sha256": sha256(output_dir / "partitions.json"),
              "environment": {"python": platform.python_version(), "numpy": np.__version__},
              "scenarios": {}}
    labels = np.array([int(r["code"] == "01") for r in rows], dtype=np.int64)
    shape = protocol["preprocessing"]["image_shape"]
    budget = protocol["preprocessing"]["gene_budget"]
    if np.prod(shape) != budget:
        raise ValueError("Grid must match gene budget exactly")
    for name, splits in scenarios.items():
        selected, x, mean, std, variance = select_scale(raw, features, splits["train"], budget)
        images = x.reshape(len(x), 1, *shape)
        np.testing.assert_array_equal(images.reshape(x.shape), x)
        # Accumulate diagnostics in float64; float32 reductions can add rounding
        # error even when the stored float32 inputs were standardized correctly.
        checked_train = x[splits["train"]].astype(np.float64)
        np.testing.assert_allclose(checked_train.mean(axis=0), 0, atol=1e-6)
        np.testing.assert_allclose(checked_train.std(axis=0), 1, atol=1e-6)
        np.savez_compressed(output_dir / f"{name}.npz", x=x, y=labels, **splits)
        report["scenarios"][name] = {
            "processed_arrays_sha256": sha256(output_dir / f"{name}.npz"),
            "float32_input_digest": hashlib.sha256(x.tobytes()).hexdigest(),
            "partitions": {part: {"samples": len(ix),
                "participants": len({rows[i]["participant"] for i in ix}),
                "class_counts": counts(rows[i]["label"] for i in ix),
                "source_site_counts": counts(rows[i]["source_site"] for i in ix),
                "sample_order_sha256": digest_json([rows[i]["sample"] for i in ix])}
                for part, ix in splits.items()},
            "patient_overlap": 0, "case_overlap": 0,
            "gene_count": budget, "grid_shape": shape, "scaled_round_trip_max_abs_error": 0.0,
            "mapping": [{"feature_index": j, "ensembl_id": features[i], "symbol": mapping[features[i]]["gene"],
                         "row": j // shape[1], "column": j % shape[1], "train_mean": float(mean[j]),
                         "train_std": float(std[j]), "train_variance": float(variance[j])}
                        for j, i in enumerate(selected)]}
        print(f"Prepared {name}: {report['scenarios'][name]['gene_count']} genes", flush=True)
    write_json(public_output, report)
    print(json.dumps({k: v["partitions"] for k, v in report["scenarios"].items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PRIVATE)
    parser.add_argument("--public-output", type=Path, default=PUBLIC)
    arguments = parser.parse_args()
    prepare(arguments.output_dir, arguments.public_output)
