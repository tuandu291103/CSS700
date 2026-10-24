"""Learning experiment 01: audit WDBC and trace a reversible fixed-grid encoding.

No classifiers, feature selection, learned layout, or attribution are fitted here.
Source feature order: UCI wdbc.names, fields 3--32.
"""

import argparse
import hashlib
import json
import platform
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
SOURCE_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/"
EXPECTED_HASHES = {
    "wdbc.data": "d606af411f3e5be8a317a5a8b652b425aaf0ff38ca683d5327ffff94c3695f4a",
    "wdbc.names": "840e04e3f20f8a5b326892f3b9cbc01c4cd6f7e6c597630b701ef6c0ac79f5ef",
}
BASE_FEATURES = (
    "radius", "texture", "perimeter", "area", "smoothness", "compactness",
    "concavity", "concave_points", "symmetry", "fractal_dimension",
)
FEATURES = tuple(f"{feature}_{summary}" for summary in ("mean", "se", "worst")
                 for feature in BASE_FEATURES)
SEED = 20261023
SHAPE = (5, 6)  # Our pedagogical row-major choice; not the SOM2IMG grid rule.


def load_data(path):
    """Parse the raw 32-field release, rejecting malformed/nonfinite predictors."""
    frame = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    if frame.shape[1] != 32 or len(frame) == 0:
        raise ValueError("Expected nonempty rows with ID, diagnosis, and 30 features")
    frame.columns = ["sample_id", "diagnosis", *FEATURES]
    if not frame.sample_id.str.fullmatch(r"[0-9]+").all():
        raise ValueError("Invalid sample ID")
    if frame.sample_id.duplicated().any():
        raise ValueError("Duplicate sample IDs require review before splitting")
    if not frame.diagnosis.isin(["B", "M"]).all():
        raise ValueError("Unexpected diagnosis or missing label")
    numeric = frame.loc[:, FEATURES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Missing or nonfinite feature values")
    return pd.concat([frame[["sample_id", "diagnosis"]], numeric], axis=1)


def verify_sources(directory):
    records = {}
    for filename, expected in EXPECTED_HASHES.items():
        payload = (directory / filename).read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise ValueError(f"Source checksum mismatch: {filename}; review version before use")
        records[filename] = {"url": SOURCE_BASE + filename, "sha256": actual,
                             "bytes": len(payload), "retrieved_date": "2026-10-23"}
    return records


def split_rows(labels, seed=SEED):
    """Fixed exploratory 70/15/15 split; rounding yields 397/86/86 for WDBC."""
    indices = np.arange(len(labels))
    development, test = train_test_split(
        indices, test_size=0.15, random_state=seed, stratify=labels,
    )
    train, validation = train_test_split(
        development, test_size=0.15 / 0.85, random_state=seed,
        stratify=np.asarray(labels)[development],
    )
    return {"train": np.sort(train), "validation": np.sort(validation), "test": np.sort(test)}


def make_mapping(features=FEATURES, shape=SHAPE):
    height, width = shape
    if height < 1 or width < 1 or len(features) > height * width:
        raise ValueError("Grid cannot hold all features")
    if not features or len(set(features)) != len(features):
        raise ValueError("Feature names must be nonempty and unique")
    return [{"feature": name, "feature_index": i, "row": i // width, "column": i % width}
            for i, name in enumerate(features)]


def validate_mapping(mapping, shape):
    if len(mapping) == 0:
        raise ValueError("Empty mapping")
    if [item["feature_index"] for item in mapping] != list(range(len(mapping))):
        raise ValueError("Mapping must be in feature-index order")
    if len({item["feature"] for item in mapping}) != len(mapping):
        raise ValueError("Duplicate feature names")
    coordinates = [(item["row"], item["column"]) for item in mapping]
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("Feature collision")
    if any(not (0 <= row < shape[0] and 0 <= col < shape[1]) for row, col in coordinates):
        raise ValueError("Coordinate outside image")


def encode(values, mapping, shape=SHAPE):
    values = np.asarray(values, dtype=np.float64)
    validate_mapping(mapping, shape)
    if values.ndim != 2 or values.shape[1] != len(mapping) or not np.isfinite(values).all():
        raise ValueError("Expected a finite sample-by-feature matrix matching the map")
    images = np.zeros((len(values), *shape), dtype=np.float64)
    occupied = np.zeros(shape, dtype=bool)
    for item in mapping:
        row, col = item["row"], item["column"]
        images[:, row, col] = values[:, item["feature_index"]]
        occupied[row, col] = True
    return images, occupied


def decode(images, mapping):
    images = np.asarray(images, dtype=np.float64)
    if images.ndim != 3 or not np.isfinite(images).all():
        raise ValueError("Expected finite sample-by-height-by-width images")
    validate_mapping(mapping, images.shape[1:])
    return np.column_stack([images[:, item["row"], item["column"]] for item in mapping])


def prepare_audit(raw_dir=ROOT / "data/raw/wdbc"):
    sources = verify_sources(raw_dir)
    frame = load_data(raw_dir / "wdbc.data")
    if len(frame) != 569 or Counter(frame.diagnosis) != {"B": 357, "M": 212}:
        raise ValueError("Release dimensions or label inventory differ from UCI record")
    values = frame.loc[:, FEATURES].to_numpy(dtype=np.float64)
    duplicates = int(frame.loc[:, FEATURES].duplicated().sum())
    if duplicates:
        raise ValueError("Duplicate predictor rows require a grouping policy before splitting")
    splits = split_rows(frame.diagnosis)
    train = values[splits["train"]]
    scaler = MinMaxScaler(clip=False).fit(train)
    mapping = make_mapping()
    checks = {}
    for name, indices in splits.items():
        scaled = scaler.transform(values[indices])
        images, occupied = encode(scaled, mapping)
        decoded = decode(images, mapping)
        restored = scaler.inverse_transform(decoded)
        np.testing.assert_array_equal(decoded, scaled)
        np.testing.assert_allclose(restored, values[indices], rtol=1e-12, atol=1e-10)
        checks[name] = {
            "rows": len(indices), "class_counts": dict(sorted(Counter(frame.diagnosis.iloc[indices]).items())),
            "row_indices_zero_based": indices.tolist(),
            "sample_ids": frame.sample_id.iloc[indices].tolist(),
            "scaled_round_trip_max_abs_error": float(np.max(np.abs(decoded - scaled))),
            "raw_round_trip_max_abs_error": float(np.max(np.abs(restored - values[indices]))),
        }
    # Only training data contribute to exploratory distributions/correlations.
    training_frame = frame.iloc[splits["train"]].loc[:, FEATURES]
    correlations = training_frame.corr(method="spearman")
    pairs = sorted(
        [{"feature_a": FEATURES[i], "feature_b": FEATURES[j], "rho": float(correlations.iloc[i, j])}
         for i in range(30) for j in range(i + 1, 30)],
        key=lambda pair: -abs(pair["rho"]),
    )[:10]
    report = {
        "experiment": "01_wisconsin_learning", "date": "2026-10-23",
        "scope": "Learning data audit and fixed-grid encoding; no trained classifier or attribution",
        "sources": sources, "dataset_doi": "10.24432/C5DW2B", "license": "CC BY 4.0",
        "environment": {"python": platform.python_version(), **{p: version(p) for p in (
            "numpy", "pandas", "scikit-learn", "matplotlib", "nbformat", "nbclient", "ipykernel")}},
        "audit": {"rows": len(frame), "predictors": len(FEATURES),
                  "class_counts": dict(sorted(Counter(frame.diagnosis).items())),
                  "missing_or_nonfinite_predictor_values": int((~np.isfinite(values)).sum()),
                  "duplicate_sample_ids": int(frame.sample_id.duplicated().sum()),
                  "duplicate_predictor_rows": duplicates,
                  "patient_independence_verified": False},
        "split_seed": SEED, "split_policy": "Stratified learning split, not a final thesis holdout",
        "splits": checks,
        "scaling": {"method": "MinMaxScaler", "fit_partition": "train", "clip": False,
                    "training_min": dict(zip(FEATURES, scaler.data_min_.tolist(), strict=True)),
                    "training_max": dict(zip(FEATURES, scaler.data_max_.tolist(), strict=True))},
        "encoding": {"shape": list(SHAPE), "dtype": "float64", "occupied_pixels": int(occupied.sum()),
                     "padding_pixels": int((~occupied).sum()), "collisions": 0, "mapping": mapping},
        "training_constant_features": [name for name in FEATURES if training_frame[name].nunique() == 1],
        "training_descriptive_statistics": json.loads(training_frame.describe().to_json()),
        "training_top_absolute_spearman_pairs": pairs,
        "example_row_index": int(splits["train"][0]),
    }
    return frame, splits, scaler, mapping, report


def trace_example(frame, splits, scaler, mapping):
    index = splits["train"][0]
    raw = frame.loc[index, list(FEATURES)].to_numpy(dtype=float)[None, :]
    scaled = scaler.transform(raw)
    images, _ = encode(scaled, mapping)
    restored = scaler.inverse_transform(decode(images, mapping))[0]
    return pd.DataFrame([
        {**item, "raw_value": raw[0, item["feature_index"]],
         "scaled_value": scaled[0, item["feature_index"]],
         "decoded_raw_value": restored[item["feature_index"]]}
        for item in mapping
    ]), images[0]


def encoding_figure(image, mapping):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6.5), layout="constrained")
    plot = ax.imshow(image, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    for item in mapping:
        row, col = item["row"], item["column"]
        label = item["feature"].replace("_", "\n", 1)
        color = "white" if image[row, col] > 0.6 else "#172a3a"
        ax.text(col, row, f"{label}\n{image[row, col]:.3f}", ha="center", va="center", fontsize=9, color=color)
    ax.set(xticks=range(6), yticks=range(5), xlabel="Column (zero-based)", ylabel="Row (zero-based)",
           title="One training sample: 30 features placed into a 5 × 6 grid\nColor = scaled input value, NOT model importance")
    fig.colorbar(plot, ax=ax, label="Training-min–max scaled value")
    return fig


def correlation_figure(frame, splits):
    import matplotlib.pyplot as plt

    # Show mean summaries for legibility; complete 30-feature statistics are in results.json.
    selected = list(FEATURES[:10])
    corr = frame.iloc[splits["train"]][selected].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 8), layout="constrained")
    plot = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set(xticks=range(10), yticks=range(10), xticklabels=BASE_FEATURES, yticklabels=BASE_FEATURES,
           title="Feature relationships before image conversion\nSpearman correlation; mean summaries; training rows only")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.colorbar(plot, ax=ax, label="Spearman ρ")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/wdbc")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/01_wisconsin")
    args = parser.parse_args()
    frame, splits, scaler, mapping, report = prepare_audit(args.raw_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    _, example_image = trace_example(frame, splits, scaler, mapping)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for filename, figure in [("encoding.png", encoding_figure(example_image, mapping)),
                             ("correlations.png", correlation_figure(frame, splits))]:
        figure.savefig(args.output_dir / filename, dpi=150)
        plt.close(figure)
    print(json.dumps({"audit": report["audit"], "splits": {k: v["rows"] for k, v in report["splits"].items()},
                      "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
