"""Command-line entry points for environment checks and the first pilot checkpoint."""

import argparse
import csv
import importlib.util
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from css700.data import load_breast_cancer_splits
from css700.encoding import GridEncoder

CORE_MODULES = ("numpy", "pandas", "yaml", "sklearn", "scipy")
RESEARCH_MODULES = (
    "torch",
    "captum",
    "shap",
    "lime",
    "xgboost",
    "matplotlib",
    "seaborn",
    "minisom",
)


def _module_status(name: str) -> str:
    return "installed" if importlib.util.find_spec(name) is not None else "missing"


def doctor(_: argparse.Namespace) -> int:
    """Print a concise environment report and fail only when core packages are absent."""

    core = {name: _module_status(name) for name in CORE_MODULES}
    research = {name: _module_status(name) for name in RESEARCH_MODULES}
    device: dict[str, Any] = {"cuda": False, "mps": False}
    if research["torch"] == "installed":
        import torch

        device = {
            "cuda": bool(torch.cuda.is_available()),
            "mps": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        }

    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "core": core,
        "research": research,
        "accelerators": device,
    }
    print(json.dumps(report, indent=2))
    return 1 if "missing" in core.values() else 0


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "uncommitted"


def _package_versions() -> dict[str, str]:
    distributions = ("numpy", "pandas", "PyYAML", "scikit-learn", "scipy", "css700")
    versions: dict[str, str] = {}
    for distribution in distributions:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = "missing"
    return versions


def _class_counts(values: np.ndarray) -> dict[str, int]:
    classes, counts = np.unique(values, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(classes, counts, strict=True)}


def pilot(args: argparse.Namespace) -> int:
    """Run the leakage and mapping checkpoint before CNN implementation."""

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["experiment"]["seed"])
    dataset_config = config["dataset"]
    bundle = load_breast_cancer_splits(
        seed=seed,
        test_size=float(dataset_config["test_size"]),
        validation_size=float(dataset_config["validation_size"]),
        stratify=bool(dataset_config["stratify"]),
    )

    encoding_config = config["encoding"]
    encoder = GridEncoder(padding_value=float(encoding_config["padding_value"]))
    train_images = encoder.fit_transform(bundle.x_train, bundle.feature_names)
    validation_images = encoder.transform(bundle.x_validation)
    test_images = encoder.transform(bundle.x_test)

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_id = f"{timestamp}__breast-cancer__grid__preprocess__seed-{seed}__pilot"
    output_root = Path(args.output_dir) / run_id
    encoding_dir = output_root / "encoding"
    split_dir = output_root / "splits"
    encoding_dir.mkdir(parents=True, exist_ok=False)
    split_dir.mkdir(parents=True, exist_ok=False)

    (output_root / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    with (encoding_dir / "feature_map.csv").open("w", newline="", encoding="utf-8") as handle:
        records = encoder.mapping_records()
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    np.save(encoding_dir / "padding_mask.npy", encoder.padding_mask())
    np.savez_compressed(
        split_dir / "indices.npz",
        train=bundle.train_indices,
        validation=bundle.validation_indices,
        test=bundle.test_indices,
    )

    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "commit_sha": _git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _package_versions(),
    }
    summary = {
        "dataset": "Breast Cancer Wisconsin (scikit-learn)",
        "features": len(bundle.feature_names),
        "classes": list(bundle.target_names),
        "split_shapes": {
            "train_tabular": list(bundle.x_train.shape),
            "validation_tabular": list(bundle.x_validation.shape),
            "test_tabular": list(bundle.x_test.shape),
            "train_images": list(train_images.shape),
            "validation_images": list(validation_images.shape),
            "test_images": list(test_images.shape),
        },
        "class_counts": {
            "train": _class_counts(bundle.y_train),
            "validation": _class_counts(bundle.y_validation),
            "test": _class_counts(bundle.y_test),
        },
        "grid": {"rows": encoder.rows, "columns": encoder.columns},
        "training_scaled_mean_max_abs": float(np.abs(bundle.x_train.mean(axis=0)).max()),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "output": str(output_root), **summary}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="css700")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor_parser = subparsers.add_parser("doctor", help="check core and research dependencies")
    doctor_parser.set_defaults(function=doctor)
    pilot_parser = subparsers.add_parser(
        "pilot", help="run the split, preprocessing, and Grid mapping checkpoint"
    )
    pilot_parser.add_argument("--config", default="configs/pilot.yaml")
    pilot_parser.add_argument("--output-dir", default="artifacts")
    pilot_parser.set_defaults(function=pilot)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
