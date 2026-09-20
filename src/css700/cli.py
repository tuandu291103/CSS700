"""Command-line entry points for environment checks and the first pilot checkpoint."""

import argparse
import csv
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
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


def _configure_runtime_cache() -> None:
    cache_root = Path(tempfile.gettempdir()) / "css700-runtime-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))


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


def train_grid(args: argparse.Namespace) -> int:
    """Train and evaluate the validation-selected Grid CNN baseline."""

    _configure_runtime_cache()
    from css700.experiments import run_grid_cnn_experiment

    summary = run_grid_cnn_experiment(Path(args.config), Path(args.output_dir))
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "ok" else 1


def explain_grad_cam(args: argparse.Namespace) -> int:
    """Generate and trace predicted-class Grad-CAM maps for a frozen Grid CNN run."""

    _configure_runtime_cache()
    from css700.experiments import run_grad_cam_experiment

    summary = run_grad_cam_experiment(Path(args.run_dir), batch_size=int(args.batch_size))
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "ok" else 1


def explain_integrated_gradients(args: argparse.Namespace) -> int:
    """Generate Integrated Gradients and compare them with frozen Grad-CAM outputs."""

    _configure_runtime_cache()
    from css700.experiments import run_integrated_gradients_experiment

    summary = run_integrated_gradients_experiment(Path(args.run_dir), Path(args.config))
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "ok" else 1


def explain_occlusion(args: argparse.Namespace) -> int:
    """Generate feature-wise occlusion scores and compare all frozen explanation outputs."""

    _configure_runtime_cache()
    from css700.experiments import run_occlusion_experiment

    summary = run_occlusion_experiment(Path(args.run_dir), Path(args.config))
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "ok" else 1


def evaluate_faithfulness(args: argparse.Namespace) -> int:
    """Evaluate three frozen explanation rankings with controlled cumulative deletion."""

    _configure_runtime_cache()
    from css700.experiments import run_faithfulness_experiment

    summary = run_faithfulness_experiment(Path(args.run_dir), Path(args.config))
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "ok" else 1


def analyze_grid_stability(args: argparse.Namespace) -> int:
    """Analyze fixed-split predictive, explanation, and faithfulness stability."""

    _configure_runtime_cache()
    from css700.experiments import run_grid_stability_experiment

    summary = run_grid_stability_experiment(
        [Path(path) for path in args.run_dir],
        Path(args.config),
        Path(args.output_dir),
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "ok" else 1


def train_tabular_baselines(args: argparse.Namespace) -> int:
    """Train and compare validation-selected Random Forest and XGBoost baselines."""

    _configure_runtime_cache()
    from css700.experiments import run_tabular_baselines_experiment

    summary = run_tabular_baselines_experiment(Path(args.config), Path(args.output_dir))
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "ok" else 1


def audit_som_mapping(args: argparse.Namespace) -> int:
    """Audit provisional train-only SOM pixel assignments without training a CNN."""

    _configure_runtime_cache()
    from css700.experiments.som_mapping_audit import run_som_mapping_audit

    summary = run_som_mapping_audit(Path(args.config), Path(args.output_dir))
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "ok" else 1


def explain_trees(args: argparse.Namespace) -> int:
    """Explain registered frozen tree models without importing a CNN."""

    _configure_runtime_cache()
    from css700.experiments.tree_shap import run_tree_shap_experiment

    summary = run_tree_shap_experiment(Path(args.config), Path(args.output_dir))
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "ok" else 1


def analyze_tree_stability(args: argparse.Namespace) -> int:
    _configure_runtime_cache()
    from css700.experiments.tree_shap_stability import run_tree_shap_stability

    summary = run_tree_shap_stability(
        [Path(path) for path in args.run_dir], Path(args.config), Path(args.output_dir)
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "ok" else 1


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
    train_parser = subparsers.add_parser(
        "train-grid", help="train and evaluate the Breast Cancer Grid CNN"
    )
    train_parser.add_argument("--config", default="configs/grid_cnn.yaml")
    train_parser.add_argument("--output-dir", default="artifacts")
    train_parser.set_defaults(function=train_grid)
    grad_cam_parser = subparsers.add_parser(
        "grad-cam", help="explain a frozen Grid CNN run with predicted-class Grad-CAM"
    )
    grad_cam_parser.add_argument("--run-dir", required=True)
    grad_cam_parser.add_argument("--batch-size", type=int, default=32)
    grad_cam_parser.set_defaults(function=explain_grad_cam)
    integrated_gradients_parser = subparsers.add_parser(
        "integrated-gradients",
        help="trace Integrated Gradients and compare them with Grad-CAM",
    )
    integrated_gradients_parser.add_argument("--run-dir", required=True)
    integrated_gradients_parser.add_argument(
        "--config", default="configs/integrated_gradients.yaml"
    )
    integrated_gradients_parser.set_defaults(function=explain_integrated_gradients)
    occlusion_parser = subparsers.add_parser(
        "occlusion",
        help="trace feature-wise Occlusion Sensitivity and compare all explanation methods",
    )
    occlusion_parser.add_argument("--run-dir", required=True)
    occlusion_parser.add_argument("--config", default="configs/occlusion.yaml")
    occlusion_parser.set_defaults(function=explain_occlusion)
    faithfulness_parser = subparsers.add_parser(
        "faithfulness",
        help="evaluate frozen explanation rankings with cumulative feature deletion",
    )
    faithfulness_parser.add_argument("--run-dir", required=True)
    faithfulness_parser.add_argument("--config", default="configs/faithfulness.yaml")
    faithfulness_parser.set_defaults(function=evaluate_faithfulness)
    stability_parser = subparsers.add_parser(
        "grid-stability",
        help="compare fixed-split Grid runs across model-training seeds",
    )
    stability_parser.add_argument("--run-dir", action="append", required=True)
    stability_parser.add_argument("--config", default="configs/grid_stability.yaml")
    stability_parser.add_argument("--output-dir", default="artifacts")
    stability_parser.set_defaults(function=analyze_grid_stability)
    tabular_parser = subparsers.add_parser(
        "train-tabular-baselines",
        help="train fixed-split Random Forest and XGBoost baselines",
    )
    tabular_parser.add_argument("--config", default="configs/tabular_baselines.yaml")
    tabular_parser.add_argument("--output-dir", default="artifacts")
    tabular_parser.set_defaults(function=train_tabular_baselines)
    som_parser = subparsers.add_parser(
        "audit-som-mapping",
        help="audit native SOM collisions with provisional training-only settings",
    )
    som_parser.add_argument("--config", default="configs/som_mapping_audit.yaml")
    som_parser.add_argument("--output-dir", default="artifacts")
    som_parser.set_defaults(function=audit_som_mapping)
    tree_parser = subparsers.add_parser("tree-shap", help="explain registered frozen tree models")
    tree_parser.add_argument("--config", default="configs/tree_shap_seed_42_v1_1.yaml")
    tree_parser.add_argument("--output-dir", default="artifacts")
    tree_parser.set_defaults(function=explain_trees)
    tree_stability = subparsers.add_parser(
        "tree-shap-stability", help="compare accepted TreeSHAP seeds"
    )
    tree_stability.add_argument("--run-dir", action="append", required=True)
    tree_stability.add_argument("--config", default="configs/tree_shap_stability.yaml")
    tree_stability.add_argument("--output-dir", default="artifacts")
    tree_stability.set_defaults(function=analyze_tree_stability)
    lime_parser = subparsers.add_parser(
        "lime-preflight", help="check LIME on training anchors only"
    )
    lime_parser.add_argument("--config", default="configs/lime_seed_42_v1.yaml")
    lime_parser.add_argument("--output-dir", default="artifacts")
    lime_parser.set_defaults(function=lime_preflight)
    locality = subparsers.add_parser(
        "lime-locality", help="run the fixed training-only LIME diagnostic"
    )
    locality.add_argument("--config", default="configs/lime_locality_diagnostic_v1.yaml")
    locality.add_argument("--output-dir", default="artifacts")
    locality.set_defaults(function=lime_locality)
    lime_test_parser = subparsers.add_parser(
        "lime-test", help="run the frozen LIME test pilot and fit-aware SHAP comparisons"
    )
    lime_test_parser.add_argument("--config", default="configs/lime_seed_42_v1.yaml")
    lime_test_parser.add_argument("--output-dir", default="artifacts")
    lime_test_parser.set_defaults(function=lime_test)
    fetch_parser = subparsers.add_parser(
        "fetch-spambase", help="pin the official UCI source archive"
    )
    fetch_parser.add_argument("--output-dir", default="data/raw")
    fetch_parser.set_defaults(function=fetch_spambase_source)
    audit_parser = subparsers.add_parser(
        "audit-spambase", help="audit pinned Spambase without splitting or training"
    )
    audit_parser.add_argument("--source-dir", required=True)
    audit_parser.add_argument("--source-lock", default="configs/spambase_source_v1.yaml")
    audit_parser.add_argument("--output-dir", default="artifacts")
    audit_parser.set_defaults(function=audit_spambase_source)
    split_parser = subparsers.add_parser(
        "split-spambase", help="freeze verified whole-feature-group partitions without training"
    )
    split_parser.add_argument("--config", default="configs/spambase_split_v1.yaml")
    split_parser.add_argument("--output-dir", default="artifacts")
    split_parser.set_defaults(function=split_spambase_source)
    split_v2_parser = subparsers.add_parser(
        "split-spambase-v2", help="run the corrected grouped allocation before scientific review"
    )
    split_v2_parser.add_argument("--config", default="configs/spambase_split_v2.yaml")
    split_v2_parser.add_argument("--output-dir", default="artifacts")
    split_v2_parser.set_defaults(function=split_spambase_source_v2)
    return parser


def lime_preflight(args: argparse.Namespace) -> int:
    """Check the frozen LIME design on training anchors only."""
    _configure_runtime_cache()
    from css700.experiments.lime_preflight import run_lime_preflight

    summary = run_lime_preflight(Path(args.config), Path(args.output_dir))
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "ok" else 1


def lime_locality(args: argparse.Namespace) -> int:
    """Reweight/refit saved training neighbors without new model queries."""
    _configure_runtime_cache()
    from css700.experiments.lime_locality import run_lime_locality

    summary = run_lime_locality(Path(args.config), Path(args.output_dir))
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "ok" else 1


def lime_test(args: argparse.Namespace) -> int:
    _configure_runtime_cache()
    from css700.experiments.lime_test import run_lime_test

    summary = run_lime_test(Path(args.config), Path(args.output_dir))
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "ok" else 1


def fetch_spambase_source(args: argparse.Namespace) -> int:
    from css700.experiments.spambase_audit import fetch_spambase

    summary = fetch_spambase(Path(args.output_dir))
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "ok" else 1


def audit_spambase_source(args: argparse.Namespace) -> int:
    from css700.experiments.spambase_audit import run_spambase_audit

    summary = run_spambase_audit(
        Path(args.source_dir), Path(args.source_lock), Path(args.output_dir)
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "ok" else 1


def split_spambase_source(args: argparse.Namespace) -> int:
    from css700.experiments.spambase_split import run_spambase_split

    summary = run_spambase_split(Path(args.config), Path(args.output_dir))
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "ok" else 1


def split_spambase_source_v2(args: argparse.Namespace) -> int:
    from css700.experiments.spambase_split_v2 import run_spambase_split_v2

    summary = run_spambase_split_v2(Path(args.config), Path(args.output_dir))
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "ok" else 1


def main() -> int:
    args = build_parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
