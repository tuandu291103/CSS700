"""Append-only train-only SOM mapping diagnostic; never a CNN result."""

import csv
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from css700.data import load_breast_cancer_splits
from css700.data.splits import split_fingerprint
from css700.encoding.mapping import audit_feature_mapping
from css700.encoding.som_diagnostic import fit_som_diagnostic


def _git_state() -> tuple[str, bool]:
    repository = Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repository, text=True, stderr=subprocess.DEVNULL
        ).strip()
        return sha, bool(dirty)
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted", True


def _write_figure(path: Path, records: list[dict[str, Any]], rows: int, columns: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    counts = np.zeros((rows, columns), dtype=int)
    for record in records:
        counts[record["row"], record["column"]] += 1
    figure, axis = plt.subplots(figsize=(7, 7))
    plot = axis.imshow(counts, cmap="YlOrRd", vmin=0, vmax=max(1, int(counts.max())))
    for row in range(rows):
        for column in range(columns):
            indices = [
                str(record["feature_index"])
                for record in records
                if record["row"] == row and record["column"] == column
            ]
            axis.text(
                column,
                row,
                ", ".join(indices) or "empty",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if counts[row, column] >= 2 else "black",
            )
    axis.set(
        title="Provisional SOM: native BMU assignments\nCell text = original feature indices",
        xlabel="Column",
        ylabel="Row",
        xticks=range(columns),
        yticks=range(rows),
    )
    axis.set_xticks(np.arange(-0.5, columns, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1)
    axis.tick_params(which="minor", bottom=False, left=False)
    figure.colorbar(
        plot, ax=axis, label="Features assigned to pixel", ticks=range(counts.max() + 1)
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_som_mapping_audit(config_path: Path, output_dir: Path) -> dict[str, Any]:
    """Inspect native collisions with fixed provisional settings and zero outcome tuning."""

    started = time.perf_counter()
    sha, dirty = _git_state()
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    if config["dataset"]["name"] != "breast_cancer":
        raise ValueError("only the registered breast_cancer diagnostic is supported")
    experiment, dataset = config["experiment"], config["dataset"]
    bundle = load_breast_cancer_splits(
        seed=int(experiment["data_split_seed"]),
        test_size=float(dataset["test_size"]),
        validation_size=float(dataset["validation_size"]),
        stratify=bool(dataset["stratify"]),
    )
    fingerprint = split_fingerprint(
        bundle.train_indices, bundle.validation_indices, bundle.test_indices
    )
    parameters = {**config["som"], "layout_seed": int(experiment["layout_seed"])}
    # Only training features enter layout fitting. Held-out values and all labels are unused.
    result = fit_som_diagnostic(bundle.x_train, bundle.feature_names, **parameters)
    repeated = fit_som_diagnostic(bundle.x_train, bundle.feature_names, **parameters)
    rows, columns = result.weights.shape[:2]
    audit = audit_feature_mapping(result.mapping, (1, rows, columns))
    checks = {
        "worktree_clean_at_start": not dirty,
        "split_matches_grid_reference": fingerprint == experiment["expected_split_fingerprint"],
        "finite_weights_and_error": bool(
            np.isfinite(result.weights).all() and np.isfinite(result.quantization_error)
        ),
        "same_seed_exact_weights": bool(np.array_equal(result.weights, repeated.weights)),
        "same_seed_exact_mapping": result.mapping == repeated.mapping,
        "all_features_accounted_for": audit["feature_count"] == len(bundle.feature_names),
        "training_scaled_mean_near_zero": bool(np.max(np.abs(bundle.x_train.mean(axis=0))) < 1e-10),
    }
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    run_id = f"{timestamp}__breast-cancer__som-bmu__layout-seed-{parameters['layout_seed']}__audit"
    root = output_dir / run_id
    root.mkdir(parents=True, exist_ok=False)
    for directory in ("encoding", "splits", "preprocessing", "figures"):
        (root / directory).mkdir()
    (root / "config.yaml").write_bytes(config_bytes)
    records = [asdict(pixel) for pixel in result.mapping]
    with (root / "encoding" / "candidate_bmu_map.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    np.save(root / "encoding" / "som_weights.npy", result.weights)
    np.savez_compressed(
        root / "splits" / "indices.npz",
        train=bundle.train_indices,
        validation=bundle.validation_indices,
        test=bundle.test_indices,
    )
    np.savez_compressed(
        root / "preprocessing" / "standard_scaler.npz",
        mean=bundle.scaler_mean,
        scale=bundle.scaler_scale,
    )
    _write_figure(root / "figures" / "bmu_occupancy.png", records, rows, columns)
    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "commit_sha": sha,
        "worktree_dirty_at_start": dirty,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: metadata.version(name)
            for name in ("numpy", "scikit-learn", "MiniSom", "PyYAML", "matplotlib", "css700")
        },
        "split_fingerprint": fingerprint,
        "training_values_sha256": hashlib.sha256(
            np.ascontiguousarray(bundle.x_train, dtype="<f8").tobytes()
        ).hexdigest(),
        "layout_fit_partition": "train_only",
        "layout_fit_shape": list(bundle.x_train.T.shape),
        "scaling": "training-only StandardScaler; diagnostic Grid parity, not paper MinMax",
        "initialization": "random training-feature vectors",
        "topology": "rectangular",
        "activation_distance": "euclidean",
        "neighborhood": "gaussian",
        "learning_and_sigma_decay": "asymptotic_decay",
        "random_order": True,
        "use_epochs": False,
        "collision_resolution": "none",
        "author_implementation_verified": False,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "quality_gate_failed",
        "scope": "diagnostic_only_not_som2img_reproduction",
        "output": str(root),
        "quality_checks": checks,
        "som2img_cnn_ready": False,
        "unresolved": [
            "author code and hyperparameters",
            "collision policy",
            "scaling and orientation",
        ],
        "grid": {"rows": rows, "columns": columns},
        "quantization_error": result.quantization_error,
        "mapping_audit": audit,
        "runtime_seconds_including_reproducibility_refit": time.perf_counter() - started,
    }
    for filename, content in (("manifest.json", manifest), ("summary.json", summary)):
        (root / filename).write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    return summary
