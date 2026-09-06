"""End-to-end artifact writer for the Breast Cancer Grid CNN pilot."""

import csv
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import yaml

from css700.config import resolve_experiment_seeds
from css700.data import load_breast_cancer_splits
from css700.encoding import GridEncoder
from css700.models import SmallTabularImageCNN
from css700.training import TrainingOutcome, train_grid_cnn

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "uncommitted"


def _git_worktree_dirty() -> bool:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return True


def _package_versions() -> dict[str, str]:
    distributions = (
        "numpy",
        "pandas",
        "PyYAML",
        "scikit-learn",
        "scipy",
        "torch",
        "captum",
        "css700",
    )
    versions: dict[str, str] = {}
    for distribution in distributions:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = "missing"
    return versions


def _write_history(path: Path, outcome: TrainingOutcome) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(outcome.history[0]))
        writer.writeheader()
        writer.writerows(outcome.history)


def _write_predictions(
    path: Path,
    *,
    sample_ids: np.ndarray,
    labels: np.ndarray,
    outcome: TrainingOutcome,
) -> None:
    fieldnames = [
        "sample_id",
        "true_label",
        "predicted_label",
        "correct",
        *[f"probability_class_{index}" for index in range(outcome.test.probabilities.shape[1])],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for position, sample_id in enumerate(sample_ids):
            row: dict[str, int | float] = {
                "sample_id": int(sample_id),
                "true_label": int(labels[position]),
                "predicted_label": int(outcome.test.predictions[position]),
                "correct": int(labels[position] == outcome.test.predictions[position]),
            }
            row.update(
                {
                    f"probability_class_{class_index}": float(probability)
                    for class_index, probability in enumerate(outcome.test.probabilities[position])
                }
            )
            writer.writerow(row)


def _write_figures(
    figures_dir: Path, outcome: TrainingOutcome, target_names: tuple[str, ...]
) -> None:
    epochs = [int(row["epoch"]) for row in outcome.history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [float(row["train_loss"]) for row in outcome.history], label="train")
    axes[0].plot(
        epochs,
        [float(row["validation_loss"]) for row in outcome.history],
        label="validation",
    )
    axes[0].axvline(outcome.best_epoch, color="black", linestyle="--", label="selected")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Training history")
    axes[0].legend()

    matrix = np.asarray(outcome.test.metrics["confusion_matrix"])
    image = axes[1].imshow(matrix, cmap="Blues")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes[1].text(column, row, str(matrix[row, column]), ha="center", va="center")
    axes[1].set(
        xticks=range(len(target_names)),
        yticks=range(len(target_names)),
        xticklabels=target_names,
        yticklabels=target_names,
        xlabel="Predicted class",
        ylabel="True class",
        title="Test confusion matrix",
    )
    figure.colorbar(image, ax=axes[1], fraction=0.046)
    figure.tight_layout()
    figure.savefig(figures_dir / "training_and_confusion_matrix.png", dpi=180)
    plt.close(figure)


def run_grid_cnn_experiment(config_path: Path, output_dir: Path) -> dict[str, Any]:
    """Train, select, evaluate, and write one immutable Grid CNN run bundle."""

    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    worktree_dirty_at_start = _git_worktree_dirty()
    seeds = resolve_experiment_seeds(config)
    dataset_config = config["dataset"]
    model_config = config["model"]
    bundle = load_breast_cancer_splits(
        seed=seeds.data_split_seed,
        test_size=float(dataset_config["test_size"]),
        validation_size=float(dataset_config["validation_size"]),
        stratify=bool(dataset_config["stratify"]),
    )
    encoder = GridEncoder(padding_value=float(config["encoding"]["padding_value"]))
    train_images = encoder.fit_transform(bundle.x_train, bundle.feature_names)
    validation_images = encoder.transform(bundle.x_validation)
    test_images = encoder.transform(bundle.x_test)

    started_at = datetime.now(UTC)
    seed_label = (
        f"split-seed-{seeds.data_split_seed}__model-seed-{seeds.model_seed}__replicate"
        if seeds.explicit_separation
        else f"seed-{seeds.model_seed}__baseline"
    )
    run_id = f"{started_at.strftime('%Y%m%d-%H%M%S')}__breast-cancer__grid__small-cnn__{seed_label}"
    run_dir = output_dir / run_id
    directories = {
        name: run_dir / name
        for name in (
            "attributions",
            "encoding",
            "figures",
            "metrics",
            "model",
            "preprocessing",
            "splits",
        )
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=False)

    outcome = train_grid_cnn(
        train_images,
        bundle.y_train,
        validation_images,
        bundle.y_validation,
        test_images,
        bundle.y_test,
        seed=seeds.model_seed,
        epochs=int(model_config["epochs"]),
        batch_size=int(model_config["batch_size"]),
        learning_rate=float(model_config["learning_rate"]),
        early_stopping_patience=int(model_config["early_stopping_patience"]),
        weight_decay=float(model_config.get("weight_decay", 0.0)),
        device_name=str(model_config.get("device", "cpu")),
    )

    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with (directories["encoding"] / "feature_map.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        records = encoder.mapping_records()
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    np.save(directories["encoding"] / "padding_mask.npy", encoder.padding_mask())
    np.savez_compressed(
        directories["splits"] / "indices.npz",
        train=bundle.train_indices,
        validation=bundle.validation_indices,
        test=bundle.test_indices,
    )
    preprocessing = {
        "fit_split": "training",
        "method": "StandardScaler",
        "feature_names": list(bundle.feature_names),
        "mean": bundle.scaler_mean.tolist(),
        "scale": bundle.scaler_scale.tolist(),
    }
    (directories["preprocessing"] / "standard_scaler.json").write_text(
        json.dumps(preprocessing, indent=2) + "\n", encoding="utf-8"
    )

    checkpoint_path = directories["model"] / "best_checkpoint.pt"
    torch.save(
        {
            "model_class": "SmallTabularImageCNN",
            "model_state_dict": outcome.model.state_dict(),
            "input_channels": int(train_images.shape[1]),
            "num_classes": len(bundle.target_names),
            "best_epoch": outcome.best_epoch,
            "seed": seeds.model_seed,
            "data_split_seed": seeds.data_split_seed,
            "model_seed": seeds.model_seed,
        },
        checkpoint_path,
    )
    reload_model = SmallTabularImageCNN(
        input_channels=train_images.shape[1], num_classes=len(bundle.target_names)
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    reload_model.load_state_dict(checkpoint["model_state_dict"])
    reload_model.eval()
    with torch.no_grad():
        reloaded_logits = reload_model(torch.from_numpy(test_images)).numpy()
    reload_difference = float(np.max(np.abs(reloaded_logits - outcome.test.logits)))

    _write_history(directories["model"] / "training_history.csv", outcome)
    _write_predictions(
        directories["model"] / "test_predictions.csv",
        sample_ids=bundle.test_indices,
        labels=bundle.y_test,
        outcome=outcome,
    )
    quality_gate = {
        "beats_majority_accuracy": bool(
            outcome.test.metrics["accuracy"] > outcome.majority_metrics["accuracy"]
        ),
        "beats_majority_macro_f1": bool(
            outcome.test.metrics["macro_f1"] > outcome.majority_metrics["macro_f1"]
        ),
        "checkpoint_reload_exact_within_1e-7": reload_difference <= 1e-7,
        "checkpoint_reload_max_abs_logit_difference": reload_difference,
        "working_tree_clean_at_start": not worktree_dirty_at_start,
    }
    quality_gate_passed = bool(
        quality_gate["beats_majority_accuracy"]
        and quality_gate["beats_majority_macro_f1"]
        and quality_gate["checkpoint_reload_exact_within_1e-7"]
        and quality_gate["working_tree_clean_at_start"]
    )
    metrics = {
        "selection_rule": "highest validation macro F1; first epoch wins ties",
        "best_epoch": outcome.best_epoch,
        "epochs_completed": len(outcome.history),
        "validation": {"loss": outcome.validation.loss, **outcome.validation.metrics},
        "test": {"loss": outcome.test.loss, **outcome.test.metrics},
        "majority_baseline": {
            "class": outcome.majority_class,
            **outcome.majority_metrics,
        },
        "quality_gate": quality_gate,
    }
    (directories["metrics"] / "predictive_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    _write_figures(directories["figures"], outcome, bundle.target_names)

    completed_at = datetime.now(UTC)
    manifest = {
        "run_id": run_id,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "duration_seconds": (completed_at - started_at).total_seconds(),
        "commit_sha": _git_sha(),
        "git_worktree_dirty_at_start": worktree_dirty_at_start,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "data_split_seed": seeds.data_split_seed,
        "model_seed": seeds.model_seed,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(model_config.get("device", "cpu")),
        "packages": _package_versions(),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (run_dir / "notes.md").write_text(
        "# Run notes\n\n"
        "This is the validation-selected Breast Cancer Wisconsin Grid CNN pilot. "
        "The test split was evaluated after checkpoint selection. Grad-CAM outputs are added "
        "by the separate explanation command.\n",
        encoding="utf-8",
    )
    return {
        "status": "ok" if quality_gate_passed else "quality_gate_failed",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "best_epoch": outcome.best_epoch,
        "validation_macro_f1": outcome.validation.metrics["macro_f1"],
        "test_accuracy": outcome.test.metrics["accuracy"],
        "test_macro_f1": outcome.test.metrics["macro_f1"],
        "majority_accuracy": outcome.majority_metrics["accuracy"],
        "quality_gate": quality_gate,
        "data_split_seed": seeds.data_split_seed,
        "model_seed": seeds.model_seed,
    }
