"""First fixed Spambase Grid training/validation condition. Test evaluation is closed."""

import json
import platform
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from css700.data.spambase_grid import load_spambase_grid_inputs
from css700.experiments.som_mapping_audit import _git_state
from css700.experiments.spambase_audit import sha256, write_json
from css700.models import SmallTabularImageCNN
from css700.training.grid_cnn import _loader, evaluate_classifier
from css700.training.validation_cnn import train_validation_cnn

TRAINING_POLICY = {
    "model_seed": 42,
    "planned_model_seeds": [7, 42, 2026],
    "architecture": "SmallTabularImageCNN",
    "device": "cpu",
    "threads": 1,
    "epochs": 100,
    "batch_size": 32,
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "patience": 12,
    "max_seconds": 600.0,
    "selection": "validation_row_macro_f1",
    "tie_policy": "earliest_with_1e-12_tolerance",
    "training_weights": "one_per_original_row",
    "refit_scaler": False,
    "test_evaluation": False,
    "validation_sensitivity": "one_over_feature_group_size",
    "explanations": False,
}


def validation_scores(labels, predictions, weights=None):
    return {
        "accuracy": float(accuracy_score(labels, predictions, sample_weight=weights)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=[0, 1],
                average="macro",
                sample_weight=weights,
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1], sample_weight=weights
        ).tolist(),
    }


def validate_training_config(config):
    if (
        set(config) != {"protocol", "preflight_review", "preflight_review_sha256", "training"}
        or config["protocol"] != "spambase_grid_train_validation_v1"
        or config["training"] != TRAINING_POLICY
    ):
        raise ValueError(
            "frozen training/validation policy changed; a versioned amendment is required"
        )


def run_spambase_grid(config_path: Path, output_dir: Path) -> dict:
    started = time.perf_counter()
    commit, dirty = _git_state()
    root = output_dir / (
        datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
        + "__spambase__grid__model-seed-42__train-validation-v1"
    )
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "running",
        "commit_sha": commit,
        "worktree_dirty_at_start": dirty,
        "scope": "training_validation_only_test_closed",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {n: version(n) for n in ("numpy", "scikit-learn", "torch")},
        "device": "cpu",
        "torch_threads_during_training": 1,
    }
    write_json(root / "manifest.json", manifest)
    hashes, checks, errors, results = {}, {"clean_worktree": not dirty}, [], {}
    completed_model = False
    try:
        if dirty:
            raise ValueError("training requires a clean prospective code/config commit")
        payload = config_path.read_bytes()
        hashes[str(config_path.resolve())] = sha256(payload)
        (root / "config.yaml").write_bytes(payload)
        config = yaml.safe_load(payload)
        validate_training_config(config)
        inputs = load_spambase_grid_inputs(
            Path(config["preflight_review"]), config["preflight_review_sha256"], hashes
        )
        if not all(sha256(Path(p).read_bytes()) == h for p, h in hashes.items()):
            raise ValueError("an input changed before training")
        checks["reviewed_split_scaler_and_mapping_verified"] = True
        a = inputs.arrays
        np.savez_compressed(root / "inputs.npz", **a)
        write_json(root / "feature_map.json", inputs.feature_map)
        policy = config["training"]
        results.update(
            split_fingerprint=inputs.split_fingerprint,
            partition_counts=inputs.partition_counts,
            model_seed=policy["model_seed"],
            target_names=["non-spam", "spam"],
            train_class_counts=np.bincount(a["y_train"], minlength=2).tolist(),
            validation_class_counts=np.bincount(a["y_validation"], minlength=2).tolist(),
            validation_feature_groups=int(round(a["validation_group_weights"].sum())),
        )

        def save_epoch(history, model, improved):
            write_json(root / "history.json", history)
            if improved:
                torch.save(
                    {
                        "model_class": "SmallTabularImageCNN",
                        "model_state_dict": model.state_dict(),
                        "input_channels": 1,
                        "num_classes": 2,
                        "best_epoch": history[-1]["epoch"],
                        "model_seed": policy["model_seed"],
                        "split_fingerprint": inputs.split_fingerprint,
                        "config_sha256": sha256(payload),
                    },
                    root / "best_checkpoint.pt",
                )

        training_started = time.perf_counter()
        outcome = train_validation_cnn(
            a["train_images"],
            a["y_train"],
            a["validation_images"],
            a["y_validation"],
            seed=policy["model_seed"],
            epochs=policy["epochs"],
            batch_size=policy["batch_size"],
            learning_rate=policy["learning_rate"],
            weight_decay=policy["weight_decay"],
            patience=policy["patience"],
            max_seconds=policy["max_seconds"],
            on_epoch=save_epoch,
        )
        results["training_runtime_seconds"] = time.perf_counter() - training_started
        completed_model = True
        checkpoint = torch.load(root / "best_checkpoint.pt", weights_only=True, map_location="cpu")
        # Use identical batch size/thread settings for the reload equality check.
        old_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with torch.random.fork_rng(devices=[]):
                reloaded = SmallTabularImageCNN(input_channels=1, num_classes=2)
                reloaded.load_state_dict(checkpoint["model_state_dict"])
                repeated = evaluate_classifier(
                    reloaded,
                    _loader(
                        a["validation_images"],
                        a["y_validation"],
                        batch_size=policy["batch_size"],
                        shuffle=False,
                        seed=policy["model_seed"],
                    ),
                    device=torch.device("cpu"),
                    num_classes=2,
                )
        finally:
            torch.set_num_threads(old_threads)
        pred = outcome.validation
        np.savez_compressed(
            root / "validation_predictions.npz",
            sample_ids=a["validation_indices"],
            labels=a["y_validation"],
            logits=pred.logits,
            probabilities=pred.probabilities,
            predictions=pred.predictions,
            group_equal_weights=a["validation_group_weights"],
        )
        majority_class = int(np.bincount(a["y_train"]).argmax())
        majority = np.full_like(a["y_validation"], majority_class)
        primary = validation_scores(a["y_validation"], pred.predictions)
        baseline = validation_scores(a["y_validation"], majority)
        # Replay the exact selection rule instead of interpreting max epoch as best epoch.
        best_score, best_epoch, stale, replay_ok = -1.0, 0, 0, True
        for position, row in enumerate(outcome.history, 1):
            improved = row["validation_macro_f1"] > best_score + 1e-12
            replay_ok = (
                replay_ok and row["epoch"] == position and row["selected_new_best"] == improved
            )
            if improved:
                best_score, best_epoch = row["validation_macro_f1"], row["epoch"]
                stale = 0
            else:
                stale += 1
            if position < len(outcome.history) and stale >= policy["patience"]:
                replay_ok = False
        checks.update(
            finite_history=bool(
                all(np.isfinite(v) for row in outcome.history for v in row.values())
            ),
            finite_probabilities=bool(
                np.isfinite(pred.probabilities).all()
                and np.allclose(pred.probabilities.sum(axis=1), 1, rtol=0, atol=1e-6)
            ),
            finite_checkpoint=all(
                bool(torch.isfinite(v).all()) for v in checkpoint["model_state_dict"].values()
            ),
            training_changed_parameters=outcome.parameters_changed,
            validation_selected_checkpoint=(
                checkpoint["best_epoch"] == best_epoch == outcome.best_epoch
                and abs(primary["macro_f1"] - best_score) <= 1e-12
            ),
            checkpoint_reload_exact=bool(np.array_equal(repeated.logits, pred.logits)),
            epoch_budget=len(outcome.history) <= policy["epochs"],
            selection_and_stopping_rule=replay_ok
            and (
                (stale == policy["patience"] and outcome.stop_reason == "early_stopping_patience")
                or (
                    len(outcome.history) == policy["epochs"]
                    and outcome.stop_reason == "epoch_limit"
                )
            ),
            optimizer_step_count=outcome.optimizer_steps
            == len(outcome.history) * int(np.ceil(len(a["y_train"]) / policy["batch_size"])),
        )
        results.update(
            epochs_completed=len(outcome.history),
            best_epoch=outcome.best_epoch,
            optimizer_steps=outcome.optimizer_steps,
            stop_reason=outcome.stop_reason,
            validation_loss=pred.loss,
            validation_primary=primary,
            validation_group_equal=validation_scores(
                a["y_validation"], pred.predictions, a["validation_group_weights"]
            ),
            majority_class_from_training=majority_class,
            validation_majority_primary=baseline,
            validation_majority_group_equal=validation_scores(
                a["y_validation"], majority, a["validation_group_weights"]
            ),
            validation_diagnostics={
                "predicted_class_counts": np.bincount(pred.predictions, minlength=2).tolist(),
                "both_classes_predicted": len(np.unique(pred.predictions)) == 2,
                "macro_f1_exceeds_majority": primary["macro_f1"] > baseline["macro_f1"],
                "accuracy_exceeds_majority": primary["accuracy"] > baseline["accuracy"],
                "interpretation": (
                    "Validation was used for checkpoint selection; "
                    "not an unbiased final performance estimate."
                ),
            },
            checkpoint_reload_max_abs_difference=float(np.abs(repeated.logits - pred.logits).max()),
        )
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    try:
        checks["all_inputs_unchanged"] = all(
            sha256(Path(p).read_bytes()) == h for p, h in hashes.items()
        )
    except OSError as error:
        checks["all_inputs_unchanged"] = False
        errors.append(str(error))
    # Preserve completed-epoch evidence, including when a timeout interrupts a later epoch.
    if (root / "history.json").exists() and "optimizer_steps" not in results:
        partial = json.loads((root / "history.json").read_text())
        results["partial_completed_epochs"] = len(partial)
        results["optimizer_steps_through_last_completed_epoch"] = partial[-1]["optimizer_steps"]
    accepted = not errors and all(checks.values())
    summary = {
        "status": "ok" if accepted else "quality_gate_failed",
        "output": str(root),
        "checks": checks,
        "errors": errors,
        **results,
        "decision": "await_independent_training_review" if accepted else "hold_training_review",
        "models_completed": int(completed_model),
        "test_rows_transformed_or_scored": 0,
        "test_evaluation_ready": False,
        "explanations_generated": 0,
        "scalers_refitted": 0,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(root / "summary.json", summary)
    manifest.update(
        status=summary["status"],
        completed_at_utc=datetime.now(UTC).isoformat(),
        source_file_sha256=hashes,
        output_file_sha256={
            p.name: sha256(p.read_bytes())
            for p in sorted(root.iterdir())
            if p.is_file() and p.name != "manifest.json"
        },
    )
    write_json(root / "manifest.json", manifest)
    return summary
