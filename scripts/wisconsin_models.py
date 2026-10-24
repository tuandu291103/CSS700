"""Experiment 02: fixed-protocol tabular baselines and a small grid CNN."""

import argparse
import copy
import hashlib
import json
import platform
import random
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler
from torch import nn

from scripts.wisconsin_learning import FEATURES, ROOT, encode, load_data, verify_sources

PROTOCOL_PATH = ROOT / "experiments/02_wisconsin_models/protocol.json"
METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "log_loss")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs(protocol, root=ROOT):
    """Verify the frozen audit, then reuse its IDs, mapping, and training extrema."""
    audit_path = root / protocol["input_audit"]
    if sha256(audit_path) != protocol["input_audit_sha256"]:
        raise ValueError("Experiment 01 audit changed; review before reusing the split")
    audit = json.loads(audit_path.read_text())
    verify_sources(root / "data/raw/wdbc")
    frame = load_data(root / "data/raw/wdbc/wdbc.data")
    raw = frame.loc[:, FEATURES].to_numpy(dtype=np.float64)
    labels = frame.diagnosis.map(protocol["label_encoding"]).to_numpy(dtype=np.int64)
    splits = {name: np.asarray(part["row_indices_zero_based"], dtype=int)
              for name, part in audit["splits"].items()}
    joined = np.concatenate(list(splits.values()))
    if not np.array_equal(np.sort(joined), np.arange(len(frame))):
        raise ValueError("Partitions must be disjoint and cover every record exactly once")
    for name, indices in splits.items():
        if frame.sample_id.iloc[indices].tolist() != audit["splits"][name]["sample_ids"]:
            raise ValueError("Row identity/order no longer matches the recorded split")
    scaler = MinMaxScaler(clip=False).fit(raw[splits["train"]])
    for name, actual in (("training_min", scaler.data_min_), ("training_max", scaler.data_max_)):
        np.testing.assert_array_equal(actual, [audit["scaling"][name][f] for f in FEATURES])
    scaled64 = scaler.transform(raw)
    values = scaled64.astype(np.float32)
    mapping = audit["encoding"]["mapping"]
    images, _ = encode(values, mapping, tuple(audit["encoding"]["shape"]))
    images = images.astype(np.float32)[:, None, :, :]
    # Row-major baseline: reshaping the numeric grid must recover identical model inputs.
    np.testing.assert_array_equal(images.reshape(len(frame), 30), values)
    data = {name: {"tabular": values[indices], "images": images[indices], "y": labels[indices],
                   "sample_ids": frame.sample_id.iloc[indices].tolist()}
            for name, indices in splits.items()}
    metadata = {
        "source_hashes": {name: entry["sha256"] for name, entry in audit["sources"].items()},
        "split_seed": audit["split_seed"], "input_audit_sha256": protocol["input_audit_sha256"],
        "float32_cast_max_abs_difference": float(np.max(np.abs(values.astype(float) - scaled64))),
        "features": list(FEATURES), "mapping": mapping,
        "scaler_min": scaler.data_min_.tolist(), "scaler_max": scaler.data_max_.tolist(),
        "scaler_scale": scaler.scale_.tolist(), "scaler_offset": scaler.min_.tolist(),
        "partitions": {name: {"sample_ids": part["sample_ids"], "y_true": part["y"].tolist()}
                       for name, part in data.items()},
    }
    return data, metadata, scaler


def metrics(y_true, probability_m):
    y_true = np.asarray(y_true)
    probability_m = np.asarray(probability_m, dtype=np.float64)
    if y_true.ndim != 1 or probability_m.shape != y_true.shape or set(y_true) != {0, 1}:
        raise ValueError("Expected aligned vectors containing both binary classes")
    if not np.isfinite(probability_m).all() or ((probability_m < 0) | (probability_m > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    predicted = (probability_m > 0.5).astype(int)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability_m)),
        "log_loss": float(log_loss(y_true, np.column_stack([1 - probability_m, probability_m]), labels=[0, 1])),
        "confusion_matrix": matrix.tolist(), "correct": int(matrix.trace()), "total": len(y_true),
    }


class SmallGridCNN(nn.Module):
    """Preserve the tiny spatial grid until global pooling; output two logits."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(16, 2)

    def forward(self, inputs):
        activations = torch.relu(self.conv1(inputs))
        activations = torch.relu(self.conv2(activations))
        return self.classifier(self.pool(activations).flatten(1))


def seed_cpu(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def cnn_probabilities(model, images):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(images)).softmax(dim=1)[:, 1].numpy().astype(float)


def train_cnn(train_images, train_y, validation_images, validation_y, config, seed):
    """The function deliberately has no test-data argument."""
    if config["max_epochs"] < 1 or config["patience"] < 1 or config["batch_size"] < 1:
        raise ValueError("Training limits must be positive")
    seed_cpu(seed)
    model = SmallGridCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    x_train, y_train = torch.from_numpy(train_images), torch.from_numpy(train_y)
    x_validation, y_validation = torch.from_numpy(validation_images), torch.from_numpy(validation_y)
    shuffle = torch.Generator().manual_seed(seed)
    best_loss, best_epoch, best_state = float("inf"), 0, None
    history = []
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        order = torch.randperm(len(y_train), generator=shuffle)
        for batch in order.split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_train[batch]), y_train[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_logits, validation_logits = model(x_train), model(x_validation)
            train_loss = float(loss_fn(train_logits, y_train))
            validation_loss = float(loss_fn(validation_logits, y_validation))
            if not np.isfinite([train_loss, validation_loss]).all():
                raise ValueError("Nonfinite training/validation loss")
            history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= config["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        restored_loss = float(loss_fn(model(x_validation), y_validation))
    if restored_loss != best_loss:
        raise AssertionError("Restored checkpoint differs from the selected validation epoch")
    return model, {"history": history, "best_epoch": best_epoch, "epochs_run": len(history),
                   "selected_validation_loss": best_loss,
                   "restored_state_sha256": state_digest(model),
                   "parameters": sum(parameter.numel() for parameter in model.parameters())}


def summarize(runs):
    summary = {}
    for name in sorted({run["model"] for run in runs}):
        chosen = [run for run in runs if run["model"] == name]
        summary[name] = {"n_runs": len(chosen), "test": {}}
        for metric in METRICS:
            scores = [run["metrics"]["test"][metric] for run in chosen]
            summary[name]["test"][metric] = {
                "mean": float(np.mean(scores)),
                "sample_sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else None,
                "minimum": min(scores), "maximum": max(scores),
            }
    return summary


def run_experiment(protocol, artifact_dir):
    data, metadata, scaler = load_inputs(protocol)
    train, validation = data["train"], data["validation"]
    fitted = []
    for name, model in (("majority_prior", DummyClassifier(**protocol["dummy"])),
                        ("logistic_regression", LogisticRegression(**protocol["logistic_regression"]))):
        model.fit(train["tabular"], train["y"])
        if name == "logistic_regression" and model.n_iter_.max() >= protocol[name]["max_iter"]:
            raise RuntimeError("Logistic regression reached iteration limit; do not silently report convergence")
        fitted.append((name, None, model, {}))
    for seed in protocol["model_seeds"]:
        forest = RandomForestClassifier(random_state=seed, **protocol["random_forest"])
        forest.fit(train["tabular"], train["y"])
        fitted.append(("random_forest", seed, forest, {}))
        cnn, diagnostics = train_cnn(train["images"], train["y"], validation["images"],
                                     validation["y"], protocol["cnn"], seed)
        fitted.append(("grid_cnn", seed, cnn, diagnostics))
        print(f"Finished seed {seed}: CNN selected epoch {diagnostics['best_epoch']} of {diagnostics['epochs_run']}", flush=True)
    # All models/checkpoints are now fixed. Only this phase scores the learning test set.
    artifact_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for name, seed, model, diagnostics in fitted:
        run = {"model": name, "seed": seed, "diagnostics": diagnostics, "metrics": {}, "predictions": {}}
        for partition, part in data.items():
            if name == "grid_cnn":
                probabilities = cnn_probabilities(model, part["images"])
            else:
                np.testing.assert_array_equal(model.classes_, [0, 1])
                probabilities = model.predict_proba(part["tabular"])[:, 1].astype(float)
            run["metrics"][partition] = metrics(part["y"], probabilities)
            if partition != "train":
                run["predictions"][partition] = {"probability_m": probabilities.tolist(),
                                                  "predicted_label": (probabilities > 0.5).astype(int).tolist()}
        if name == "grid_cnn":
            filename = f"grid_cnn_{seed}.pt"
            torch.save({"state_dict": model.state_dict(), "seed": seed, "metadata": metadata,
                        "diagnostics": diagnostics, "protocol": protocol}, artifact_dir / filename)
        else:
            filename = f"{name}_{seed if seed is not None else 'fixed'}.joblib"
            joblib.dump({"model": model, "scaler": scaler, "metadata": metadata, "protocol": protocol},
                        artifact_dir / filename, compress=3)
        run["local_checkpoint_filename"] = filename
        runs.append(run)
    return {"experiment": protocol["experiment"], "date": "2026-10-23", "protocol": protocol,
            "protocol_sha256": sha256(PROTOCOL_PATH), "source_code_sha256": sha256(__file__),
            "environment": {"python": platform.python_version(), "system": platform.system(),
                            "machine": platform.machine(), **{p: version(p) for p in
                            ("torch", "numpy", "scipy", "pandas", "scikit-learn", "matplotlib")}},
            "data": metadata, "runs": runs, "summary": summarize(runs)}


def make_figures(report, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cnn_runs = [run for run in report["runs"] if run["model"] == "grid_cnn"]
    fig, axes = plt.subplots(1, len(cnn_runs), figsize=(13, 4), layout="constrained", squeeze=False)
    for axis, run in zip(axes[0], cnn_runs, strict=True):
        history = run["diagnostics"]["history"]
        epochs = [entry["epoch"] for entry in history]
        axis.plot(epochs, [entry["train_loss"] for entry in history], label="Training", color="#176b91")
        axis.plot(epochs, [entry["validation_loss"] for entry in history], label="Validation", color="#c35a19")
        axis.axvline(run["diagnostics"]["best_epoch"], linestyle="--", color="#555555", label="Selected epoch")
        axis.set(title=f"Seed {run['seed']}", xlabel="Epoch", ylabel="Cross-entropy (lower is better)")
        axis.legend(fontsize=8)
    fig.suptitle("CNN training diagnostics — no test feedback during fitting")
    fig.savefig(output_dir / "learning_curves.png", dpi=150)
    plt.close(fig)
    # Every run is shown, rather than selecting the most favorable seed.
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), layout="constrained")
    names = {"majority_prior": "Majority/prior", "logistic_regression": "Logistic regression",
             "random_forest": "Random forest", "grid_cnn": "Fixed-grid CNN"}
    for axis, run in zip(axes.flat, report["runs"], strict=True):
        matrix = np.asarray(run["metrics"]["test"]["confusion_matrix"])
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=54)
        for row in range(2):
            for col in range(2):
                axis.text(col, row, str(matrix[row, col]), ha="center", va="center",
                          color="white" if matrix[row, col] > 30 else "black", fontsize=13)
        suffix = f"\nSeed {run['seed']}" if run["seed"] is not None else "\nFixed run"
        axis.set(title=names[run["model"]] + suffix, xticks=[0, 1], yticks=[0, 1],
                 xticklabels=["B", "M"], yticklabels=["B", "M"], xlabel="Predicted", ylabel="Actual")
    fig.suptitle("Learning-test errors: the same 86 samples in every panel")
    fig.savefig(output_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/02_wisconsin_models")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/02_wisconsin_models")
    args = parser.parse_args()
    if (args.output_dir / "results.json").exists():
        parser.error("Results already exist; use a separate --output-dir for a reproducibility run")
    protocol = json.loads(PROTOCOL_PATH.read_text())
    report = run_experiment(protocol, args.artifact_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    make_figures(report, args.output_dir)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
