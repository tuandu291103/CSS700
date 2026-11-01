"""Fixed-protocol genomic learning comparison; all fitting precedes test scoring."""

import argparse
import copy
import json
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch import nn

from scripts.audit_tcga_brca import ROOT, sha256
from scripts.prepare_tcga_pilot import PRIVATE, PROTOCOL, PUBLIC, write_json
from scripts.wisconsin_models import (
    SmallGridCNN,
    cnn_probabilities,
    metrics,
    seed_cpu,
    state_digest,
)

OUTPUT = ROOT / "experiments/06_tcga_brca_pilot"
SCORES = ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "log_loss", "recall_normal", "recall_tumor")


def class_weights(labels):
    if set(labels) != {0, 1}:
        raise ValueError("Training requires both binary classes")
    return (len(labels) / (2 * np.bincount(labels, minlength=2))).astype(np.float32)


def prediction_metrics(labels, probabilities):
    result = metrics(labels, probabilities)
    matrix = np.asarray(result["confusion_matrix"])
    result["recall_normal"] = float(matrix[0, 0] / matrix[0].sum())
    result["recall_tumor"] = float(matrix[1, 1] / matrix[1].sum())
    return result


def fit_cnn(train_x, train_y, validation_x, validation_y, config, seed):
    """Explicit development-only signature; no test arrays or test checkpoint selection."""
    seed_cpu(seed)
    model = SmallGridCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    weights = class_weights(train_y)
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights))
    x, y = torch.from_numpy(train_x), torch.from_numpy(train_y)
    vx, vy = torch.from_numpy(validation_x), torch.from_numpy(validation_y)
    shuffle = torch.Generator().manual_seed(seed)
    best_loss, best_epoch, best_state = float("inf"), 0, None
    history = []
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        for batch in torch.randperm(len(y), generator=shuffle).split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x[batch]), y[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss = float(loss_fn(model(x), y))
            validation_loss = float(loss_fn(model(vx), vy))
        if not np.isfinite([train_loss, validation_loss]).all():
            raise ValueError("Nonfinite development loss")
        history.append({"epoch": epoch, "train_weighted_loss": train_loss,
                        "validation_weighted_loss": validation_loss})
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= config["patience"]:
            break
    if best_state is None:
        raise ValueError("No valid checkpoint selected")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        restored = float(loss_fn(model(vx), vy))
    if restored != best_loss:
        raise AssertionError("Validation checkpoint restoration mismatch")
    return model, {"history": history, "best_epoch": best_epoch, "epochs_run": len(history),
                   "selected_validation_weighted_loss": best_loss, "class_weights": weights.tolist(),
                   "state_sha256": state_digest(model),
                   "parameters": sum(p.numel() for p in model.parameters())}


def load_prepared(data_dir, preparation_path):
    protocol = json.loads(PROTOCOL.read_text())
    preparation = json.loads(preparation_path.read_text())
    if preparation["protocol_sha256"] != sha256(PROTOCOL):
        raise ValueError("Protocol changed since preprocessing")
    if preparation["preparation_code_sha256"] != sha256(ROOT / "scripts/prepare_tcga_pilot.py"):
        raise ValueError("Preprocessing code changed")
    if preparation["private_partitions_sha256"] != sha256(data_dir / "partitions.json"):
        raise ValueError("Frozen patient partitions changed")
    data = {}
    for scenario, description in preparation["scenarios"].items():
        path = data_dir / f"{scenario}.npz"
        if sha256(path) != description["processed_arrays_sha256"]:
            raise ValueError("Processed arrays changed")
        with np.load(path, allow_pickle=False) as stored:
            data[scenario] = {name: stored[name].copy() for name in stored.files}
    return protocol, preparation, data


def summarize(runs):
    summary = {}
    for name in sorted({r["model"] for r in runs}):
        chosen = [r for r in runs if r["model"] == name]
        summary[name] = {"n_runs": len(chosen), "test": {}}
        for metric in SCORES:
            values = [r["metrics"]["test"][metric] for r in chosen]
            summary[name]["test"][metric] = {"mean": float(np.mean(values)),
                "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}
    return summary


def run(data_dir, preparation_path, output_dir, artifact_dir):
    if (output_dir / "results.json").exists() or artifact_dir.exists():
        raise FileExistsError("Refusing to replace results or checkpoints; use new directories for reruns")
    protocol, _preparation, data = load_prepared(data_dir, preparation_path)
    artifact_dir.mkdir(parents=True)
    fitted = []
    for scenario, arrays in data.items():
        x, y = arrays["x"], arrays["y"]
        images = x.reshape(len(x), 1, *protocol["preprocessing"]["image_shape"])
        train, val = arrays["train"], arrays["validation"]
        for name, classifier in (
            ("majority_prior", DummyClassifier(**protocol["dummy"])),
            ("logistic_regression", LogisticRegression(**protocol["logistic_regression"])),
        ):
            classifier.fit(x[train], y[train])
            diagnostics = {}
            if name == "logistic_regression":
                if classifier.n_iter_.max() >= protocol[name]["max_iter"]:
                    raise RuntimeError("Logistic regression reached iteration limit")
                diagnostics = {"iterations": classifier.n_iter_.tolist()}
            fitted.append((scenario, name, None, classifier, diagnostics))
        for seed in protocol["model_seeds"]:
            forest = RandomForestClassifier(random_state=seed, **protocol["random_forest"])
            forest.fit(x[train], y[train])
            fitted.append((scenario, "random_forest", seed, forest, {}))
            cnn, diagnostics = fit_cnn(images[train], y[train], images[val], y[val], protocol["cnn"], seed)
            fitted.append((scenario, "grid_cnn", seed, cnn, diagnostics))
            print(f"{scenario}, seed {seed}: selected epoch {diagnostics['best_epoch']}/{diagnostics['epochs_run']}; test not scored", flush=True)
    print("All models in BOTH scenarios fixed; beginning one-time test scoring", flush=True)
    report = {"experiment": protocol["experiment"], "date": protocol["recorded_date"],
              "protocol_sha256": sha256(PROTOCOL), "preparation_sha256": sha256(preparation_path),
              "model_code_sha256": sha256(Path(__file__)),
              "reused_wisconsin_model_code_sha256": sha256(ROOT / "scripts/wisconsin_models.py"),
              "environment": {"python": platform.python_version(), "machine": platform.machine(),
                  **{package: version(package) for package in ("numpy", "torch", "scikit-learn", "joblib", "matplotlib")}},
              "scenarios": {name: {"runs": []} for name in data}}
    predictions = {"protocol_sha256": sha256(PROTOCOL), "scenarios": {name: [] for name in data}}
    for scenario, name, seed, model, diagnostics in fitted:
        arrays = data[scenario]
        run_result = {"model": name, "seed": seed, "diagnostics": diagnostics, "metrics": {}}
        private_result = {"model": name, "seed": seed, "partitions": {}}
        for part in ("train", "validation", "test"):
            ix = arrays[part]
            x, y = arrays["x"][ix], arrays["y"][ix]
            if name == "grid_cnn":
                probability = cnn_probabilities(model, x.reshape(len(x), 1, *protocol["preprocessing"]["image_shape"]))
            else:
                np.testing.assert_array_equal(model.classes_, [0, 1])
                probability = model.predict_proba(x)[:, 1].astype(float)
            run_result["metrics"][part] = prediction_metrics(y, probability)
            private_result["partitions"][part] = {"row_indices": ix.tolist(), "y_true": y.tolist(),
                                                   "probability_tumor": probability.tolist()}
        basename = f"{scenario}_{name}_{seed if seed is not None else 'fixed'}"
        if name == "grid_cnn":
            torch.save({"state_dict": model.state_dict(), "protocol_sha256": sha256(PROTOCOL),
                        "preparation_sha256": sha256(preparation_path)}, artifact_dir / f"{basename}.pt")
            # Verify serialized model against every saved held-out prediction.
            restored = SmallGridCNN()
            restored.load_state_dict(torch.load(artifact_dir / f"{basename}.pt", weights_only=True)["state_dict"])
            for part in ("validation", "test"):
                ix = arrays[part]
                reloaded = cnn_probabilities(restored, arrays["x"][ix].reshape(len(ix), 1, 16, 16))
                np.testing.assert_array_equal(reloaded, private_result["partitions"][part]["probability_tumor"])
        else:
            joblib.dump(model, artifact_dir / f"{basename}.joblib", compress=3)
            restored = joblib.load(artifact_dir / f"{basename}.joblib")
            for part in ("validation", "test"):
                np.testing.assert_array_equal(restored.predict_proba(arrays["x"][arrays[part]])[:, 1],
                    private_result["partitions"][part]["probability_tumor"])
        report["scenarios"][scenario]["runs"].append(run_result)
        predictions["scenarios"][scenario].append(private_result)
    for description in report["scenarios"].values():
        description["summary"] = summarize(description["runs"])
    write_json(artifact_dir / "predictions.json", predictions)
    report["private_predictions_sha256"] = sha256(artifact_dir / "predictions.json")
    write_json(output_dir / "results.json", report)
    make_figure(report, output_dir)
    print(json.dumps({name: result["summary"] for name, result in report["scenarios"].items()}, indent=2))


def make_figure(report, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for row, (scenario, description) in enumerate(report["scenarios"].items()):
        runs = [r for r in description["runs"] if r["model"] == "grid_cnn"]
        for axis, run_result in zip(axes[row], runs, strict=True):
            history = run_result["diagnostics"]["history"]
            epochs = [h["epoch"] for h in history]
            axis.plot(epochs, [h["train_weighted_loss"] for h in history], label="Training")
            axis.plot(epochs, [h["validation_weighted_loss"] for h in history], label="Validation")
            axis.axvline(run_result["diagnostics"]["best_epoch"], color="#555555", linestyle="--", label="Selected")
            axis.set(title=f"{scenario.replace('_', ' ')} | seed {run_result['seed']}",
                     xlabel="Epoch", ylabel="Weighted cross-entropy")
            axis.legend(fontsize=8)
    fig.suptitle("TCGA-BRCA learning pilot: validation-only checkpoint selection")
    fig.savefig(output_dir / "learning_curves.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PRIVATE)
    parser.add_argument("--preparation", type=Path, default=PUBLIC)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/tcga_brca_pilot")
    args = parser.parse_args()
    run(args.data_dir, args.preparation, args.output_dir, args.artifact_dir)
