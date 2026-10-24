"""Experiment 04: paired synthetic shape/background controls, not thesis evidence."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from scripts.wisconsin_explanations import integrated_gradients
from scripts.wisconsin_learning import ROOT
from scripts.wisconsin_models import cnn_probabilities, metrics, sha256, train_cnn

PROTOCOL = ROOT / "experiments/04_shortcut_sandbox/protocol.json"


def generate_partition(count, seed):
    """Balance both shapes and independent backgrounds; share foreground across views."""
    if count < 4 or count % 4:
        raise ValueError("Count must be a positive multiple of four")
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(2, dtype=np.int64), count // 2)
    rng.shuffle(labels)
    masks = np.zeros((count, 1, 12, 12), dtype=bool)
    foreground = np.zeros(masks.shape, dtype=np.float32)
    for index, label in enumerate(labels):
        row, column = rng.integers(-1, 2, size=2)
        height, width = (2, 6) if label == 0 else (6, 2)
        top, left = (12 - height) // 2 + row, (12 - width) // 2 + column
        masks[index, 0, top:top + height, left:left + width] = True
        foreground[index][masks[index]] = rng.uniform(0.87, 0.93, height * width)
    independent = np.empty(count, dtype=np.int64)
    for label in (0, 1):
        colors = np.tile([0, 1], count // 4)
        rng.shuffle(colors)
        independent[labels == label] = colors
    assignments = {"correlated": labels.copy(), "independent": independent, "reversed": 1 - labels}
    images, backgrounds = {}, {}
    for name, colors in assignments.items():
        levels = np.where(colors == 0, 0.15, 0.65).astype(np.float32)
        backgrounds[name] = np.broadcast_to(levels[:, None, None, None], masks.shape).copy()
        images[name] = np.where(masks, foreground, backgrounds[name])
    return {"labels": labels, "masks": masks, "foreground": foreground,
            "images": images, "backgrounds": backgrounds, "background_labels": assignments}


def array_digest(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str((array.shape, str(array.dtype))).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def paired_diagnostic(labels, correlated, reversed_probabilities):
    original = np.where(labels == 1, correlated, 1 - correlated)
    changed = np.where(labels == 1, reversed_probabilities, 1 - reversed_probabilities)
    return {"prediction_flip_rate": float(np.mean((correlated > 0.5) != (reversed_probabilities > 0.5))),
            "mean_true_class_probability_drop": float(np.mean(original - changed))}


def attribution_mass(attribution, mask):
    magnitude = np.abs(attribution)
    foreground, background = float(magnitude[mask].sum()), float(magnitude[~mask].sum())
    total = foreground + background
    return {"foreground_pixels": int(mask.sum()), "background_pixels": int((~mask).sum()),
            "foreground_absolute_mass": foreground, "background_absolute_mass": background,
            "foreground_fraction": foreground / total if total else None,
            "foreground_mean_absolute_per_pixel": foreground / int(mask.sum()),
            "background_mean_absolute_per_pixel": background / int((~mask).sum())}


def explain_examples(model, test):
    cases = []
    for index in range(8):
        image = test["images"]["correlated"][index:index + 1]
        label = int(test["labels"][index])
        references = {"background_only": test["backgrounds"]["correlated"][index:index + 1],
                      "black": np.zeros_like(image)}
        case = {"test_position": index, "true_label": label, "image": image[0, 0].tolist(),
                "foreground_mask": test["masks"][index, 0].tolist(), "references": {}}
        for name, baseline in references.items():
            values, diagnostic = integrated_gradients(model, image, baseline, sign=1.0 if label else -1.0)
            probability = cnn_probabilities(model, baseline)[0]
            case["references"][name] = {"attribution": values[0, 0].tolist(),
                                       "mass": attribution_mass(values, test["masks"][index:index + 1]),
                                       "numerical_check": diagnostic,
                                       "reference_true_class_probability": float(probability if label else 1 - probability)}
        cases.append(case)
    return cases


def summarize(runs):
    summary = {}
    for regime in ("correlated", "independent"):
        selected = [run for run in runs if run["training_regime"] == regime]
        summary[regime] = {"n_seeds": len(selected), "test_accuracy": {}, "test_macro_f1": {}}
        for view in ("correlated", "independent", "reversed"):
            for metric in ("accuracy", "macro_f1"):
                scores = [run["test_metrics"][view][metric] for run in selected]
                summary[regime][f"test_{metric}"][view] = {
                    "mean": float(np.mean(scores)), "sample_sd": float(np.std(scores, ddof=1)),
                    "minimum": min(scores), "maximum": max(scores)}
        for key in ("prediction_flip_rate", "mean_true_class_probability_drop"):
            summary[regime][key] = float(np.mean([r["paired_background_change"][key] for r in selected]))
    return summary


def run_experiment(config, artifact_dir):
    partitions = {name: generate_partition(config[f"{name}_rows"], config["data_seed"] + offset)
                  for offset, name in enumerate(("training", "validation", "test"))}
    train, validation, test = (partitions[name] for name in ("training", "validation", "test"))
    fitted = []
    for regime in ("correlated", "independent"):
        for seed in config["model_seeds"]:
            model, diagnostics = train_cnn(train["images"][regime], train["labels"],
                                           validation["images"][regime], validation["labels"], config["cnn"], seed)
            fitted.append((regime, seed, model, diagnostics))
            print(f"Fitted {regime} background, seed {seed}, selected epoch {diagnostics['best_epoch']}", flush=True)
    # All six checkpoints are selected before any test scoring or attribution.
    artifact_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for regime, seed, model, diagnostics in fitted:
        probabilities = {view: cnn_probabilities(model, images) for view, images in test["images"].items()}
        filename = f"{regime}_{seed}.pt"
        torch.save({"state_dict": model.state_dict(), "protocol": config, "diagnostics": diagnostics,
                    "training_regime": regime, "seed": seed}, artifact_dir / filename)
        run = {"training_regime": regime, "seed": seed, "diagnostics": diagnostics,
               "local_checkpoint_filename": filename,
               "test_metrics": {view: metrics(test["labels"], values) for view, values in probabilities.items()},
               "probability_vertical": {view: values.tolist() for view, values in probabilities.items()},
               "paired_background_change": paired_diagnostic(test["labels"], probabilities["correlated"], probabilities["reversed"])}
        if seed == config["explanation"]["reference_seed"]:
            run["explanations"] = explain_examples(model, test)
        runs.append(run)
    return {"experiment": config["experiment"], "protocol": config, "protocol_sha256": sha256(PROTOCOL),
            "source_code_sha256": sha256(__file__),
            "attribution_helper_sha256": sha256(ROOT / "scripts/wisconsin_explanations.py"),
            "training_helper_sha256": sha256(ROOT / "scripts/wisconsin_models.py"),
            "environment": {name: version(name) for name in ("torch", "captum", "numpy", "scikit-learn")},
            "partition_seeds": {name: config["data_seed"] + offset for offset, name in enumerate(partitions)},
            "data_hashes": {name: array_digest(part["labels"], part["masks"], *part["images"].values()) for name, part in partitions.items()},
            "test_labels": test["labels"].tolist(), "runs": runs, "summary": summarize(runs)}


def make_figures(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    test = generate_partition(report["protocol"]["test_rows"], report["partition_seeds"]["test"])
    fig, axes = plt.subplots(2, 3, figsize=(9, 6), layout="constrained")
    for row, label in enumerate((0, 1)):
        index = int(np.flatnonzero(test["labels"] == label)[0])
        for col, view in enumerate(("correlated", "independent", "reversed")):
            axes[row, col].imshow(test["images"][view][index, 0], cmap="gray", vmin=0, vmax=1)
            axes[row, col].set(title=f"{'Horizontal' if label == 0 else 'Vertical'} · {view}", xticks=[], yticks=[])
    fig.suptitle("Paired test inputs: same shape, location and foreground pixels; backgrounds vary")
    fig.savefig(output / "paired_inputs.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for regime, color in (("correlated", "#ad681d"), ("independent", "#176b91")):
        selected = [r for r in report["runs"] if r["training_regime"] == regime]
        for run in selected:
            axis.plot(range(3), [100 * run["test_metrics"][view]["accuracy"] for view in ("correlated", "independent", "reversed")],
                      color=color, alpha=0.35, marker=".")
        means = [100 * report["summary"][regime]["test_accuracy"][view]["mean"] for view in ("correlated", "independent", "reversed")]
        axis.plot(range(3), means, color=color, linewidth=2.5, marker="o", label=f"{regime.capitalize()} background training")
    axis.set(xticks=range(3), xticklabels=["Correlated", "Independent", "Reversed"], ylim=(-3, 103),
             ylabel="Test accuracy (%)", xlabel="Paired background condition", title="Three fixed seeds per training regime; lines include each seed")
    axis.legend()
    fig.savefig(output / "background_sensitivity.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(10, 7), layout="constrained")
    for row, regime in enumerate(("correlated", "independent")):
        run = next(r for r in report["runs"] if r["training_regime"] == regime and "explanations" in r)
        case = run["explanations"][0]  # fixed first record, not chosen by its explanation
        axes[row, 0].imshow(case["image"], cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"{regime.capitalize()} training\nSame first test image")
        for col, reference in enumerate(("background_only", "black"), start=1):
            explanation = case["references"][reference]
            values = np.array(explanation["attribution"])
            maximum = max(float(np.abs(values).max()), 1e-12)
            axes[row, col].imshow(values / maximum, cmap="RdBu_r", vmin=-1, vmax=1)
            fraction = explanation["mass"]["foreground_fraction"]
            axes[row, col].set_title(f"IG · {reference.replace('_', ' ')} reference\nForeground |IG| fraction: {fraction:.1%}")
        for axis in axes[row]:
            axis.set(xticks=[], yticks=[])
    fig.suptitle("Reference sensitivity: background-only reference forces zero background IG\nEach map normalized separately; red supports / blue opposes true shape")
    fig.savefig(output / "reference_sensitivity.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROTOCOL.parent)
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/04_shortcut_sandbox")
    args = parser.parse_args()
    if (args.output_dir / "results.json").exists() or (args.artifact_dir.exists() and any(args.artifact_dir.iterdir())):
        parser.error("Use fresh output/artifact directories; existing runs will not be overwritten")
    report = run_experiment(json.loads(PROTOCOL.read_text()), args.artifact_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    make_figures(report, args.output_dir)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
