"""Experiment 07: inspect pinned author code; isolate layout from value encoding.

The upstream checkout is a local, ignored dependency, not vendored thesis code.
This script changes neither it nor previous experiments. Generated evidence is
written only into a fresh result location. No learning-test inputs are evaluated.
"""

import argparse
import itertools
import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from scripts.wisconsin_learning import (
    BASE_FEATURES,
    FEATURES,
    ROOT,
    load_data,
    verify_sources,
)
from scripts.wisconsin_models import cnn_probabilities, metrics, sha256, train_cnn

PROTOCOL_PATH = ROOT / "experiments/07_som2img_layout/protocol.json"
DEFAULT_UPSTREAM = ROOT / ".cache/tab2img-bench"
METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "log_loss")


def import_upstream(path, commit):
    """Require the pinned clean Git checkout before importing executable code."""
    path = Path(path).resolve()
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual != commit:
        raise ValueError(f"Upstream commit mismatch: {actual}")
    dirty = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=normal"], text=True,
    ).strip()
    if dirty:
        raise ValueError("Upstream checkout must have no tracked changes or untracked source files")
    sys.path.insert(0, str(path / "src"))
    import tab2img
    if Path(tab2img.__file__).resolve() != path / "src/tab2img/__init__.py":
        raise ValueError("Another tab2img package was imported")
    files = [path / p for p in ("README.md", "LICENSE", "requirements.txt", "pyproject.toml")]
    files += sorted((path / "src").rglob("*.py"))
    return tab2img, {str(p.relative_to(path)): sha256(p) for p in files}


def prepare_data(protocol):
    audit_path = ROOT / protocol["input_audit"]
    if sha256(audit_path) != protocol["input_audit_sha256"]:
        raise ValueError("Original data audit has changed")
    audit = json.loads(audit_path.read_text())
    sources = verify_sources(ROOT / "data/raw/wdbc")
    frame = load_data(ROOT / "data/raw/wdbc/wdbc.data")
    train = np.array(audit["splits"]["train"]["row_indices_zero_based"])
    evaluation = np.array(audit["splits"]["validation"]["row_indices_zero_based"])
    excluded = set(audit["splits"]["test"]["row_indices_zero_based"])
    labels = frame.diagnosis.map({"B": 0, "M": 1}).to_numpy(dtype=np.int64)
    fit, stop = train_test_split(train, test_size=protocol["inner_stop_size"],
                                 stratify=labels[train], random_state=protocol["inner_split_seed"])
    indices = {"fit": np.sort(fit), "inner_stop": np.sort(stop), "evaluation": evaluation}
    used = np.concatenate(list(indices.values()))
    if len(set(used)) != len(used) or excluded.intersection(used):
        raise AssertionError("Partition overlap or forbidden learning-test use")
    raw = frame.loc[:, FEATURES].to_numpy(dtype=np.float64)
    scaler = MinMaxScaler(clip=False).fit(raw[indices["fit"]])
    data, manifest = {}, {}
    for name, rows in indices.items():
        data[name] = {"raw": raw[rows], "scaled": scaler.transform(raw[rows]), "y": labels[rows]}
        manifest[name] = {"n": len(rows), "row_indices_zero_based": rows.tolist(),
                          "sample_ids": frame.sample_id.iloc[rows].tolist(),
                          "labels": labels[rows].tolist(),
                          "class_counts": {str(k): int(v) for k, v in
                                           zip(*np.unique(labels[rows], return_counts=True), strict=True)}}
    return data, {"partitions": manifest, "sources": sources, "features": list(FEATURES),
                  "scaler_min": scaler.data_min_.tolist(), "scaler_max": scaler.data_max_.tolist(),
                  "excluded_learning_test_count": len(excluded)}, scaler


def positions(feature_map, n_features):
    feature_map = np.asarray(feature_map)
    placed = feature_map[feature_map >= 0]
    if placed.size != n_features or not np.array_equal(np.sort(placed), np.arange(n_features)):
        raise ValueError("Expected every feature exactly once; no collisions or omissions")
    return np.array([np.argwhere(feature_map == i)[0] for i in range(n_features)])


def decode(images, feature_map, n_features):
    pos = positions(feature_map, n_features)
    return np.asarray(images)[:, pos[:, 0], pos[:, 1]]


def edges(feature_map):
    pairs = []
    for (r, c), feature in np.ndenumerate(feature_map):
        if feature < 0:
            continue
        for rr, cc in ((r + 1, c), (r, c + 1)):
            if rr < feature_map.shape[0] and cc < feature_map.shape[1] and feature_map[rr, cc] >= 0:
                pairs.append(tuple(sorted((int(feature), int(feature_map[rr, cc])))))
    return set(pairs)


def layout_statistics(feature_map, scaled_fit):
    n_features = scaled_fit.shape[1]
    pos = positions(feature_map, n_features)
    adjacent = sorted(edges(feature_map))
    a, b = np.array(adjacent).T
    rmses = np.sqrt(np.mean((scaled_fit[:, a] - scaled_fit[:, b]) ** 2, axis=0))
    correlation = np.corrcoef(scaled_fit, rowvar=False)
    if not np.isfinite(correlation).all():
        raise ValueError("Constant features require a separately specified correlation policy")
    # The ten measurement names have three summaries each in the source dictionary.
    groups = [i % len(BASE_FEATURES) for i in range(n_features)]
    same_group = [(i, j) for i, j in itertools.combinations(range(n_features), 2)
                  if groups[i] == groups[j]]
    distance = [int(np.abs(pos[i] - pos[j]).sum()) for i, j in same_group]
    return {"n_features": n_features, "n_neighbor_pairs": len(adjacent),
            "adjacent_profile_rmse": float(rmses.mean()),
            "adjacent_absolute_pearson": float(np.abs(correlation[a, b]).mean()),
            "same_measurement_mean_manhattan": float(np.mean(distance)),
            "same_measurement_pair_count": len(same_group)}


def encoding_audit(tab2img, values, sample_ids):
    # A row-major arrangement permits a direct feature-by-feature numeric audit.
    images = values.reshape(-1, 5, 6)
    def encoded(quantize, clip):
        return tab2img.encode_images(images, quantize=quantize, clip=clip,
                                     channels=1)[..., 0].reshape(values.shape)
    default, clipped, floating = encoded(True, False), encoded(True, True), encoded(False, False)
    outside = (values < 0) | (values > 1)
    materially_different = np.abs(default.astype(float) - clipped.astype(float)) > 0.5
    changed = []
    for row, col in np.argwhere(outside):
        changed.append({"sample_id": sample_ids[row], "feature": FEATURES[col],
                        "feature_index": int(col), "scaled_float64": float(values[row, col]),
                        "default_uint8": float(default[row, col]),
                        "clipped_uint8": float(clipped[row, col]),
                        "float32": float(floating[row, col])})
    toy = np.array([[[1.02]]])
    return {"total_values": int(values.size), "outside_training_range": int(outside.sum()),
            "samples_outside_training_range": int(outside.any(axis=1).sum()),
            "default_vs_clipped_difference_over_half_range": int(materially_different.sum()),
            "samples_with_difference_over_half_range": int(materially_different.any(axis=1).sum()),
            "default_max_absolute_error": float(np.max(np.abs(values - default))),
            "clipped_max_absolute_error": float(np.max(np.abs(values - clipped))),
            "float32_max_absolute_error": float(np.max(np.abs(values - floating))),
            "all_out_of_range_values": changed,
            "synthetic_1_02_default": float(tab2img.encode_images(toy)[0, 0, 0, 0]),
            "synthetic_1_02_clipped": float(tab2img.encode_images(toy, clip=True)[0, 0, 0, 0]),
            "note": "Numerical input audit only; not an estimate of classifier performance impact."}


def run(protocol, upstream, checkpoint_dir):
    tab2img, hashes = import_upstream(upstream, protocol["upstream_commit"])
    data, manifest, scaler = prepare_data(protocol)
    shape = tuple(protocol["shape"])
    layouts, image_batches = [], {}
    specs = [("Fixed", None)] + [(m, s) for m in ("SnakeRandom", "SOM2IMG")
                                for s in protocol["layout_seeds"]]
    for method, seed in specs:
        key = f"{method}_{seed}" if seed is not None else method
        if method == "Fixed":
            fm = np.arange(len(FEATURES)).reshape(shape)
            batches = {p: v["scaled"].reshape(-1, *shape) for p, v in data.items()}
            fitted = None
        else:
            settings = protocol["som"] if method == "SOM2IMG" else {}
            fitted = tab2img.create(method, image_shape=shape, random_state=seed, **settings)
            fitted.fit(data["fit"]["raw"])
            np.testing.assert_array_equal(fitted.value_scaler_.data_min_, scaler.data_min_)
            np.testing.assert_array_equal(fitted.value_scaler_.data_max_, scaler.data_max_)
            fm = fitted.feature_map_
            batches = {p: fitted.transform(v["raw"]) for p, v in data.items()}
        pos = positions(fm, len(FEATURES))
        record = {"key": key, "method": method, "layout_seed": seed,
                  "feature_map": fm.tolist(),
                  "mapping": [{"feature": f, "feature_index": i, "row": int(pos[i, 0]),
                               "column": int(pos[i, 1])} for i, f in enumerate(FEATURES)],
                  "statistics": layout_statistics(fm, data["fit"]["scaled"]),
                  "round_trip_max_abs": {}}
        for partition, batch in batches.items():
            recovered = decode(batch, fm, len(FEATURES))
            np.testing.assert_array_equal(recovered, data[partition]["scaled"])
            record["round_trip_max_abs"][partition] = float(np.max(np.abs(recovered - data[partition]["scaled"])))
        if method == "SOM2IMG":
            bmus = np.array([fitted.som_.winner(v) for v in data["fit"]["scaled"].T])
            distances = np.abs(pos - bmus).sum(axis=1)
            record["bmu_audit"] = {"unique_bmus_before_spillover": len(np.unique(bmus, axis=0)),
                                   "features_moved_from_bmu": int((distances > 0).sum()),
                                   "maximum_spillover_manhattan": int(distances.max()),
                                   "bmus_in_feature_order": bmus.tolist()}
        layouts.append(record)
        image_batches[key] = {p: np.ascontiguousarray(b[:, None].astype(np.float32))
                              for p, b in batches.items()}
        print(f"Prepared {key}: {record['statistics']}", flush=True)
    random_generator = np.random.default_rng(protocol["layout_null"]["seed"])
    null = [layout_statistics(random_generator.permutation(len(FEATURES)).reshape(shape),
                              data["fit"]["scaled"])
            for _ in range(protocol["layout_null"]["n_permutations"])]
    numeric_metrics = ("adjacent_profile_rmse", "adjacent_absolute_pearson", "same_measurement_mean_manhattan")
    null_summary = {name: {"mean": float(np.mean([r[name] for r in null])),
                           "q025": float(np.quantile([r[name] for r in null], 0.025)),
                           "q975": float(np.quantile([r[name] for r in null], 0.975))}
                    for name in numeric_metrics}
    stability = []
    for method in ("SnakeRandom", "SOM2IMG"):
        chosen = [r for r in layouts if r["method"] == method]
        for left, right in itertools.combinations(chosen, 2):
            a, b = edges(np.array(left["feature_map"])), edges(np.array(right["feature_map"]))
            stability.append({"method": method, "left": left["key"], "right": right["key"],
                              "neighbor_jaccard": len(a & b) / len(a | b)})
    audit = encoding_audit(tab2img, data["evaluation"]["scaled"], manifest["partitions"]["evaluation"]["sample_ids"])
    fitted_models = []
    for layout in layouts:
        batches = image_batches[layout["key"]]
        for seed in protocol["cnn_seeds"]:
            model, diagnostics = train_cnn(batches["fit"], data["fit"]["y"],
                                          batches["inner_stop"], data["inner_stop"]["y"], protocol["cnn"], seed)
            fitted_models.append((layout, seed, model, diagnostics))
            print(f"Fitted {layout['key']} CNN seed {seed}; inner-stop epoch {diagnostics['best_epoch']}", flush=True)
    # Only now evaluate all frozen models. No selection uses these evaluation scores.
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for layout, seed, model, diagnostics in fitted_models:
        batch = image_batches[layout["key"]]["evaluation"]
        probabilities = cnn_probabilities(model, batch)
        filename = f"{layout['key']}_cnn_{seed}.pt"
        path = checkpoint_dir / filename
        if path.exists():
            raise FileExistsError(path)
        torch.save(model.state_dict(), path)
        runs.append({"method": layout["method"], "layout_key": layout["key"],
                     "layout_seed": layout["layout_seed"], "cnn_seed": seed, "diagnostics": diagnostics,
                     "evaluation": metrics(data["evaluation"]["y"], probabilities),
                     "probability_m": probabilities.tolist(), "local_checkpoint_filename": filename})
    summaries = {}
    for method in protocol["methods"]:
        chosen = [r for r in runs if r["method"] == method]
        summaries[method] = {"n_models": len(chosen), "metrics": {}}
        for metric in METRICS:
            scores = [r["evaluation"][metric] for r in chosen]
            summaries[method]["metrics"][metric] = {"mean": float(np.mean(scores)),
                                                     "minimum": min(scores), "maximum": max(scores),
                                                     "descriptive_sample_sd": float(np.std(scores, ddof=1))}
    paired = []
    for layout_seed in protocol["layout_seeds"]:
        for cnn_seed in protocol["cnn_seeds"]:
            som = next(r for r in runs if r["method"] == "SOM2IMG" and r["layout_seed"] == layout_seed and r["cnn_seed"] == cnn_seed)
            control = next(r for r in runs if r["method"] == "SnakeRandom" and r["layout_seed"] == layout_seed and r["cnn_seed"] == cnn_seed)
            paired.append({"layout_seed": layout_seed, "cnn_seed": cnn_seed,
                           "som_minus_random": {m: som["evaluation"][m] - control["evaluation"][m] for m in METRICS}})
    return {"date": protocol["recorded_date"], "protocol": protocol,
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in
                              (Path(__file__), ROOT / "scripts/wisconsin_models.py", ROOT / "scripts/wisconsin_learning.py")},
            "upstream": {"url": protocol["upstream_url"], "commit": protocol["upstream_commit"], "file_sha256": hashes},
            "environment": {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(),
                            **{p: version(p) for p in ("numpy", "scipy", "pandas", "scikit-learn", "torch", "MiniSom", "matplotlib", "PyYAML")}},
            "data": manifest, "layouts": layouts, "random_layout_reference": null_summary,
            "random_reference_values": {m: [r[m] for r in null] for m in numeric_metrics},
            "layout_stability": stability, "encoding_audit": audit, "runs": runs,
            "summary": summaries, "paired_som_random": paired}


def make_figures(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    colors = plt.get_cmap("tab10")(np.arange(10))
    examples = [next(r for r in result["layouts"] if r["method"] == m) for m in result["protocol"]["methods"]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), layout="constrained")
    short_names = ("radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "conc. points", "symmetry", "fractal dim.")
    for ax, layout in zip(axes, examples, strict=True):
        fm = np.array(layout["feature_map"])
        ax.imshow(fm % 10, cmap=ListedColormap(colors), vmin=-0.5, vmax=9.5)
        for (r, c), index in np.ndenumerate(fm):
            summary = ("mean", "SE", "worst")[index // 10]
            ax.text(c, r, f"{index}: {short_names[index % 10]}\n{summary}", ha="center", va="center", fontsize=7,
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1})
        ax.set_title(layout["key"].replace("_", " · seed "), fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Same 30 features, different positions\nColor = measurement family, NOT input value or model importance", fontsize=15)
    fig.legend(handles=[Patch(facecolor=colors[i], label=short_names[i]) for i in range(10)],
               loc="outside lower center", ncol=5, frameon=False, fontsize=9)
    fig.savefig(output / "feature_layouts.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained")
    names = ("adjacent_profile_rmse", "adjacent_absolute_pearson", "same_measurement_mean_manhattan")
    titles = ("Neighbor profile difference ↓", "Neighbor |Pearson r| ↑", "Same-measurement distance ↓")
    palette = {"Fixed": "#586174", "SnakeRandom": "#d57c28", "SOM2IMG": "#247b78"}
    for ax, name, title in zip(axes, names, titles, strict=True):
        reference = result["random_layout_reference"][name]
        ax.axhspan(reference["q025"], reference["q975"], color="#cbd1d9", alpha=0.4)
        ax.axhline(reference["mean"], color="#6b7280", linestyle="--", linewidth=1)
        for i, method in enumerate(result["protocol"]["methods"]):
            scores = [r["statistics"][name] for r in result["layouts"] if r["method"] == method]
            offsets = np.linspace(-0.12, 0.12, len(scores)) if len(scores) > 1 else [0]
            ax.scatter(i + np.array(offsets), scores, color=palette[method], s=45, zorder=3)
        ax.set_xticks(range(3), ["Fixed", "Random", "SOM"])
        ax.set_xlim(-0.45, 2.45)
        ax.set_title(title, fontsize=11)
    fig.suptitle("Layout diagnostics on 318 fit rows — not biological validation\nPoints = layout seeds; gray band = central 95% of 1,000 random layouts", fontsize=13)
    fig.savefig(output / "layout_diagnostics.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), layout="constrained")
    for ax, metric, title in zip(axes, ("balanced_accuracy", "macro_f1", "log_loss"),
                                 ("Balanced accuracy ↑", "Macro F1 ↑", "Log loss ↓"), strict=True):
        for i, method in enumerate(result["protocol"]["methods"]):
            selected = [r for r in result["runs"] if r["method"] == method]
            scores = [r["evaluation"][metric] for r in selected]
            ax.scatter(i + np.linspace(-0.15, 0.15, len(scores)), scores, color=palette[method], s=34, alpha=0.85)
            ax.plot([i - 0.22, i + 0.22], [np.mean(scores)] * 2, color="black", linewidth=2)
        ax.set_xticks(range(3), ["Fixed\n3 models", "Random\n9 models", "SOM\n9 models"])
        ax.set_title(title, fontsize=12)
        if metric != "log_loss":
            ax.set_ylim(0.8, 1.005)
    fig.suptitle("Matched CNNs on 86 reused validation records\nDots = all runs; bars = means. Dependent exploratory repeats, not confidence intervals.", fontsize=13)
    fig.savefig(output / "prediction_comparison.png", dpi=160)
    plt.close(fig)

    audit = result["encoding_audit"]
    records = sorted(audit["all_out_of_range_values"], key=lambda r: abs(r["scaled_float64"] - r["default_uint8"]), reverse=True)[:4]
    fig, ax = plt.subplots(figsize=(10, 4.6), layout="constrained")
    for shift, field, label, color in ((-0.25, "scaled_float64", "Original scaled value", "#247b78"),
                                      (0, "default_uint8", "Upstream default uint8", "#c35454"),
                                      (0.25, "clipped_uint8", "Clipped uint8", "#d6a135")):
        ax.bar(np.arange(len(records)) + shift, [r[field] for r in records], width=0.24, label=label, color=color)
    ax.set_xticks(range(len(records)), [f"{r['feature']}\nID {r['sample_id']}" for r in records], fontsize=9)
    ax.set_ylabel("Numeric pixel value")
    ax.set_title("Encoding audit: largest numeric distortions (illustrations, not selected model results)\nAn above-range measurement can become a near-zero uint8 pixel", fontsize=12)
    ax.legend(fontsize=9)
    fig.savefig(output / "encoding_audit.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--output-dir", type=Path, default=PROTOCOL_PATH.parent)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "artifacts/07_som2img_layout")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "results.json").exists() or args.checkpoint_dir.exists():
        raise FileExistsError("Choose fresh results and checkpoint locations")
    protocol = json.loads(PROTOCOL_PATH.read_text())
    result = run(protocol, args.upstream, args.checkpoint_dir)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    (args.output_dir / "results.json").write_text(payload)
    make_figures(result, args.output_dir)
    print(json.dumps({"summary": result["summary"], "encoding_audit": result["encoding_audit"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
