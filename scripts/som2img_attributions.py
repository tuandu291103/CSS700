"""Experiment 08: feature-space explanations of the frozen Experiment 07 CNNs."""

import argparse
import itertools
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from captum.attr import Occlusion
from scipy.stats import spearmanr

from scripts.som2img_learning import decode, positions, prepare_data
from scripts.wisconsin_explanations import LogitMargin, integrated_gradients
from scripts.wisconsin_learning import FEATURES, ROOT
from scripts.wisconsin_models import (
    SmallGridCNN,
    cnn_probabilities,
    seed_cpu,
    sha256,
    state_digest,
)

PROTOCOL = ROOT / "experiments/08_som2img_attributions/protocol.json"
EXPLAINERS = ("integrated_gradients", "single_feature_occlusion")
LAYOUT_METHODS = ("Fixed", "SnakeRandom", "SOM2IMG")
AGREEMENT_METRICS = ("signed_spearman", "absolute_spearman", "top5_jaccard")


def encode(values, feature_map):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected a finite sample-by-feature matrix")
    feature_map = np.asarray(feature_map)
    pos = positions(feature_map, values.shape[1])
    images = np.zeros((len(values), *feature_map.shape), dtype=np.float32)
    images[:, pos[:, 0], pos[:, 1]] = values
    return images[:, None]


def choose_cases(labels, per_class):
    labels = np.asarray(labels)
    if per_class < 1 or set(labels.tolist()) != {0, 1}:
        raise ValueError("Expected binary labels and a positive case count")
    selected = []
    for label in (0, 1):
        available = np.flatnonzero(labels == label)
        if len(available) < per_class:
            raise ValueError("Insufficient cases for the specified balanced selection")
        selected.extend(available[:per_class].tolist())
    return sorted(selected)


def predictor(model, feature_map, sign):
    def predict(values):
        images = encode(values, feature_map)
        with torch.no_grad():
            return LogitMargin(model, sign)(torch.from_numpy(images)).numpy().astype(float)
    return predict


def compare(first, second, k=5):
    first, second = np.asarray(first), np.asarray(second)
    if first.shape != second.shape or first.ndim != 1 or not 1 <= k <= first.size:
        raise ValueError("Expected equal vectors and a valid top-k")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Nonfinite attribution")
    signed = None if np.ptp(first) == 0 or np.ptp(second) == 0 else float(spearmanr(first, second).statistic)
    a, b = np.abs(first), np.abs(second)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return {"signed_spearman": signed, "absolute_spearman": None, "top5_jaccard": None}
    left = set(np.argsort(-a, kind="stable")[:k])
    right = set(np.argsort(-b, kind="stable")[:k])
    return {"signed_spearman": signed, "absolute_spearman": float(spearmanr(a, b).statistic),
            "top5_jaccard": len(left & right) / len(left | right)}


def occlusion(model, row, baseline, feature_map, sign):
    values = Occlusion(LogitMargin(model, sign)).attribute(
        torch.from_numpy(encode(row[None], feature_map)),
        sliding_window_shapes=(1, 1, 1), strides=1,
        baselines=torch.from_numpy(encode(baseline[None], feature_map)),
        perturbations_per_eval=30,
    ).detach().numpy()
    decoded = decode(values[:, 0], feature_map, len(row))[0]
    changed = np.repeat(row[None], len(row), axis=0)
    changed[np.arange(len(row)), np.arange(len(row))] = baseline
    predict = predictor(model, feature_map, sign)
    direct = predict(row[None])[0] - predict(changed)
    np.testing.assert_allclose(decoded, direct, atol=2e-5, rtol=1e-5)
    return decoded, float(np.max(np.abs(decoded - direct)))


def replacement(predict, row, baseline, scores, config, case_position):
    original = float(predict(row[None])[0])
    rng = np.random.default_rng(config["seed"] + case_position)
    report = {}
    for k in config["k"]:
        subsets = np.array([rng.choice(len(row), k, replace=False) for _ in range(config["random_repeats"])])
        batch = np.repeat(row[None], len(subsets), axis=0)
        batch[np.arange(len(subsets))[:, None], subsets] = baseline[subsets]
        random_drops = original - predict(batch)
        methods = {}
        for method, values in scores.items():
            if np.ptp(values) == 0:
                methods[method] = {"eligible": False, "reason": "constant signed attribution"}
                continue
            selected = np.argsort(-values, kind="stable")[:k]
            changed = row.copy()
            changed[selected] = baseline[selected]
            drop = original - float(predict(changed[None])[0])
            percentile = np.mean(random_drops < drop) + 0.5 * np.mean(random_drops == drop)
            methods[method] = {"eligible": True, "selected_indices": selected.tolist(),
                               "positive_attribution_count": int((values > 0).sum()),
                               "score_drop": drop, "excess_over_random_mean": drop - float(random_drops.mean()),
                               "random_midrank_percentile": float(percentile)}
        report[str(k)] = {"random_drops": random_drops.tolist(),
                          "random_mean_drop": float(random_drops.mean()), "methods": methods}
    return report


def describe(values):
    valid = [v for v in values if v is not None]
    return {"n": len(valid), "undefined_n": len(values) - len(valid),
            "mean": float(np.mean(valid)) if valid else None,
            "minimum": min(valid) if valid else None, "maximum": max(valid) if valid else None}


def describe_agreement(rows):
    return {metric: describe([r[metric] for r in rows]) for metric in AGREEMENT_METRICS}


def pairwise_stability(runs):
    comparisons = []
    for left, right in itertools.combinations(runs, 2):
        if left["method"] != right["method"]:
            continue
        axis = None
        if left["layout_key"] == right["layout_key"]:
            axis = "cnn_seed"
        elif left["cnn_seed"] == right["cnn_seed"]:
            axis = "layout_seed"
        if axis is None:
            continue
        for a, b in zip(left["cases"], right["cases"], strict=True):
            if a["position"] != b["position"] or a["label"] != b["label"]:
                raise ValueError("Unmatched case or target in stability comparison")
            for reference in a["references"]:
                for explainer in EXPLAINERS:
                    comparisons.append({"axis": axis, "method": left["method"], "reference": reference,
                                        "explainer": explainer, "left": left["key"], "right": right["key"],
                                        "position": a["position"],
                                        **compare(a["references"][reference]["vectors"][explainer],
                                                  b["references"][reference]["vectors"][explainer])})
    return comparisons


def summarize(runs, pairs):
    summary = {}
    for method in LAYOUT_METHODS:
        models = [r for r in runs if r["method"] == method]
        cases = [c for r in models for c in r["cases"]]
        entry = {"models": len(models), "model_case_observations": len(cases),
                 "correct_predictions": sum(c["prediction"] == c["label"] for c in cases),
                 "references": {}, "stability": {}, "reference_sensitivity": {}}
        for reference in ("fit_median", "fit_minimum"):
            records = [c["references"][reference] for c in cases]
            ref = {"ig_occlusion_agreement": describe_agreement([r["method_agreement"] for r in records]),
                   "ig_input_difference_agreement": describe_agreement([r["input_difference_agreement"] for r in records]),
                   "ig_maximum_normalized_residual": max(r["ig_diagnostics"]["normalized_residual"] for r in records),
                   "ig_tolerance_failures": sum(not r["ig_diagnostics"]["numerical_tolerance_met"] for r in records),
                   "maximum_occlusion_direct_error": max(r["occlusion_direct_max_error"] for r in records),
                   "replacement": {}}
            for explainer in EXPLAINERS:
                ref["replacement"][explainer] = {}
                for k in ("1", "3", "5"):
                    rows = [r["replacement"][k]["methods"][explainer] for r in records]
                    eligible = [r for r in rows if r["eligible"]]
                    ref["replacement"][explainer][k] = {
                        "eligible_n": len(eligible), "ineligible_n": len(rows) - len(eligible),
                        "exceeding_random_mean_n": sum(r["excess_over_random_mean"] > 0 for r in eligible),
                        "fraction_exceeding_random_mean": float(np.mean([r["excess_over_random_mean"] > 0 for r in eligible])) if eligible else None,
                        "random_midrank_percentile": describe([r["random_midrank_percentile"] for r in eligible]),
                        "excess_over_random_mean": describe([r["excess_over_random_mean"] for r in eligible])}
            entry["references"][reference] = ref
        for axis in ("cnn_seed", "layout_seed"):
            entry["stability"][axis] = {}
            for reference in ("fit_median", "fit_minimum"):
                entry["stability"][axis][reference] = {}
                for explainer in EXPLAINERS:
                    chosen = [p for p in pairs if p["method"] == method and p["axis"] == axis
                              and p["reference"] == reference and p["explainer"] == explainer]
                    entry["stability"][axis][reference][explainer] = describe_agreement(chosen)
        for explainer in EXPLAINERS:
            entry["reference_sensitivity"][explainer] = describe_agreement([
                compare(c["references"]["fit_median"]["vectors"][explainer],
                        c["references"]["fit_minimum"]["vectors"][explainer]) for c in cases])
        entry["reset_ig_agreement"] = describe_agreement([c["reset"]["agreement"] for c in cases])
        entry["reset_ig_tolerance_failures"] = sum(not c["reset"]["diagnostics"]["numerical_tolerance_met"] for c in cases)
        summary[method] = entry
    return summary


def load_frozen(config):
    previous_path = ROOT / config["previous_results"]
    if sha256(previous_path) != config["previous_results_sha256"]:
        raise ValueError("Experiment 07 results changed")
    previous = json.loads(previous_path.read_text())
    previous_protocol = ROOT / "experiments/07_som2img_layout/protocol.json"
    if sha256(previous_protocol) != config["previous_protocol_sha256"]:
        raise ValueError("Experiment 07 protocol changed")
    for name, digest in previous["source_hashes"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Recorded model/data source changed: {name}")
    data, manifest, _ = prepare_data(previous["protocol"])
    if manifest != previous["data"]:
        raise ValueError("Data manifest changed")
    layouts = {r["key"]: r for r in previous["layouts"]}
    frozen = []
    for run in previous["runs"]:
        fm = np.array(layouts[run["layout_key"]]["feature_map"])
        seed_cpu(run["cnn_seed"])
        model = SmallGridCNN().eval()
        path = ROOT / config["checkpoint_directory"] / run["local_checkpoint_filename"]
        model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
        if state_digest(model) != run["diagnostics"]["restored_state_sha256"]:
            raise ValueError(f"Model weights changed: {path.name}")
        images = encode(data["evaluation"]["scaled"], fm)
        np.testing.assert_array_equal(decode(images[:, 0], fm, len(FEATURES)),
                                      data["evaluation"]["scaled"].astype(np.float32))
        probabilities = cnn_probabilities(model, images)
        np.testing.assert_array_equal(probabilities, run["probability_m"])
        frozen.append((run, fm, model, sha256(path)))
    if len(frozen) != 21 or len(layouts) != 7:
        raise ValueError("Incomplete model/layout inventory")
    return data, manifest, frozen


def run_experiment(config):
    data, manifest, frozen = load_frozen(config)
    selected = choose_cases(data["evaluation"]["y"], config["cases_per_class"])
    references = {"fit_median": np.median(data["fit"]["scaled"], axis=0).astype(np.float32),
                  "fit_minimum": np.zeros(len(FEATURES), dtype=np.float32)}
    output = []
    ig_config = config["integrated_gradients"]
    for saved, fm, model, checkpoint_hash in frozen:
        key = f"{saved['layout_key']}_cnn_{saved['cnn_seed']}"
        seed_cpu(saved["cnn_seed"] + 9100)
        reset_model = SmallGridCNN().eval()
        record = {k: saved[k] for k in ("method", "layout_key", "layout_seed", "cnn_seed")}
        record.update({"key": key, "state_sha256": state_digest(model), "checkpoint_sha256": checkpoint_hash,
                       "reset_state_sha256": state_digest(reset_model), "cases": []})
        for position in selected:
            row = data["evaluation"]["scaled"][position].astype(np.float32)
            label = int(data["evaluation"]["y"][position])
            sign = float(2 * label - 1)
            images = encode(row[None], fm)
            predict = predictor(model, fm, sign)
            case = {"position": position, "sample_id": manifest["partitions"]["evaluation"]["sample_ids"][position],
                    "label": label, "target_sign": sign, "prediction": int(saved["probability_m"][position] > 0.5),
                    "probability_m": saved["probability_m"][position], "target_score": float(predict(row[None])[0]),
                    "references": {}}
            for reference, baseline in references.items():
                ig, diagnostics = integrated_gradients(model, images, encode(baseline[None], fm), sign,
                                                       steps=ig_config["steps"], maximum_steps=ig_config["maximum_steps"])
                ig = decode(ig[:, 0], fm, len(FEATURES))[0]
                occ, direct_error = occlusion(model, row, baseline, fm, sign)
                scores = dict(zip(EXPLAINERS, (ig, occ), strict=True))
                case["references"][reference] = {
                    "vectors": {k: v.tolist() for k, v in scores.items()}, "ig_diagnostics": diagnostics,
                    "occlusion_direct_max_error": direct_error, "method_agreement": compare(ig, occ),
                    "input_difference_agreement": compare(ig, np.abs(row - baseline)),
                    "replacement": replacement(predict, row, baseline, scores, config["replacement"], position)}
            reset_ig, reset_diag = integrated_gradients(reset_model, images, encode(references["fit_median"][None], fm),
                                                       sign, steps=ig_config["steps"], maximum_steps=ig_config["maximum_steps"])
            reset_ig = decode(reset_ig[:, 0], fm, len(FEATURES))[0]
            case["reset"] = {"vector": reset_ig.tolist(), "diagnostics": reset_diag,
                             "agreement": compare(case["references"]["fit_median"]["vectors"][EXPLAINERS[0]], reset_ig)}
            record["cases"].append(case)
        if state_digest(model) != record["state_sha256"]:
            raise AssertionError("Attribution modified trained weights")
        output.append(record)
        print(f"Completed {key}: {len(selected)} matched cases, two references and reset control", flush=True)
    pairs = pairwise_stability(output)
    paths = [Path(__file__), ROOT / "scripts/som2img_learning.py", ROOT / "scripts/wisconsin_explanations.py",
             ROOT / "scripts/wisconsin_models.py", ROOT / "scripts/wisconsin_learning.py"]
    return {"date": config["recorded_date"], "protocol": config, "protocol_sha256": sha256(PROTOCOL),
            "source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in paths},
            "environment": {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(),
                            **{name: version(name) for name in ("numpy", "scipy", "torch", "captum", "matplotlib")}},
            "features": list(FEATURES), "selected_evaluation_positions": selected,
            "selected_scaled_values": data["evaluation"]["scaled"][selected].astype(np.float32).tolist(),
            "references": {k: v.tolist() for k, v in references.items()},
            "verification": {"all_saved_probabilities_exact": True, "all_state_hashes_match": True,
                             "excluded_learning_test_count": manifest["excluded_learning_test_count"]},
            "runs": output, "pairwise_stability": pairs, "summary": summarize(output, pairs)}


def make_figures(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Fixed": "#586777", "SnakeRandom": "#c28735", "SOM2IMG": "#237f85"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, axis, title in zip(axes[:2], ("cnn_seed", "layout_seed"),
                               ("CNN seeds, fixed layout", "Layout seeds, fixed CNN seed"), strict=True):
        for i, method in enumerate(LAYOUT_METHODS):
            rows = [p for p in report["pairwise_stability"] if p["method"] == method and p["axis"] == axis
                    and p["reference"] == "fit_median" and p["explainer"] == EXPLAINERS[0]]
            pair_keys = sorted({(p["left"], p["right"]) for p in rows})
            means = [np.mean([p["absolute_spearman"] for p in rows if (p["left"], p["right"]) == key]) for key in pair_keys]
            if means:
                ax.scatter(i + np.linspace(-0.12, 0.12, len(means)), means, color=colors[method], s=32)
                ax.plot([i - 0.2, i + 0.2], [np.mean(means)] * 2, color=colors[method], lw=2)
            else:
                ax.text(i, 0.1, "one layout", ha="center", fontsize=8)
        ax.set_title(title)
    for i, method in enumerate(LAYOUT_METHODS):
        runs = [r for r in report["runs"] if r["method"] == method]
        means = [np.mean([compare(c["references"]["fit_median"]["vectors"][EXPLAINERS[0]],
                                  c["references"]["fit_minimum"]["vectors"][EXPLAINERS[0]])["absolute_spearman"]
                          for c in r["cases"]]) for r in runs]
        axes[2].scatter(i + np.linspace(-0.12, 0.12, len(means)), means, color=colors[method], s=32)
        axes[2].plot([i - 0.2, i + 0.2], [np.mean(means)] * 2, color=colors[method], lw=2)
    axes[2].set_title("Median vs minimum reference")
    for ax in axes:
        ax.set_xticks(range(3), ("Fixed", "Random", "SOM"))
        ax.set_ylim(-0.1, 1.05)
        ax.set_ylabel("IG magnitude rank correlation")
    fig.suptitle("Feature-space stability: dots are pair/model means across 12 cases, not confidence intervals")
    fig.savefig(output / "attribution_stability.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, reference in zip(axes, ("fit_median", "fit_minimum"), strict=True):
        for i, method in enumerate(LAYOUT_METHODS):
            for j, explainer in enumerate(EXPLAINERS):
                runs = [r for r in report["runs"] if r["method"] == method]
                means = [np.mean([c["references"][reference]["replacement"]["5"]["methods"][explainer]["random_midrank_percentile"]
                                  for c in r["cases"]]) for r in runs]
                x = i + (j - 0.5) * 0.3
                ax.scatter(x + np.linspace(-0.05, 0.05, len(means)), means, color=colors[method],
                           marker="o" if j == 0 else "s", s=35)
        ax.set_xticks(range(3), ("Fixed", "Random", "SOM"))
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="gray", lw=0.7, ls="--")
        ax.set_title(reference.replace("_", " ").title())
        ax.set_ylabel("Mean within-model random percentile")
    fig.suptitle("Top-5 signed-support replacement: circles = IG, squares = occlusion\n100 matched random subsets; descriptive score, not a p-value")
    fig.savefig(output / "feature_replacement.png", dpi=180)
    plt.close(fig)

    chosen = [next(r for r in report["runs"] if r["method"] == m) for m in LAYOUT_METHODS]
    matrix = np.array([r["cases"][0]["references"]["fit_median"]["vectors"][EXPLAINERS[0]] for r in chosen])
    # Normalize each row for display only; raw signed vectors are retained in JSON.
    matrix /= np.maximum(np.abs(matrix).sum(axis=1, keepdims=True), 1e-12)
    fig, ax = plt.subplots(figsize=(13, 4), constrained_layout=True)
    limit = float(np.abs(matrix).max())
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(range(30), report["features"], rotation=65, ha="right", fontsize=8)
    ax.set_yticks(range(3), ("Fixed", "Random 42", "SOM 42"))
    case = chosen[0]["cases"][0]
    ax.set_title(f"Same case, named features: ID {case['sample_id']}, target {'malignant' if case['label'] else 'benign'}\n"
                 "First recorded CNN/layout seeds; median-reference IG, normalized absolute sum = 1")
    fig.colorbar(im, ax=ax, label="Signed share of total absolute IG")
    fig.savefig(output / "named_feature_example.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/08_som2img_attributions")
    args = parser.parse_args()
    if (args.output_dir / "results.json").exists():
        raise FileExistsError("Choose a fresh output directory; existing evidence is never overwritten")
    config = json.loads(PROTOCOL.read_text())
    result = run_experiment(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    make_figures(result, args.output_dir)
    print(f"Saved {len(result['runs'])} model records and three figures to {args.output_dir}")


if __name__ == "__main__":
    main()
