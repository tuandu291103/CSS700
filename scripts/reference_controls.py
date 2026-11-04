"""Experiment 09: model-free ranking controls and real-record IG references."""

import argparse
import itertools
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np

from scripts.som2img_attributions import (
    LAYOUT_METHODS,
    compare,
    decode,
    describe,
    describe_agreement,
    encode,
    load_frozen,
    predictor,
)
from scripts.wisconsin_explanations import integrated_gradients
from scripts.wisconsin_learning import FEATURES, ROOT
from scripts.wisconsin_models import sha256, state_digest

PROTOCOL = ROOT / "experiments/09_reference_controls/protocol.json"
RANKERS = ("ig_magnitude", "distance", "ig_support")


def select_reference_positions(labels, per_class, seed):
    labels = np.asarray(labels)
    if set(labels.tolist()) != {0, 1} or per_class < 1:
        raise ValueError("Reference selection requires binary labels and a positive count")
    rng = np.random.default_rng(seed)
    result = []
    for label in (0, 1):
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < per_class:
            raise ValueError("Insufficient fitting references")
        result.extend(sorted(rng.choice(candidates, per_class, replace=False).tolist()))
    return result


def reference_inventory(data, manifest, config):
    fit = data["fit"]["scaled"].astype(np.float32)
    selected = select_reference_positions(data["fit"]["y"], config["per_class"], config["seed"])
    refs = {
        "fit_median": {"kind": "constructed", "label": None,
                       "values": np.median(data["fit"]["scaled"], axis=0).astype(np.float32).tolist()},
        "fit_minimum": {"kind": "constructed", "label": None, "values": np.zeros(fit.shape[1]).tolist()},
    }
    for position in selected:
        refs[f"real_fit_{position}"] = {
            "kind": "real", "label": int(data["fit"]["y"][position]), "fit_position": position,
            "original_row_index": manifest["partitions"]["fit"]["row_indices_zero_based"][position],
            "sample_id": manifest["partitions"]["fit"]["sample_ids"][position], "values": fit[position].tolist()}
    return refs


def random_subsets(n_features, k, repeats, seed, case_position):
    if not 1 <= k <= n_features or repeats < 1:
        raise ValueError("Invalid random subset dimensions")
    rng = np.random.default_rng(seed + case_position)
    return np.array([rng.choice(n_features, k, replace=False) for _ in range(repeats)])


def midrank(value, reference):
    reference = np.asarray(reference)
    return float(np.mean(reference < value) + 0.5 * np.mean(reference == value))


def outcome(left, right):
    tolerance = 1e-6 * max(1.0, abs(left), abs(right))
    if abs(left - right) <= tolerance:
        return "tie"
    return "win" if left > right else "loss"


def ranking_control(predict, row, baseline, ig, subsets):
    row, baseline, ig = np.asarray(row), np.asarray(baseline), np.asarray(ig)
    if row.shape != baseline.shape or row.shape != ig.shape or row.ndim != 1:
        raise ValueError("Expected aligned input, reference and attribution vectors")
    if not all(np.isfinite(a).all() for a in (row, baseline, ig)):
        raise ValueError("Nonfinite control input")
    original = float(predict(row[None])[0])
    batch = np.repeat(row[None], len(subsets), axis=0)
    batch[np.arange(len(subsets))[:, None], subsets] = baseline[subsets]
    random_drops = original - predict(batch)
    scores = {"ig_magnitude": np.abs(ig), "distance": np.abs(row - baseline), "ig_support": ig}
    rankings = {}
    for name, values in scores.items():
        if np.ptp(values) == 0:
            rankings[name] = {"eligible": False, "reason": "constant ranking vector"}
            continue
        selected = np.argsort(-values, kind="stable")[:subsets.shape[1]]
        changed = row.copy()
        changed[selected] = baseline[selected]
        drop = original - float(predict(changed[None])[0])
        rankings[name] = {"eligible": True, "selected_indices": selected.tolist(),
                          "signed_score_drop": drop, "absolute_score_change": abs(drop),
                          "absolute_random_percentile": midrank(abs(drop), np.abs(random_drops)),
                          "signed_random_percentile": midrank(drop, random_drops)}
    a, b = rankings["ig_magnitude"], rankings["distance"]
    primary = {"eligible": a["eligible"] and b["eligible"]}
    if primary["eligible"]:
        primary.update({"outcome": outcome(a["absolute_score_change"], b["absolute_score_change"]),
                        "absolute_change_difference": a["absolute_score_change"] - b["absolute_score_change"],
                        "percentile_difference": a["absolute_random_percentile"] - b["absolute_random_percentile"]})
    a = rankings["ig_support"]
    secondary = {"eligible": a["eligible"] and b["eligible"]}
    if secondary["eligible"]:
        secondary.update({"outcome": outcome(a["signed_score_drop"], b["signed_score_drop"]),
                          "signed_drop_difference": a["signed_score_drop"] - b["signed_score_drop"]})
    return {"random_signed_drops": random_drops.tolist(), "rankings": rankings,
            "primary": primary, "secondary": secondary}


def stability_comparisons(runs):
    pairs = []
    for left, right in itertools.combinations(runs, 2):
        if left["method"] != right["method"]:
            continue
        if left["layout_key"] == right["layout_key"]:
            axis = "cnn_seed"
        elif left["cnn_seed"] == right["cnn_seed"]:
            axis = "layout_seed"
        else:
            continue
        for a, b in zip(left["cases"], right["cases"], strict=True):
            if (a["position"], a["label"]) != (b["position"], b["label"]):
                raise ValueError("Unmatched stability cases")
            for reference in a["references"]:
                pairs.append({"method": left["method"], "axis": axis, "left": left["key"], "right": right["key"],
                              "reference": reference, "position": a["position"],
                              **compare(a["references"][reference]["ig"], b["references"][reference]["ig"])})
    return pairs


def sensitivity_comparisons(runs, references):
    real = [name for name, r in references.items() if r["kind"] == "real"]
    pairs = []
    choices = list(itertools.combinations(real, 2)) + [("fit_median", name) for name in real]
    for run in runs:
        for case in run["cases"]:
            for a, b in choices:
                group = "median_vs_real" if a == "fit_median" else (
                    "same_reference_class" if references[a]["label"] == references[b]["label"] else "cross_reference_class")
                pairs.append({"method": run["method"], "model": run["key"], "position": case["position"],
                              "left": a, "right": b, "group": group,
                              **compare(case["references"][a]["ig"], case["references"][b]["ig"])})
    return pairs


def summarize_controls(rows):
    primary = [r["control"]["primary"] for r in rows if r["control"]["primary"]["eligible"]]
    secondary = [r["control"]["secondary"] for r in rows if r["control"]["secondary"]["eligible"]]
    return {
        "observations": len(rows), "primary_eligible": len(primary),
        "primary_outcomes": {name: sum(r["outcome"] == name for r in primary) for name in ("win", "tie", "loss")},
        "primary_percentile_difference": describe([r["percentile_difference"] for r in primary]),
        "secondary_outcomes": {name: sum(r["outcome"] == name for r in secondary) for name in ("win", "tie", "loss")},
        "absolute_random_percentile": {name: describe([
            r["control"]["rankings"][name]["absolute_random_percentile"] for r in rows if r["control"]["rankings"][name]["eligible"]])
            for name in RANKERS},
        "ig_distance_agreement": describe_agreement([r["distance_agreement"] for r in rows]),
        "ig_tolerance_failures": sum(not r["ig_diagnostics"]["numerical_tolerance_met"] for r in rows),
        "max_ig_normalized_residual": max(r["ig_diagnostics"]["normalized_residual"] for r in rows),
    }


def summarize(runs, references, stability, sensitivity):
    real = [name for name, r in references.items() if r["kind"] == "real"]
    conditions = {name: [name] for name in references} | {"real_all": real}
    summary = {}
    for method in LAYOUT_METHODS:
        cases = [c for r in runs if r["method"] == method for c in r["cases"]]
        item = {"reference_conditions": {}, "stability": {}, "reference_sensitivity": {}, "real_class_strata": {}}
        for condition, names in conditions.items():
            rows = [c["references"][name] for c in cases for name in names]
            item["reference_conditions"][condition] = summarize_controls(rows)
            item["stability"][condition] = {
                axis: describe_agreement([p for p in stability if p["method"] == method and p["axis"] == axis
                                          and p["reference"] in names]) for axis in ("cnn_seed", "layout_seed")}
        for group in ("same_reference_class", "cross_reference_class", "median_vs_real"):
            item["reference_sensitivity"][group] = describe_agreement([
                p for p in sensitivity if p["method"] == method and p["group"] == group])
        for same in (True, False):
            rows = [c["references"][name] for c in cases for name in real
                    if (references[name]["label"] == c["label"]) == same]
            item["real_class_strata"]["same_as_case" if same else "opposite_to_case"] = summarize_controls(rows)
        summary[method] = item
    return summary


def prepare(config):
    path = ROOT / config["attribution_results"]
    if sha256(path) != config["attribution_results_sha256"]:
        raise ValueError("Experiment 08 evidence changed")
    previous = json.loads(path.read_text())
    if sha256(ROOT / "experiments/08_som2img_attributions/protocol.json") != config["attribution_protocol_sha256"]:
        raise ValueError("Experiment 08 protocol changed")
    for name, digest in previous["source_hashes"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Attribution source changed: {name}")
    data, manifest, frozen = load_frozen(previous["protocol"])
    selected = previous["selected_evaluation_positions"]
    np.testing.assert_array_equal(data["evaluation"]["scaled"][selected].astype(np.float32), previous["selected_scaled_values"])
    references = reference_inventory(data, manifest, config["real_references"])
    for name in ("fit_median", "fit_minimum"):
        np.testing.assert_array_equal(references[name]["values"], previous["references"][name])
    if {r["original_row_index"] for r in references.values() if r["kind"] == "real"} & set(manifest["partitions"]["evaluation"]["row_indices_zero_based"]):
        raise AssertionError("Reference selected from evaluation rows")
    return previous, data, manifest, frozen, references


def run_experiment(config):
    previous, data, manifest, frozen, references = prepare(config)
    selected = previous["selected_evaluation_positions"]
    old_runs = {r["key"]: r for r in previous["runs"]}
    settings = config["replacement"]
    banks = {position: random_subsets(len(FEATURES), settings["k"], settings["random_repeats"], settings["seed"], position)
             for position in selected}
    runs = []
    for saved, fm, model, checkpoint_hash in frozen:
        key = f"{saved['layout_key']}_cnn_{saved['cnn_seed']}"
        old = old_runs[key]
        if state_digest(model) != old["state_sha256"] or checkpoint_hash != old["checkpoint_sha256"]:
            raise ValueError("Model differs from Experiment 08")
        record = {name: old[name] for name in ("key", "method", "layout_key", "layout_seed", "cnn_seed", "state_sha256", "checkpoint_sha256")}
        record["cases"] = []
        old_cases = {c["position"]: c for c in old["cases"]}
        for position in selected:
            prior = old_cases[position]
            row = data["evaluation"]["scaled"][position].astype(np.float32)
            label = int(data["evaluation"]["y"][position])
            if label != prior["label"]:
                raise ValueError("Case target changed")
            predict = predictor(model, fm, prior["target_sign"])
            score = float(predict(row[None])[0])
            if score != prior["target_score"]:
                raise ValueError("Saved target score changed")
            case = {name: prior[name] for name in ("position", "sample_id", "label", "target_sign", "prediction", "target_score")}
            case["references"] = {}
            for name, ref in references.items():
                baseline = np.array(ref["values"], dtype=np.float32)
                if ref["kind"] == "constructed":
                    vector = np.array(prior["references"][name]["vectors"]["integrated_gradients"])
                    diag = prior["references"][name]["ig_diagnostics"]
                    origin = "reused_experiment_08"
                else:
                    image_ig, diag = integrated_gradients(
                        model, encode(row[None], fm), encode(baseline[None], fm), prior["target_sign"],
                        steps=config["integrated_gradients"]["steps"], maximum_steps=config["integrated_gradients"]["maximum_steps"])
                    vector = decode(image_ig[:, 0], fm, len(FEATURES))[0]
                    origin = "computed_real_reference"
                case["references"][name] = {
                    "ig": vector.tolist(), "ig_diagnostics": diag, "ig_origin": origin,
                    "distance_agreement": compare(vector, np.abs(row - baseline)),
                    "control": ranking_control(predict, row, baseline, vector, banks[position])}
            record["cases"].append(case)
        if state_digest(model) != record["state_sha256"]:
            raise AssertionError("Attribution changed weights")
        runs.append(record)
        print(f"Completed {key}: four real references and two reused contrasts", flush=True)
    stability = stability_comparisons(runs)
    sensitivity = sensitivity_comparisons(runs, references)
    dependencies = [Path(__file__), ROOT / "scripts/som2img_attributions.py", ROOT / "scripts/som2img_learning.py",
                    ROOT / "scripts/wisconsin_explanations.py", ROOT / "scripts/wisconsin_models.py", ROOT / "scripts/wisconsin_learning.py"]
    return {"date": config["recorded_date"], "protocol": config, "protocol_sha256": sha256(PROTOCOL),
            "source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in dependencies},
            "environment": {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(),
                            **{name: version(name) for name in ("numpy", "scipy", "torch", "captum", "matplotlib")}},
            "references": references, "features": list(FEATURES), "selected_evaluation_positions": selected,
            "selected_scaled_values": previous["selected_scaled_values"],
            "random_feature_subsets": {str(k): v.tolist() for k, v in banks.items()},
            "verification": {"all_saved_model_states_and_probabilities_match": True,
                             "excluded_learning_test_count": manifest["excluded_learning_test_count"]},
            "runs": runs, "seed_stability": stability, "reference_sensitivity": sensitivity,
            "summary": summarize(runs, references, stability, sensitivity)}


def make_figures(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Fixed": "#586777", "SnakeRandom": "#c28735", "SOM2IMG": "#237f85"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), constrained_layout=True)
    for ax, condition, title in zip(axes, ("fit_median", "fit_minimum", "real_all"),
                                    ("Median", "Minimum", "Four real fitting records"), strict=True):
        for i, method in enumerate(LAYOUT_METHODS):
            stats = report["summary"][method]["reference_conditions"][condition]
            counts = stats["primary_outcomes"]
            start = 0
            for name, color in (("win", "#237f85"), ("tie", "#bac2c9"), ("loss", "#c28735")):
                value = counts[name] / stats["primary_eligible"]
                ax.barh(i, value, left=start, color=color, label=name.title() if i == 0 else None)
                if value > 0.09:
                    ax.text(start + value / 2, i, f"{value:.0%}", ha="center", va="center", color="white" if name != "tie" else "black")
                start += value
        ax.set_yticks(range(3), ("Fixed", "Random", "SOM"))
        ax.set_xlim(0, 1)
        ax.set_xlabel("Fraction of paired observations")
        ax.set_title(title)
        ax.invert_yaxis()
    axes[0].legend(ncol=3, loc="lower left", bbox_to_anchor=(0, -0.34), frameon=False)
    fig.suptitle("Magnitude-ranked IG vs model-free distance: top-5 absolute score change\nDependent model/case/reference observations; wins are diagnostic, not biological validation")
    fig.savefig(output / "model_free_control.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    real = {k for k, v in report["references"].items() if v["kind"] == "real"}
    for ax, axis, title in zip(axes, ("cnn_seed", "layout_seed"),
                               ("CNN seeds within layout", "Layout seeds within CNN seed"), strict=True):
        for i, method in enumerate(LAYOUT_METHODS):
            for shift, references, marker in ((-0.14, {"fit_median"}, "o"), (0.14, real, "s")):
                rows = [p for p in report["seed_stability"] if p["method"] == method and p["axis"] == axis and p["reference"] in references]
                keys = sorted({(p["left"], p["right"]) for p in rows})
                values = [np.mean([p["absolute_spearman"] for p in rows if (p["left"], p["right"]) == key]) for key in keys]
                if values:
                    ax.scatter(i + shift + np.linspace(-0.045, 0.045, len(values)), values, color=colors[method], marker=marker, s=34)
                    ax.plot([i + shift - 0.08, i + shift + 0.08], [np.mean(values)] * 2, color="black", lw=1.5)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.1, 1.05)
        ax.set_xticks(range(3), ("Fixed", "Random", "SOM"))
        ax.set_title(title)
        ax.set_ylabel("IG magnitude-rank correlation")
    axes[1].text(0, 0.1, "one layout", ha="center", fontsize=8)
    fig.suptitle("Seed stability with median (circles) and real-record references (squares)\nDots average matched cases/references within each model pair; no confidence intervals")
    fig.savefig(output / "real_reference_stability.png", dpi=180)
    plt.close(fig)

    run = next(r for r in report["runs"] if r["method"] == "SOM2IMG")
    case = run["cases"][0]
    names = list(report["references"])
    labels = [name if ref["kind"] == "constructed" else f"ID {ref['sample_id']} ({'M' if ref['label'] else 'B'})"
              for name, ref in report["references"].items()]
    matrix = np.array([case["references"][name]["ig"] for name in names])
    matrix /= np.maximum(np.abs(matrix).sum(axis=1, keepdims=True), 1e-12)
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    limit = float(np.abs(matrix).max())
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_yticks(range(len(names)), labels, fontsize=9)
    ax.set_xticks(range(30), report["features"], rotation=65, ha="right", fontsize=8)
    ax.set_title(f"Same SOM model and case {case['sample_id']}; only the reference changes\n"
                 "First recorded SOM/CNN seed and case; signed IG normalized to unit absolute sum")
    fig.colorbar(im, ax=ax, label="Signed share of total absolute IG")
    fig.savefig(output / "reference_example.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/09_reference_controls")
    args = parser.parse_args()
    if (args.output_dir / "results.json").exists():
        raise FileExistsError("Choose a fresh output directory")
    result = run_experiment(json.loads(PROTOCOL.read_text()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    make_figures(result, args.output_dir)
    print(f"Saved all {len(result['runs'])} models and three figures to {args.output_dir}")


if __name__ == "__main__":
    main()
