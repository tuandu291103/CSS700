"""Learning experiment 03: frozen-model explanations and bounded diagnostics."""

import argparse
import copy
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import shap
import torch
from captum.attr import IntegratedGradients, LayerGradCam
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr
from torch import nn

from scripts.wisconsin_learning import FEATURES, ROOT, decode, encode
from scripts.wisconsin_models import (
    SmallGridCNN,
    load_inputs,
    seed_cpu,
    sha256,
    state_digest,
)

PROTOCOL = ROOT / "experiments/03_wisconsin_explanations/protocol.json"
MODEL_RESULTS = ROOT / "experiments/02_wisconsin_models/results.json"


class LogitMargin(nn.Module):
    def __init__(self, model, sign=1.0):
        super().__init__()
        self.model = model
        self.sign = sign

    def forward(self, inputs):
        logits = self.model(inputs)
        return self.sign * (logits[:, 1] - logits[:, 0])


def predictor(model, mapping, shape=(5, 6), sign=1.0):
    """Tabular perturbations are re-encoded through the frozen feature mapping."""
    def predict(values):
        images, _ = encode(np.asarray(values), mapping, shape)
        images = images.astype(np.float32)[:, None]
        model.eval()
        with torch.no_grad():
            return np.concatenate([
                LogitMargin(model, sign)(torch.from_numpy(batch)).numpy().astype(float)
                for batch in np.array_split(images, max(1, int(np.ceil(len(images) / 256))))
            ])
    return predict


def integrated_gradients(model, images, baseline, sign=1.0, steps=512, maximum_steps=2048):
    """Refine quadrature by a preset completeness-residual rule, not by ranking outcome."""
    score = LogitMargin(model, sign)
    inputs = torch.as_tensor(images, dtype=torch.float32)
    reference = torch.as_tensor(baseline, dtype=torch.float32)
    with torch.no_grad():
        difference = float((score(inputs) - score(reference)).item())
    while True:
        attribution, delta = IntegratedGradients(score).attribute(
            inputs, baselines=reference, n_steps=steps, method="gausslegendre",
            internal_batch_size=64, return_convergence_delta=True,
        )
        absolute = abs(float(delta.item()))
        relative = absolute / max(1.0, abs(difference))
        if relative <= 0.001 or steps >= maximum_steps:
            break
        steps = min(steps * 2, maximum_steps)
    return attribution.detach().numpy(), {
        "steps": steps, "score_difference": difference, "sum_attribution": float(attribution.detach().sum()),
        "absolute_completeness_residual": absolute, "normalized_residual": relative,
        "numerical_tolerance_met": relative <= 0.001,
    }


def gradcam(model, images, sign=1.0):
    # conv2 precedes ReLU in the frozen network; do not silently relabel this as input attribution.
    values = LayerGradCam(LogitMargin(model, sign), model.conv2).attribute(
        torch.as_tensor(images, dtype=torch.float32), relu_attributions=True,
    )
    return values.detach().numpy()


def kernel_shap(predict, row, baseline, seed, samples=512):
    np.random.seed(seed)
    explainer = shap.KernelExplainer(predict, np.asarray(baseline)[None], link="identity")
    values = np.asarray(explainer.shap_values(np.asarray(row)[None], nsamples=samples,
                                            l1_reg=0.0, silent=True)).reshape(-1)
    difference = float(predict(np.asarray(row)[None])[0] - predict(np.asarray(baseline)[None])[0])
    return values, {"expected_value": float(explainer.expected_value),
                    "additivity_residual": float(values.sum() - difference)}


def lime_effects(predict, row, baseline, training, seed, samples=2048):
    explainer = LimeTabularExplainer(
        training, mode="regression", feature_names=[str(i) for i in range(training.shape[1])],
        discretize_continuous=False, sample_around_instance=True, feature_selection="none",
        kernel_width=np.sqrt(training.shape[1]) * 0.75, random_state=seed,
    )
    explanation = explainer.explain_instance(row, predict, num_features=len(row), num_samples=samples)
    coefficients = np.zeros(len(row))
    for index, weight in explanation.local_exp[1]:
        coefficients[index] = weight
    # Standardized local slopes are not baseline-relative contributions by themselves.
    effects = coefficients * (row - baseline) / explainer.scaler.scale_
    return effects, {"standardized_local_coefficients": coefficients.tolist(),
                     "training_scale": explainer.scaler.scale_.tolist(),
                     "weighted_neighborhood_r2": float(explanation.score),
                     "local_prediction": float(np.asarray(explanation.local_pred).item()),
                     "original_score": float(predict(row[None])[0]),
                     "local_prediction_abs_error": abs(float(np.asarray(explanation.local_pred).item()) - float(predict(row[None])[0]))}


def agreement(first, second, k=5):
    first, second = np.abs(first), np.abs(second)
    if np.ptp(first) == 0 or np.ptp(second) == 0:
        return {"absolute_spearman": None, "top5_jaccard": None, "reason": "constant magnitude map"}
    a = set(np.argsort(-first, kind="stable")[:k])
    b = set(np.argsort(-second, kind="stable")[:k])
    return {"absolute_spearman": float(spearmanr(first, second).statistic),
            "top5_jaccard": len(a & b) / len(a | b), "reason": None}


def deletion_check(predict, row, baseline, scores, seed=5101):
    original = float(predict(row[None])[0])
    rng = np.random.default_rng(seed)
    result = {}
    for k in (1, 3, 5):
        random_orders = np.array([rng.choice(len(row), k, replace=False) for _ in range(100)])
        perturbed = np.repeat(row[None], 100, axis=0)
        for index, selected in enumerate(random_orders):
            perturbed[index, selected] = baseline[selected]
        random_drop = original - predict(perturbed)
        entry = {"random_mean_drop": float(random_drop.mean()),
                 "random_sd_drop": float(random_drop.std(ddof=1)), "methods": {}}
        for name, attribution in scores.items():
            if np.ptp(attribution) == 0:
                entry["methods"][name] = {"eligible": False, "reason": "constant map"}
                continue
            selected = np.argsort(-attribution, kind="stable")[:k]
            changed = row.copy()
            changed[selected] = baseline[selected]
            drop = original - float(predict(changed[None])[0])
            entry["methods"][name] = {"eligible": True, "selected_indices": selected.tolist(),
                                       "selected_features": [FEATURES[i] for i in selected],
                                       "score_drop": drop, "excess_over_random_mean": drop - float(random_drop.mean())}
        result[str(k)] = entry
    return result


def choose_cases(labels, predictions):
    primary = sorted([int(i) for label in (0, 1) for i in np.flatnonzero(labels == label)[:6]])
    errors = np.flatnonzero(labels != predictions)
    illustrative = [int(errors[0])] if len(errors) and int(errors[0]) not in primary else []
    return primary, primary + illustrative


def summarize(cases):
    primary = [case for case in cases if case["primary_summary"]]
    summary = {"primary_n": len(primary), "illustrative_n": len(cases) - len(primary)}
    summary["numerical_checks"] = {
        "maximum_ig_normalized_residual": max(case["ig_diagnostics"]["normalized_residual"] for case in cases),
        "ig_tolerance_failures": sum(not case["ig_diagnostics"]["numerical_tolerance_met"] for case in cases),
        "maximum_kernel_shap_additivity_residual": max(abs(case["shap_diagnostics"]["additivity_residual"]) for case in cases),
        "maximum_logistic_shap_exact_effect_error": max(case["logistic_regression"]["shap_max_error_vs_exact"] for case in cases),
    }
    summary["deletion_k5"] = {}
    for method in ("gradcam", "integrated_gradients", "kernel_shap", "lime_local_effect"):
        rows = [case["deletion"]["5"]["methods"][method] for case in primary]
        valid = [row for row in rows if row["eligible"]]
        summary["deletion_k5"][method] = {
            "eligible_n": len(valid),
            "mean_excess_score_drop_over_random": float(np.mean([x["excess_over_random_mean"] for x in valid])) if valid else None,
            "fraction_exceeding_random_mean": sum(x["excess_over_random_mean"] > 0 for x in valid) / len(valid) if valid else None,
        }
    summary["agreement"] = {}
    for key in primary[0]["agreement"]:
        rows = [case["agreement"][key] for case in primary]
        valid = [row for row in rows if row["absolute_spearman"] is not None]
        summary["agreement"][key] = {"eligible_n": len(valid), **{
            metric: float(np.mean([x[metric] for x in valid])) if valid else None
            for metric in ("absolute_spearman", "top5_jaccard")}}
    summary["lime_neighborhood_r2_mean"] = float(np.mean([c["lime_diagnostics"]["weighted_neighborhood_r2"] for c in primary]))
    return summary


def run_experiment(config):
    if sha256(MODEL_RESULTS) != config["model_results_sha256"]:
        raise ValueError("Frozen prediction results changed")
    previous = json.loads(MODEL_RESULTS.read_text())
    data, metadata, _ = load_inputs(previous["protocol"])
    artifact_dir = ROOT / "artifacts/02_wisconsin_models"
    saved = torch.load(artifact_dir / f"grid_cnn_{config['model_seed']}.pt", weights_only=True, map_location="cpu")
    seed_cpu(config["model_seed"])
    model = SmallGridCNN()
    model.load_state_dict(saved["state_dict"])
    model.eval()
    if state_digest(model) != config["model_state_sha256"]:
        raise ValueError("Frozen model weights changed")
    base_predict = predictor(model, metadata["mapping"])
    reference_run = next(run for run in previous["runs"] if run["model"] == "grid_cnn" and run["seed"] == config["model_seed"])
    val = data["validation"]
    predictions = (base_predict(val["tabular"]) > 0).astype(int)
    np.testing.assert_array_equal(predictions, reference_run["predictions"]["validation"]["predicted_label"])
    primary, selected = choose_cases(val["y"], predictions)
    baseline = np.median(data["train"]["tabular"], axis=0).astype(np.float32)
    baseline_image = encode(baseline[None], metadata["mapping"])[0].astype(np.float32)[:, None]
    logistic = joblib.load(artifact_dir / "logistic_regression_fixed.joblib")["model"]
    randomized = {}
    for name in ("head_reset", "all_reset"):
        seed_cpu(config["randomization"]["seed"])
        candidate = copy.deepcopy(model) if name == "head_reset" else SmallGridCNN()
        if name == "head_reset":
            candidate.classifier.reset_parameters()
        randomized[name] = candidate.eval()
    cases = []
    for position in selected:
        row, image = val["tabular"][position], val["images"][position:position + 1]
        sign = 1.0 if predictions[position] else -1.0
        predict = predictor(model, metadata["mapping"], sign=sign)
        ig_image, ig_check = integrated_gradients(model, image, baseline_image, sign)
        ig = decode(ig_image[:, 0], metadata["mapping"])[0]
        zero_image, zero_check = integrated_gradients(model, image, np.zeros_like(baseline_image), sign)
        zero_ig = decode(zero_image[:, 0], metadata["mapping"])[0]
        cam = decode(gradcam(model, image, sign)[:, 0], metadata["mapping"])[0]
        sv, shap_check = kernel_shap(predict, row, baseline, 3101 + position)
        sv_repeat, _ = kernel_shap(predict, row, baseline, 3102 + position)
        lv, lime_check = lime_effects(predict, row, baseline, data["train"]["tabular"], 4101 + position)
        lv_repeat, _ = lime_effects(predict, row, baseline, data["train"]["tabular"], 4102 + position)
        scores = {"gradcam": cam, "integrated_gradients": ig, "kernel_shap": sv, "lime_local_effect": lv}
        comparisons = {
            "ig_vs_shap_same_cnn": agreement(ig, sv), "ig_vs_gradcam_spatial_proxy": agreement(ig, cam),
            "ig_vs_lime_local_effect": agreement(ig, lv), "ig_median_vs_zero_reference": agreement(ig, zero_ig),
            "shap_two_sampling_seeds": agreement(sv, sv_repeat), "lime_two_sampling_seeds": agreement(lv, lv_repeat),
        }
        randomized_details = {}
        for name, candidate in randomized.items():
            new_ig_image, check = integrated_gradients(candidate, image, baseline_image, sign)
            new_ig = decode(new_ig_image[:, 0], metadata["mapping"])[0]
            new_cam = decode(gradcam(candidate, image, sign)[:, 0], metadata["mapping"])[0]
            comparisons[f"ig_vs_{name}"] = agreement(ig, new_ig)
            comparisons[f"gradcam_vs_{name}"] = agreement(cam, new_cam)
            randomized_details[name] = {"state_sha256": state_digest(candidate), "ig": new_ig.tolist(),
                                        "gradcam": new_cam.tolist(), "ig_diagnostics": check,
                                        "fixed_target_score": float(predictor(candidate, metadata["mapping"], sign=sign)(row[None])[0])}
        def lr_predict(values, target_sign=sign):
            return target_sign * logistic.decision_function(values)
        exact = sign * logistic.coef_[0] * (row.astype(float) - baseline.astype(float))
        lr_shap, lr_check = kernel_shap(lr_predict, row, baseline, 3101 + position)
        lr_lime, lr_lime_check = lime_effects(lr_predict, row, baseline, data["train"]["tabular"], 4101 + position)
        comparisons["cnn_ig_vs_logistic_exact_different_models"] = agreement(ig, exact)
        case = {
            "validation_position": position, "sample_id": val["sample_ids"][position],
            "primary_summary": position in primary, "true_label": int(val["y"][position]),
            "original_predicted_label": int(predictions[position]), "target_sign": sign,
            "original_target_margin": float(predict(row[None])[0]), "scaled_features": row.tolist(),
            "attributions": {name: values.tolist() for name, values in scores.items()},
            "ig_zero_reference": zero_ig.tolist(), "shap_repeat": sv_repeat.tolist(), "lime_repeat": lv_repeat.tolist(),
            "ig_diagnostics": ig_check, "ig_zero_diagnostics": zero_check,
            "shap_diagnostics": shap_check, "lime_diagnostics": lime_check,
            "deletion": deletion_check(predict, row, baseline, scores, 5101 + position),
            "agreement": comparisons, "randomization": randomized_details,
            "logistic_regression": {"exact_median_relative_effect": exact.tolist(), "kernel_shap": lr_shap.tolist(),
                                    "lime_local_effect": lr_lime.tolist(), "shap_diagnostics": lr_check,
                                    "lime_diagnostics": lr_lime_check,
                                    "shap_max_error_vs_exact": float(np.max(np.abs(lr_shap - exact))),
                                    "lime_max_error_vs_exact": float(np.max(np.abs(lr_lime - exact)))},
        }
        cases.append(case)
        print(f"Explained validation position {position}; IG residual {ig_check['normalized_residual']:.3g}", flush=True)
    if state_digest(model) != config["model_state_sha256"]:
        raise AssertionError("Attribution changed the frozen model")
    return {"experiment": config["experiment"], "protocol": config, "protocol_sha256": sha256(PROTOCOL),
            "source_code_sha256": sha256(__file__), "environment": {p: version(p) for p in
            ("torch", "captum", "shap", "lime", "numpy", "scikit-learn")},
            "features": list(FEATURES), "mapping": metadata["mapping"], "training_median_scaled": baseline.tolist(),
            "primary_validation_positions": primary, "cases": cases, "summary": summarize(cases)}


def make_figures(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Purpose-selected: first primary M, first primary B, and illustrative error if available.
    primary = [c for c in report["cases"] if c["primary_summary"]]
    chosen = [next(c for c in primary if c["true_label"] == label) for label in (1, 0)]
    errors = [c for c in report["cases"] if c["true_label"] != c["original_predicted_label"]]
    if errors:
        chosen.append(errors[0])
    columns = [("gradcam", "Grad-CAM spatial proxy"), ("integrated_gradients", "Integrated Gradients"),
               ("kernel_shap", "Kernel SHAP"), ("lime_local_effect", "LIME local effect")]
    fig, axes = plt.subplots(len(chosen), 4, figsize=(15, 3.4 * len(chosen)), layout="constrained", squeeze=False)
    for row_index, case in enumerate(chosen):
        for col, (method, label) in enumerate(columns):
            axis = axes[row_index, col]
            values = np.array(case["attributions"][method]).reshape(5, 6)
            maximum = max(float(np.max(np.abs(values))), 1e-12)
            display = values / maximum
            axis.imshow(display, cmap="RdBu_r", vmin=-1, vmax=1)
            for i in range(30):
                axis.text(i % 6, i // 6, str(i), ha="center", va="center", fontsize=8,
                          color="white" if abs(display.flat[i]) > 0.65 else "black")
            axis.set(xticks=[], yticks=[], title=label)
            if col == 0:
                axis.set_ylabel(f"ID {case['sample_id']}\nTrue {'M' if case['true_label'] else 'B'} / predicted {'M' if case['original_predicted_label'] else 'B'}")
    fig.suptitle("Same frozen CNN and target; each map normalized separately for display\nRed supports original predicted class; blue opposes. Numbers are feature indices, not ranks.", fontsize=13)
    fig.savefig(output / "explanation_examples.png", dpi=150)
    plt.close(fig)
    methods = list(report["summary"]["deletion_k5"])
    means = [report["summary"]["deletion_k5"][m]["mean_excess_score_drop_over_random"] for m in methods]
    fig, axis = plt.subplots(figsize=(9, 4.5), layout="constrained")
    axis.bar(["Grad-CAM\nspatial proxy", "Integrated\nGradients", "Kernel SHAP", "LIME\nlocal effect"], means,
             color=["#777777", "#176b91", "#347849", "#ad681d"])
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(ylabel="Mean margin drop minus random mean", title="Five-feature replacement diagnostic\n12 purposive validation examples; training-median replacement; not causal evidence")
    fig.savefig(output / "deletion_diagnostic.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROTOCOL.parent)
    args = parser.parse_args()
    if (args.output_dir / "results.json").exists():
        parser.error("Use a fresh output directory; existing results will not be overwritten")
    report = run_experiment(json.loads(PROTOCOL.read_text()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    make_figures(report, args.output_dir)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
