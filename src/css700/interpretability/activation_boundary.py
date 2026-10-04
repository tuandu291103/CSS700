"""Inference-preserving Grad-CAM for the fixed sequential CNN, not input gradients."""

import time

import numpy as np
import torch

from css700.encoding.grid import FeaturePixel
from css700.interpretability.grad_cam import GradCAMResult
from css700.interpretability.tracing import trace_attributions_to_features
from css700.models import SmallTabularImageCNN

MODES = ("nnpack_on_no_grad", "nnpack_on_grad", "nnpack_off_no_grad", "nnpack_off_grad")
STAGES = ("conv1", "relu1", "conv2_pre_relu", "relu2", "pooled", "classifier")


def boundary_grad_cam(model, inputs, targets):
    """Differentiate only the unchanged head from its inference-path pre-ReLU activation."""
    if type(model) is not SmallTabularImageCNN or model.training:
        raise ValueError("boundary Grad-CAM requires the fixed CNN in evaluation mode")
    if inputs.ndim != 4 or inputs.shape[1:] != (1, 8, 8):
        raise ValueError("expected [rows, 1, 8, 8] inputs")
    if targets.shape != (len(inputs),) or targets.dtype != torch.long:
        raise ValueError("one integer target per row required")
    if bool(((targets < 0) | (targets >= model.classifier.out_features)).any()):
        raise ValueError("target outside classifier range")
    with torch.no_grad():
        activation = model.features[:3](inputs)
    with torch.enable_grad():
        leaf = activation.detach().requires_grad_(True)
        logits = model.classifier(model.pool(model.features[3](leaf)).flatten(1))
        gradient = torch.autograd.grad(logits.gather(1, targets[:, None]).sum(), leaf)[0]
        raw = (gradient.mean((2, 3), keepdim=True) * leaf).sum(1, keepdim=True).relu()
        normalized = raw / raw.amax((2, 3), keepdim=True).clamp_min(1e-12)
    return GradCAMResult(raw.detach(), normalized.detach(), targets.detach(), logits.detach())


def _stages(model, inputs):
    values = []
    value = inputs
    for layer in model.features:
        value = layer(value)
        values.append(value)
    values.append(model.pool(value).flatten(1))
    values.append(model.classifier(values[-1]))
    return [v.detach().numpy() for v in values]


def bitwise_equal(left, right):
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes() == right.tobytes()
    )


def logit_comparison(left, right):
    """Keep v1 tolerances; comparisons do not select a replacement computation path."""
    failed = ~np.isclose(left, right, rtol=1e-5, atol=1e-6)
    return {
        "exact": bitwise_equal(left, right),
        "max_abs_difference": float(np.max(np.abs(left - right))),
        "failed_values": int(failed.sum()),
        "affected_rows": int(failed.any(axis=1).sum()),
        "predictions_equal": bool(np.array_equal(left.argmax(1), right.argmax(1))),
    }


def explain_boundary_validation(source, previous, *, batch_size=32, max_seconds=120.0):
    """Four fixed diagnostic forwards and one fixed candidate; no retry or model selection."""
    if batch_size <= 0 or max_seconds <= 0:
        raise ValueError("positive explanation budget required")
    old_threads = torch.get_num_threads()
    old_determinism = torch.are_deterministic_algorithms_enabled()
    old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    started = time.perf_counter()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            model = SmallTabularImageCNN().cpu().eval()
            model.load_state_dict(source.checkpoint["model_state_dict"])
            before = {k: v.clone() for k, v in model.state_dict().items()}
            images, p = source.arrays["validation_images"], source.predictions
            parts = {k: [] for k in ("raw_maps", "normalized_maps", "target_classes", "logits")}
            diagnostic = {k: [] for k in MODES}
            stage_max = dict.fromkeys(STAGES, 0.0)
            for start in range(0, len(images), batch_size):
                if time.perf_counter() - started > max_seconds:
                    raise TimeoutError("fixed v2 numerical/Grad-CAM budget exceeded")
                inputs = torch.from_numpy(images[start : start + batch_size].copy())
                targets = torch.from_numpy(p["predictions"][start : start + batch_size].copy())
                reference_stages = None
                for mode, enabled, gradients in zip(
                    MODES, (True, True, False, False), (False, True, False, True), strict=True
                ):
                    with torch.backends.nnpack.flags(enabled=enabled):
                        with torch.set_grad_enabled(gradients):
                            stages = _stages(model, inputs)
                    diagnostic[mode].append(stages[-1])
                    if mode == MODES[0]:
                        reference_stages = stages
                    elif mode == MODES[1]:
                        for name, a, b in zip(STAGES, reference_stages, stages, strict=True):
                            stage_max[name] = max(stage_max[name], float(np.max(np.abs(a - b))))
                with torch.backends.nnpack.flags(enabled=True):
                    result = boundary_grad_cam(model, inputs, targets)
                for name, value in zip(
                    parts,
                    (result.raw, result.normalized, result.target_classes, result.logits),
                    strict=True,
                ):
                    parts[name].append(value.numpy())
            if time.perf_counter() - started > max_seconds:
                raise TimeoutError("fixed v2 numerical/Grad-CAM budget exceeded")
            outputs = {k: np.concatenate(v) for k, v in parts.items()}
            diagnostic = {k: np.concatenate(v) for k, v in diagnostic.items()}
            outputs["inference_logits"] = diagnostic[MODES[0]]
            mapping = [FeaturePixel(**r) for r in source.feature_map]
            for input_key, output_key in (
                ("raw_maps", "feature_scores"),
                ("normalized_maps", "normalized_feature_scores"),
            ):
                outputs[output_key] = trace_attributions_to_features(outputs[input_key], mapping)
            probe = outputs["raw_maps"].copy()
            probe[:, ~source.arrays["occupied_mask"]] = 1e6
            checks = {
                "frozen_inference_logits_exact": bitwise_equal(
                    outputs["inference_logits"], p["logits"]
                ),
                "held_v1_gradient_logits_replayed_exactly": bitwise_equal(
                    diagnostic[MODES[1]], previous["logits"]
                ),
                "candidate_logits_exact_frozen_inference": bitwise_equal(
                    outputs["logits"], p["logits"]
                ),
                "original_gradient_logit_tolerance_passed": np.allclose(
                    outputs["logits"], p["logits"], rtol=1e-5, atol=1e-6
                ),
                "predicted_classes_exact": np.array_equal(
                    outputs["logits"].argmax(1), p["predictions"]
                ),
                "frozen_targets_exact": np.array_equal(outputs["target_classes"], p["predictions"]),
                "all_outputs_finite": all(
                    np.isfinite(v).all() for v in [*outputs.values(), *diagnostic.values()]
                ),
                "raw_maps_input_sized": outputs["raw_maps"].shape == images.shape,
                "nonnegative_raw_maps": np.all(outputs["raw_maps"] >= 0),
                "display_maps_bounded": np.all(
                    (outputs["normalized_maps"] >= 0) & (outputs["normalized_maps"] <= 1)
                ),
                "all_57_features_traced": outputs["feature_scores"].shape == (len(images), 57),
                "feature_values_exact": np.array_equal(
                    outputs["feature_scores"],
                    outputs["raw_maps"][:, source.arrays["occupied_mask"]],
                ),
                "padding_excluded_with_reversed_mapping": np.array_equal(
                    trace_attributions_to_features(probe, list(reversed(mapping))),
                    outputs["feature_scores"],
                ),
                "weights_unchanged": all(
                    torch.equal(before[k], v) for k, v in model.state_dict().items()
                ),
                "no_parameter_gradients": all(p.grad is None for p in model.parameters()),
                "no_forward_hooks": all(not layer._forward_hooks for layer in model.modules()),
            }
            comparisons = {
                "on_grad_vs_on_no_grad": logit_comparison(
                    diagnostic[MODES[1]], diagnostic[MODES[0]]
                ),
                "off_grad_vs_off_no_grad": logit_comparison(
                    diagnostic[MODES[3]], diagnostic[MODES[2]]
                ),
                "off_no_grad_vs_on_grad": logit_comparison(
                    diagnostic[MODES[2]], diagnostic[MODES[1]]
                ),
                "off_grad_vs_on_grad": logit_comparison(diagnostic[MODES[3]], diagnostic[MODES[1]]),
                "candidate_vs_frozen_inference": logit_comparison(outputs["logits"], p["logits"]),
            }
            old_top = np.argsort(-previous["feature_scores"], axis=1, kind="stable")[:, :5]
            new_top = np.argsort(-outputs["feature_scores"], axis=1, kind="stable")[:, :5]
            changed = 0
            for j in range(len(images)):
                old = {int(k) for k in old_top[j] if previous["feature_scores"][j, k] > 0}
                new = {int(k) for k in new_top[j] if outputs["feature_scores"][j, k] > 0}
                changed += old != new
            numerical = {
                "nnpack_available": bool(torch.backends.nnpack.is_available()),
                "comparisons": comparisons,
                "nnpack_on_stage_max_abs_no_grad_vs_grad": stage_max,
                "v2_vs_held_v1_raw_map_max_abs_difference": float(
                    np.max(np.abs(outputs["raw_maps"] - previous["raw_maps"]))
                ),
                "v2_vs_held_v1_positive_top5_membership_changed_rows": changed,
            }
            return outputs, diagnostic, {k: bool(v) for k, v in checks.items()}, numerical
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_determinism, warn_only=old_warn)
