"""Single-feature finite effects through the accepted, fixed-batch inference path."""

import time

import numpy as np
import torch

from css700.evaluation.grouped_deletion import softmax64
from css700.interpretability.activation_boundary import bitwise_equal
from css700.models import SmallTabularImageCNN


def replace_one(images, coordinate, reference):
    values = images.copy()
    values.reshape(len(images), -1)[:, coordinate] = reference.reshape(-1)[coordinate]
    return values


def signed_effects(logits, original, targets):
    index = targets[None, :, None, None]
    base = original[np.arange(len(targets)), targets].astype(np.float64)
    base_prob = softmax64(original)[np.arange(len(targets)), targets]
    return {
        "logit": base[None, :, None]
        - np.take_along_axis(logits.astype(np.float64), index, axis=-1)[..., 0],
        "probability": base_prob[None, :, None]
        - np.take_along_axis(softmax64(logits), index, axis=-1)[..., 0],
    }


def evaluate_occlusion(
    source, coordinates, references, *, partial_path=None, batch_size=32, max_seconds=120.0
):
    images = source.arrays["validation_images"]
    n, d = len(images), len(coordinates)
    if batch_size <= 0 or max_seconds <= 0:
        raise ValueError("positive Occlusion inference budget required")
    if coordinates.dtype.kind not in "iu" or len(set(coordinates.tolist())) != d:
        raise ValueError("unique integer feature coordinates required")
    if not np.array_equal(np.sort(coordinates), np.flatnonzero(source.arrays["occupied_mask"])):
        raise ValueError("coordinates must cover mapped features only")
    if (
        references.shape != (2, 1, 8, 8)
        or references.dtype != np.float32
        or not np.isfinite(references).all()
        or np.any(references[:, ~source.arrays["occupied_mask"]] != 0)
    ):
        raise ValueError("two finite float32 references with zero padding required")
    shape = (2, n, d, 2)
    logits = (
        np.lib.format.open_memmap(partial_path, mode="w+", dtype=np.float32, shape=shape)
        if partial_path
        else np.empty(shape, np.float32)
    )
    logits[:] = np.nan
    old = (
        torch.get_num_threads(),
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    started, calls = time.perf_counter(), 0
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with (
            torch.random.fork_rng(devices=[]),
            torch.backends.nnpack.flags(enabled=True),
            torch.no_grad(),
        ):
            model = SmallTabularImageCNN().cpu().eval()
            model.load_state_dict(source.checkpoint["model_state_dict"])
            before = {k: v.clone() for k, v in model.state_dict().items()}

            def predict(values):
                nonlocal calls
                parts = []
                for start in range(0, n, batch_size):
                    if time.perf_counter() - started > max_seconds:
                        raise TimeoutError("fixed Occlusion inference budget exceeded")
                    parts.append(
                        model(torch.from_numpy(values[start : start + batch_size].copy())).numpy()
                    )
                    calls += 1
                return np.concatenate(parts)

            original = predict(images)
            if not bitwise_equal(original, source.predictions["logits"]):
                raise ValueError("exact original inference replay failed before Occlusion")
            masks_exact, unmodified_exact = True, True
            for r, ref in enumerate(references):
                for j, coordinate in enumerate(coordinates):
                    values = replace_one(images, coordinate, ref)
                    mask = np.zeros(64, bool)
                    mask[coordinate] = True
                    flat = values.reshape(n, 64)
                    masks_exact &= bitwise_equal(flat[:, ~mask], images.reshape(n, 64)[:, ~mask])
                    masks_exact &= np.all(flat[:, coordinate] == ref.flat[coordinate])
                    logits[r, :, j] = predict(values)
                    unchanged = images.reshape(n, 64)[:, coordinate] == ref.flat[coordinate]
                    unmodified_exact &= bitwise_equal(logits[r, unchanged, j], original[unchanged])
            if time.perf_counter() - started > max_seconds:
                raise TimeoutError("fixed Occlusion inference budget exceeded")
            checks = {
                "exact_original_inference": True,
                "exactly_one_feature_and_other_pixels_unchanged": bool(masks_exact),
                "unchanged_input_rows_exact_logits": bool(unmodified_exact),
                "all_logits_finite": bool(np.isfinite(logits).all()),
                "weights_unchanged": all(
                    torch.equal(before[k], v) for k, v in model.state_dict().items()
                ),
                "no_parameter_gradients": all(p.grad is None for p in model.parameters()),
            }
            return (
                logits,
                original,
                checks,
                {
                    "forward_calls": calls,
                    "conceptual_model_evaluations": n * (1 + 2 * d),
                    "inference_seconds": time.perf_counter() - started,
                },
            )
    finally:
        if isinstance(logits, np.memmap):
            logits.flush()
        torch.set_num_threads(old[0])
        torch.use_deterministic_algorithms(old[1], warn_only=old[2])
