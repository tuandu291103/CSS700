"""Feature-only deletion with the accepted CPU inference path and fixed row batches."""

import time

import numpy as np
import torch

from css700.interpretability.activation_boundary import bitwise_equal
from css700.models import SmallTabularImageCNN


def deletion_rankings(scores, groups, repeats=100, seed=20261004):
    if scores.ndim != 2 or len(groups) != len(scores) or repeats < 1:
        raise ValueError("scores/groups/repeat dimensions invalid")
    if not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError("finite nonnegative Grad-CAM scores required")
    _, codes = np.unique(groups, return_inverse=True)
    random = np.argsort(
        np.random.default_rng(seed).random((repeats, len(np.unique(codes)), scores.shape[1])),
        axis=-1,
        kind="stable",
    )[:, codes]
    return np.concatenate(
        [
            np.argsort(-scores, axis=1, kind="stable")[None],
            np.argsort(scores, axis=1, kind="stable")[None],
            random,
        ]
    ).astype(np.int16)


def replace_ranked(images, order, coordinates, reference, k):
    if not 0 <= k <= len(coordinates):
        raise ValueError("deletion count outside feature range")
    result = images.copy()
    locations = coordinates[order[:, :k]]
    result.reshape(len(images), -1)[np.arange(len(images))[:, None], locations] = reference.reshape(
        -1
    )[locations]
    return result


def evaluate_deletions(
    source,
    rankings,
    coordinates,
    references,
    steps,
    *,
    partial_path=None,
    batch_size=32,
    max_seconds=300.0,
):
    images = source.arrays["validation_images"]
    n, d = len(images), len(coordinates)
    if batch_size <= 0 or max_seconds <= 0:
        raise ValueError("positive inference budget required")
    if (
        rankings.ndim != 3
        or rankings.shape[1:] != (n, d)
        or not np.all(np.sort(rankings, axis=-1) == np.arange(d))
    ):
        raise ValueError("every ranking must be a full feature permutation")
    if list(steps) != sorted(set(steps)) or steps[0] != 0 or steps[-1] != d:
        raise ValueError("steps must increase from zero to all features")
    if len(set(coordinates.tolist())) != d or not np.array_equal(
        np.sort(coordinates), np.flatnonzero(source.arrays["occupied_mask"])
    ):
        raise ValueError("coordinates must cover mapped features only")
    if (
        references.shape != (2, 1, 8, 8)
        or references.dtype != np.float32
        or not np.isfinite(references).all()
    ):
        raise ValueError("two finite float32 reference images required")
    if np.any(references[:, ~source.arrays["occupied_mask"]] != 0):
        raise ValueError("reference padding must stay zero")
    shape = (2, len(rankings), len(steps), n, 2)
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
                        raise TimeoutError("fixed deletion inference budget exceeded")
                    parts.append(
                        model(torch.from_numpy(values[start : start + batch_size].copy())).numpy()
                    )
                    calls += 1
                return np.concatenate(parts)

            original = predict(images)
            if not bitwise_equal(original, source.predictions["logits"]):
                raise ValueError("exact original inference replay failed before deletion")
            logits[:, :, 0] = original
            padding_unchanged = True
            for r, reference in enumerate(references):
                for repeat, order in enumerate(rankings):
                    for j, k in enumerate(steps[1:], 1):
                        values = replace_ranked(images, order, coordinates, reference, k)
                        padding_unchanged &= np.array_equal(
                            values[:, ~source.arrays["occupied_mask"]],
                            images[:, ~source.arrays["occupied_mask"]],
                        )
                        logits[r, repeat, j] = predict(values)
            if time.perf_counter() - started > max_seconds:
                raise TimeoutError("fixed deletion inference budget exceeded")
            checks = {
                "exact_original_inference": True,
                "complete_feature_permutations": True,
                "padding_unchanged": bool(padding_unchanged),
                "all_logits_finite": bool(np.isfinite(logits).all()),
                "zero_deletion_exact": bool(np.all(logits[:, :, 0] == original)),
                "full_deletion_order_independent": all(
                    bitwise_equal(logits[r, i, -1], logits[r, 0, -1])
                    for r in range(2)
                    for i in range(len(rankings))
                ),
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
                    "conceptual_model_evaluations": n * (1 + 2 * len(rankings) * (len(steps) - 1)),
                    "inference_seconds": time.perf_counter() - started,
                },
            )
    finally:
        if isinstance(logits, np.memmap):
            logits.flush()
        torch.set_num_threads(old[0])
        torch.use_deterministic_algorithms(old[1], warn_only=old[2])
