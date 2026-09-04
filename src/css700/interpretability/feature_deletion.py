"""Cumulative feature-deletion evaluation using explicit ranked feature masks."""

from dataclasses import dataclass
from math import ceil

import torch
from torch import nn


@dataclass(frozen=True)
class FeatureDeletionResult:
    """Model responses after cumulative deletion along one or more rankings."""

    target_logit_drops: torch.Tensor
    target_probability_drops: torch.Tensor
    predictions: torch.Tensor
    target_logits: torch.Tensor
    target_probabilities: torch.Tensor
    original_logits: torch.Tensor
    original_probabilities: torch.Tensor
    target_classes: torch.Tensor
    deleted_feature_counts: torch.Tensor
    batch_forward_calls: int
    conceptual_model_evaluations: int


def _batched_logits(
    model: nn.Module, inputs: torch.Tensor, *, batch_size: int
) -> tuple[torch.Tensor, int]:
    parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            parts.append(model(inputs[start : start + batch_size]))
    return torch.cat(parts), ceil(len(inputs) / batch_size)


def cumulative_feature_deletion(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    rankings: torch.Tensor,
    feature_masks: torch.Tensor,
    replacement_values: torch.Tensor,
    k_values: list[int] | tuple[int, ...],
    target_classes: torch.Tensor | None = None,
    inference_batch_size: int = 512,
) -> FeatureDeletionResult:
    """Evaluate cumulative deletion for ranked features.

    Rankings may have shape ``[samples, features]`` or
    ``[ranking_repeats, samples, features]``. Every row must be a complete permutation of the
    feature indices. Results use shape ``[ranking_repeats, deletion_steps, samples]``.
    """

    if inputs.ndim != 4:
        raise ValueError("inputs must have shape [samples, channels, rows, columns]")
    if feature_masks.ndim != 4 or feature_masks.shape[1:] != inputs.shape[1:]:
        raise ValueError("feature_masks must have shape [features, channels, rows, columns]")
    if feature_masks.shape[0] == 0:
        raise ValueError("feature_masks must contain at least one feature")
    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be positive")

    masks = feature_masks.to(device=inputs.device, dtype=torch.bool)
    if not torch.all(masks.flatten(start_dim=1).any(dim=1)):
        raise ValueError("every feature mask must select at least one input location")

    ranking_tensor = rankings.to(device=inputs.device, dtype=torch.long)
    if ranking_tensor.ndim == 2:
        ranking_tensor = ranking_tensor.unsqueeze(0)
    if ranking_tensor.ndim != 3:
        raise ValueError(
            "rankings must have shape [samples, features] or [repeats, samples, features]"
        )
    repeat_count, sample_count, feature_count = ranking_tensor.shape
    if sample_count != inputs.shape[0] or feature_count != masks.shape[0]:
        raise ValueError("rankings must cover every input sample and feature")
    expected = torch.arange(feature_count, device=inputs.device).expand_as(ranking_tensor)
    if not torch.equal(torch.sort(ranking_tensor, dim=-1).values, expected):
        raise ValueError("every ranking row must be a permutation of all feature indices")

    steps = tuple(int(k) for k in k_values)
    if not steps or steps[0] != 0 or tuple(sorted(set(steps))) != steps:
        raise ValueError("k_values must be unique, increasing, and begin with zero")
    if steps[-1] > feature_count:
        raise ValueError("k_values cannot exceed the number of features")

    replacement = replacement_values.to(device=inputs.device, dtype=inputs.dtype)
    if replacement.shape[0] == 1 and sample_count != 1:
        replacement = replacement.expand_as(inputs)
    if replacement.shape != inputs.shape:
        raise ValueError("replacement_values must match inputs or contain one broadcastable sample")

    model.eval()
    original_logits, batch_forward_calls = _batched_logits(
        model, inputs, batch_size=inference_batch_size
    )
    original_probabilities = torch.softmax(original_logits, dim=1)
    if target_classes is None:
        target_classes = original_logits.argmax(dim=1)
    targets = target_classes.to(device=inputs.device, dtype=torch.long)
    if targets.shape != (sample_count,):
        raise ValueError("target_classes must contain one class index per input sample")
    if torch.any(targets < 0) or torch.any(targets >= original_logits.shape[1]):
        raise ValueError("target_classes contains an invalid class index")

    step_count = len(steps)
    class_count = original_logits.shape[1]
    logits = torch.empty(
        (repeat_count, step_count, sample_count, class_count),
        device=inputs.device,
        dtype=original_logits.dtype,
    )
    logits[:, 0] = original_logits.unsqueeze(0)
    deleted_counts = torch.empty(
        (repeat_count, step_count, sample_count), device=inputs.device, dtype=torch.long
    )
    deleted_counts[:, 0] = 0
    conceptual_model_evaluations = sample_count

    source = inputs.unsqueeze(0).expand(repeat_count, *inputs.shape)
    replacements = replacement.unsqueeze(0).expand_as(source)
    for step_position, k in enumerate(steps[1:], start=1):
        selected_masks = masks[ranking_tensor[:, :, :k]].any(dim=2)
        perturbed = torch.where(selected_masks, replacements, source)
        flat_logits, calls = _batched_logits(
            model,
            perturbed.reshape(repeat_count * sample_count, *inputs.shape[1:]),
            batch_size=inference_batch_size,
        )
        logits[:, step_position] = flat_logits.reshape(repeat_count, sample_count, class_count)
        deleted_counts[:, step_position] = k
        batch_forward_calls += calls
        conceptual_model_evaluations += repeat_count * sample_count

    probabilities = torch.softmax(logits, dim=-1)
    expanded_targets = targets.view(1, 1, sample_count, 1).expand(
        repeat_count, step_count, sample_count, 1
    )
    target_logits = logits.gather(-1, expanded_targets).squeeze(-1)
    target_probabilities = probabilities.gather(-1, expanded_targets).squeeze(-1)
    original_target_logits = original_logits.gather(1, targets[:, None]).squeeze(1)
    original_target_probabilities = original_probabilities.gather(1, targets[:, None]).squeeze(1)
    return FeatureDeletionResult(
        target_logit_drops=(original_target_logits.view(1, 1, -1) - target_logits).detach(),
        target_probability_drops=(
            original_target_probabilities.view(1, 1, -1) - target_probabilities
        ).detach(),
        predictions=logits.argmax(dim=-1).detach(),
        target_logits=target_logits.detach(),
        target_probabilities=target_probabilities.detach(),
        original_logits=original_logits.detach(),
        original_probabilities=original_probabilities.detach(),
        target_classes=targets.detach(),
        deleted_feature_counts=deleted_counts.detach(),
        batch_forward_calls=batch_forward_calls,
        conceptual_model_evaluations=conceptual_model_evaluations,
    )
