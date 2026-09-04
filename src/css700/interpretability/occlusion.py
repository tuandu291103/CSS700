"""Feature-wise Occlusion Sensitivity with explicit perturbation masks."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class OcclusionSensitivityResult:
    """Signed target-score drops for one masked feature at a time."""

    logit_drops: torch.Tensor
    probability_drops: torch.Tensor
    occluded_target_logits: torch.Tensor
    occluded_target_probabilities: torch.Tensor
    original_logits: torch.Tensor
    original_probabilities: torch.Tensor
    target_classes: torch.Tensor
    batch_forward_calls: int


def occlusion_sensitivity(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    feature_masks: torch.Tensor,
    replacement_values: torch.Tensor,
    target_classes: torch.Tensor | None = None,
    perturbation_batch_size: int = 512,
) -> OcclusionSensitivityResult:
    """Mask each complete feature region and measure its target-class score drop.

    ``feature_masks`` has shape ``[features, channels, rows, columns]``. Each mask may
    contain one pixel (the current Grid contract) or a complete multi-pixel region for a
    future encoder. Positive drops indicate support for the target; negative drops indicate
    that replacing the feature increased the target score.
    """

    if inputs.ndim != 4:
        raise ValueError("inputs must have shape [samples, channels, rows, columns]")
    if feature_masks.ndim != 4 or feature_masks.shape[1:] != inputs.shape[1:]:
        raise ValueError("feature_masks must have shape [features, channels, rows, columns]")
    if feature_masks.shape[0] == 0:
        raise ValueError("feature_masks must contain at least one feature")
    if perturbation_batch_size <= 0:
        raise ValueError("perturbation_batch_size must be positive")

    masks = feature_masks.to(device=inputs.device, dtype=torch.bool)
    if not torch.all(masks.flatten(start_dim=1).any(dim=1)):
        raise ValueError("every feature mask must select at least one input location")

    replacement = replacement_values.to(device=inputs.device, dtype=inputs.dtype)
    if replacement.shape[0] == 1 and inputs.shape[0] != 1:
        replacement = replacement.expand_as(inputs)
    if replacement.shape != inputs.shape:
        raise ValueError("replacement_values must match inputs or contain one broadcastable sample")

    model.eval()
    with torch.no_grad():
        original_logits = model(inputs)
        original_probabilities = torch.softmax(original_logits, dim=1)
    if target_classes is None:
        target_classes = original_logits.argmax(dim=1)
    target_classes = target_classes.to(device=inputs.device, dtype=torch.long)
    if target_classes.shape != (inputs.shape[0],):
        raise ValueError("target_classes must contain one class index per input sample")
    if torch.any(target_classes < 0) or torch.any(target_classes >= original_logits.shape[1]):
        raise ValueError("target_classes contains an invalid class index")

    sample_count = inputs.shape[0]
    feature_count = masks.shape[0]
    total_perturbations = sample_count * feature_count
    occluded_logits = torch.empty(
        (sample_count, feature_count), device=inputs.device, dtype=original_logits.dtype
    )
    occluded_probabilities = torch.empty_like(occluded_logits)
    flat_occluded_logits = occluded_logits.view(-1)
    flat_occluded_probabilities = occluded_probabilities.view(-1)

    batch_forward_calls = 1
    with torch.no_grad():
        for start in range(0, total_perturbations, perturbation_batch_size):
            stop = min(start + perturbation_batch_size, total_perturbations)
            flat_positions = torch.arange(start, stop, device=inputs.device)
            sample_positions = torch.div(flat_positions, feature_count, rounding_mode="floor")
            feature_positions = flat_positions.remainder(feature_count)
            selected_inputs = inputs.index_select(0, sample_positions)
            selected_replacements = replacement.index_select(0, sample_positions)
            selected_masks = masks.index_select(0, feature_positions)
            perturbed = torch.where(selected_masks, selected_replacements, selected_inputs)
            perturbed_logits = model(perturbed)
            perturbed_probabilities = torch.softmax(perturbed_logits, dim=1)
            selected_targets = target_classes.index_select(0, sample_positions)
            flat_occluded_logits[start:stop] = perturbed_logits.gather(
                1, selected_targets[:, None]
            ).squeeze(1)
            flat_occluded_probabilities[start:stop] = perturbed_probabilities.gather(
                1, selected_targets[:, None]
            ).squeeze(1)
            batch_forward_calls += 1

    original_target_logits = original_logits.gather(1, target_classes[:, None])
    original_target_probabilities = original_probabilities.gather(1, target_classes[:, None])
    return OcclusionSensitivityResult(
        logit_drops=(original_target_logits - occluded_logits).detach(),
        probability_drops=(original_target_probabilities - occluded_probabilities).detach(),
        occluded_target_logits=occluded_logits.detach(),
        occluded_target_probabilities=occluded_probabilities.detach(),
        original_logits=original_logits.detach(),
        original_probabilities=original_probabilities.detach(),
        target_classes=target_classes.detach(),
        batch_forward_calls=batch_forward_calls,
    )
