"""Input-level Integrated Gradients with explicit completeness diagnostics."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class IntegratedGradientsResult:
    """Signed input attributions and per-sample completeness measurements."""

    attributions: torch.Tensor
    convergence_delta: torch.Tensor
    relative_convergence_delta: torch.Tensor
    output_difference: torch.Tensor
    target_classes: torch.Tensor
    logits: torch.Tensor


def integrated_gradients(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    baselines: torch.Tensor,
    target_classes: torch.Tensor | None = None,
    n_steps: int = 128,
    integration_method: str = "gausslegendre",
    internal_batch_size: int | None = None,
) -> IntegratedGradientsResult:
    """Calculate predicted- or explicit-class Integrated Gradients.

    Captum defines convergence delta as the difference between the summed attributions and the
    target-score change from baseline to input. Relative delta divides its magnitude by that
    target-score change and is retained alongside the signed raw attribution tensor.
    """

    if inputs.ndim != 4:
        raise ValueError("inputs must have shape [samples, channels, rows, columns]")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if baselines.shape[0] == 1 and inputs.shape[0] != 1:
        baselines = baselines.expand_as(inputs)
    if baselines.shape != inputs.shape:
        raise ValueError("baselines must match inputs or contain one broadcastable sample")

    from captum.attr import IntegratedGradients

    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        if target_classes is None:
            target_classes = logits.argmax(dim=1)
    target_classes = target_classes.to(device=inputs.device, dtype=torch.long)
    if target_classes.shape != (inputs.shape[0],):
        raise ValueError("target_classes must contain one class index per input sample")

    explainer = IntegratedGradients(model)
    attributions, convergence_delta = explainer.attribute(
        inputs,
        baselines=baselines,
        target=target_classes,
        n_steps=n_steps,
        method=integration_method,
        internal_batch_size=internal_batch_size,
        return_convergence_delta=True,
    )
    with torch.no_grad():
        baseline_logits = model(baselines)
        input_scores = logits.gather(1, target_classes[:, None]).squeeze(1)
        baseline_scores = baseline_logits.gather(1, target_classes[:, None]).squeeze(1)
        output_difference = input_scores - baseline_scores
        relative_delta = convergence_delta.abs() / output_difference.abs().clamp_min(1e-12)
    return IntegratedGradientsResult(
        attributions=attributions.detach(),
        convergence_delta=convergence_delta.detach(),
        relative_convergence_delta=relative_delta.detach(),
        output_difference=output_difference.detach(),
        target_classes=target_classes.detach(),
        logits=logits.detach(),
    )
