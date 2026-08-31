"""Class-targeted Grad-CAM for artificial tabular images."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as functional


@dataclass(frozen=True)
class GradCAMResult:
    """Raw positive-evidence maps plus display-normalized copies."""

    raw: torch.Tensor
    normalized: torch.Tensor
    target_classes: torch.Tensor
    logits: torch.Tensor


def grad_cam(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    target_layer: nn.Module,
    target_classes: torch.Tensor | None = None,
) -> GradCAMResult:
    """Return Grad-CAM maps resized to the input grid.

    Channel weights are spatially averaged output gradients. Standard ReLU sign handling retains
    positive evidence for the target class. Per-map normalization is for display only; callers
    should retain and trace ``raw`` values for analysis.
    """

    if inputs.ndim != 4:
        raise ValueError("inputs must have shape [samples, channels, rows, columns]")
    activations: list[torch.Tensor] = []

    def capture_activation(
        _: nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        activations.append(output)

    handle = target_layer.register_forward_hook(capture_activation)
    try:
        model.eval()
        logits = model(inputs)
        if not activations:
            raise RuntimeError("the target layer did not run during the forward pass")
        if target_classes is None:
            target_classes = logits.argmax(dim=1)
        target_classes = target_classes.to(device=logits.device, dtype=torch.long)
        if target_classes.shape != (inputs.shape[0],):
            raise ValueError("target_classes must contain one class index per input sample")
        target_scores = logits.gather(1, target_classes[:, None]).sum()
        gradients = torch.autograd.grad(target_scores, activations[-1])[0]
        channel_weights = gradients.mean(dim=(2, 3), keepdim=True)
        raw = torch.relu((channel_weights * activations[-1]).sum(dim=1, keepdim=True))
        if raw.shape[2:] != inputs.shape[2:]:
            raw = functional.interpolate(
                raw, size=inputs.shape[2:], mode="bilinear", align_corners=False
            )
        maxima = raw.amax(dim=(2, 3), keepdim=True)
        normalized = torch.where(maxima > 0, raw / maxima.clamp_min(1e-12), raw)
        return GradCAMResult(
            raw=raw.detach(),
            normalized=normalized.detach(),
            target_classes=target_classes.detach(),
            logits=logits.detach(),
        )
    finally:
        handle.remove()
