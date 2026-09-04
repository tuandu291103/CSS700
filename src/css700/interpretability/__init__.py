"""Attribution generation and tracing utilities."""

from css700.interpretability.feature_deletion import (
    FeatureDeletionResult,
    cumulative_feature_deletion,
)
from css700.interpretability.grad_cam import GradCAMResult, grad_cam
from css700.interpretability.integrated_gradients import (
    IntegratedGradientsResult,
    integrated_gradients,
)
from css700.interpretability.occlusion import (
    OcclusionSensitivityResult,
    occlusion_sensitivity,
)
from css700.interpretability.tracing import trace_attributions_to_features

__all__ = [
    "FeatureDeletionResult",
    "GradCAMResult",
    "IntegratedGradientsResult",
    "OcclusionSensitivityResult",
    "cumulative_feature_deletion",
    "grad_cam",
    "integrated_gradients",
    "occlusion_sensitivity",
    "trace_attributions_to_features",
]
