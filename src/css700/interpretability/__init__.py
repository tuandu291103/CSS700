"""Lazy attribution exports keep tree and CNN native runtimes isolated."""

from importlib import import_module
from typing import Any

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
    "ProbabilityTreeSHAP",
]

_EXPORT_MODULES = {
    "FeatureDeletionResult": "feature_deletion",
    "cumulative_feature_deletion": "feature_deletion",
    "GradCAMResult": "grad_cam",
    "grad_cam": "grad_cam",
    "IntegratedGradientsResult": "integrated_gradients",
    "integrated_gradients": "integrated_gradients",
    "OcclusionSensitivityResult": "occlusion",
    "occlusion_sensitivity": "occlusion",
    "trace_attributions_to_features": "tracing",
    "ProbabilityTreeSHAP": "tree_shap",
}


def __getattr__(name: str) -> Any:
    module = _EXPORT_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    loaded = import_module(f"css700.interpretability.{module}")
    # Importing a submodule also installs its name on the package. Replace that
    # module-valued alias with the public callable (e.g. grad_cam) on first load.
    for export, owner in _EXPORT_MODULES.items():
        if owner == module:
            globals()[export] = getattr(loaded, export)
    return globals()[name]
