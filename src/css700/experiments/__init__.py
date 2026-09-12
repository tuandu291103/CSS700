"""Versioned experiment runners with optional-stack lazy imports."""

from importlib import import_module
from typing import Any

__all__ = [
    "run_faithfulness_experiment",
    "run_grad_cam_experiment",
    "run_grid_cnn_experiment",
    "run_grid_stability_experiment",
    "run_integrated_gradients_experiment",
    "run_occlusion_experiment",
    "run_tabular_baselines_experiment",
    "run_tree_shap_experiment",
]

_EXPORT_MODULES = {
    "run_faithfulness_experiment": "css700.experiments.faithfulness",
    "run_grad_cam_experiment": "css700.experiments.grad_cam",
    "run_grid_cnn_experiment": "css700.experiments.grid_cnn",
    "run_grid_stability_experiment": "css700.experiments.stability",
    "run_integrated_gradients_experiment": "css700.experiments.integrated_gradients",
    "run_occlusion_experiment": "css700.experiments.occlusion",
    "run_tabular_baselines_experiment": "css700.experiments.tabular_baselines",
    "run_tree_shap_experiment": "css700.experiments.tree_shap",
}


def __getattr__(name: str) -> Any:
    """Load only the experiment stack needed by the requested command."""

    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name), name)
