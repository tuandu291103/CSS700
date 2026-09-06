"""Shared configuration contracts."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExperimentSeeds:
    """Resolved data and model seeds with legacy-config compatibility."""

    data_split_seed: int
    model_seed: int
    explicit_separation: bool


def resolve_experiment_seeds(config: dict[str, Any]) -> ExperimentSeeds:
    """Resolve explicit split/model seeds or fall back to the legacy shared seed."""

    experiment = config.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("config must contain an experiment mapping")
    legacy_seed = experiment.get("seed")
    split_seed = experiment.get("data_split_seed", legacy_seed)
    model_seed = experiment.get("model_seed", legacy_seed)
    if split_seed is None or model_seed is None:
        raise ValueError("experiment must define both data_split_seed and model_seed, or seed")
    return ExperimentSeeds(
        data_split_seed=int(split_seed),
        model_seed=int(model_seed),
        explicit_separation=("data_split_seed" in experiment or "model_seed" in experiment),
    )
