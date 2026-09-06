import pytest

from css700.config import resolve_experiment_seeds


def test_legacy_shared_seed_remains_supported() -> None:
    seeds = resolve_experiment_seeds({"experiment": {"seed": 42}})
    assert seeds.data_split_seed == 42
    assert seeds.model_seed == 42
    assert not seeds.explicit_separation


def test_explicit_split_and_model_seeds_are_separated() -> None:
    seeds = resolve_experiment_seeds({"experiment": {"data_split_seed": 42, "model_seed": 7}})
    assert seeds.data_split_seed == 42
    assert seeds.model_seed == 7
    assert seeds.explicit_separation


def test_incomplete_seed_contract_is_rejected() -> None:
    with pytest.raises(ValueError, match="both data_split_seed and model_seed"):
        resolve_experiment_seeds({"experiment": {"model_seed": 7}})
