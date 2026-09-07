import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from css700.data import load_breast_cancer_splits
from css700.data.splits import split_fingerprint
from css700.encoding.mapping import audit_feature_mapping
from css700.encoding.som_diagnostic import fit_som_diagnostic

pytest.importorskip("minisom")

SETTINGS = {"layout_seed": 42, "sigma": 1.0, "learning_rate": 0.5, "update_steps": 20}


def test_native_bmu_can_collide_even_with_sufficient_capacity() -> None:
    # Identical feature vectors have the same nearest neuron, irrespective of capacity.
    values = np.tile(np.arange(10, dtype=float)[:, None], (1, 4))
    first = fit_som_diagnostic(values, ["a", "b", "c", "d"], **SETTINGS)
    repeated = fit_som_diagnostic(values, ["a", "b", "c", "d"], **SETTINGS)
    assert first.weights.shape == (2, 2, 10)
    assert first.mapping == repeated.mapping
    np.testing.assert_array_equal(first.weights, repeated.weights)
    audit = audit_feature_mapping(first.mapping, (1, 2, 2))
    assert audit["pixel_capacity"] == audit["feature_count"] == 4
    assert audit["occupied_pixels"] == 1
    assert audit["features_in_collisions"] == 4
    assert not audit["one_to_one"]


def test_som_uses_feature_vectors_and_manuscript_grid_orientation() -> None:
    values = np.random.default_rng(1).normal(size=(12, 30))
    result = fit_som_diagnostic(values, [f"f{i}" for i in range(30)], **SETTINGS)
    assert result.weights.shape == (6, 5, 12)
    assert len(result.mapping) == 30
    assert np.isfinite(result.quantization_error)
    with pytest.raises(ValueError, match="finite"):
        fit_som_diagnostic(values * np.nan, [f"f{i}" for i in range(30)], **SETTINGS)


def test_audit_runner_does_not_fit_heldout_values_or_labels(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("matplotlib")
    from css700.experiments import som_mapping_audit as runner

    bundle = load_breast_cancer_splits()
    poisoned = replace(
        bundle,
        x_validation=np.full_like(bundle.x_validation, np.nan),
        x_test=np.full_like(bundle.x_test, np.nan),
        y_train=np.full_like(bundle.y_train, -999),
        y_validation=np.full_like(bundle.y_validation, -999),
        y_test=np.full_like(bundle.y_test, -999),
    )
    monkeypatch.setattr(runner, "load_breast_cancer_splits", lambda **_: poisoned)
    monkeypatch.setattr(runner, "_git_state", lambda: ("test-fixture", False))
    calls = []

    def checked_fit(values, names, **settings):
        np.testing.assert_array_equal(values, bundle.x_train)
        calls.append(values.shape)
        return fit_som_diagnostic(values, names, **settings)

    monkeypatch.setattr(runner, "fit_som_diagnostic", checked_fit)
    config = yaml.safe_load(Path("configs/som_mapping_audit.yaml").read_text())
    config["som"]["update_steps"] = 20
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    summary = runner.run_som_mapping_audit(config_path, tmp_path)
    assert summary["status"] == "ok"
    assert not summary["som2img_cnn_ready"]
    assert calls == [(397, 30), (397, 30)]
    root = Path(summary["output"])
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["layout_fit_shape"] == [30, 397]
    assert manifest["split_fingerprint"] == config["experiment"]["expected_split_fingerprint"]
    assert (root / "encoding" / "candidate_bmu_map.csv").is_file()
    assert (root / "figures" / "bmu_occupancy.png").is_file()
    assert not (root / "model").exists()
    monkeypatch.setattr(runner, "_git_state", lambda: ("test-fixture", True))
    dirty = runner.run_som_mapping_audit(config_path, tmp_path)
    assert dirty["status"] == "quality_gate_failed"
    assert dirty["output"] != summary["output"]


def test_shared_fingerprint_preserves_reference_and_order() -> None:
    bundle = load_breast_cancer_splits()
    assert split_fingerprint(
        bundle.train_indices, bundle.validation_indices, bundle.test_indices
    ) == ("79e85d4ceae692441b84f6921174e89879dc7eca4e27e8e6f6744799a67aa7cb")
    assert split_fingerprint(
        bundle.train_indices[::-1], bundle.validation_indices, bundle.test_indices
    ) != ("79e85d4ceae692441b84f6921174e89879dc7eca4e27e8e6f6744799a67aa7cb")
