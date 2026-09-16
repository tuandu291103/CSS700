"""Frozen provenance gates for the original v1 LIME test pilot."""

import importlib
from pathlib import Path

import numpy as np
import yaml

from css700.experiments.lime_locality import inventory_digest, load_preflight
from css700.experiments.lime_preflight import validate_lime_config
from css700.experiments.tree_shap_sources import file_sha256, load_sources
from css700.experiments.tree_shap_stability_sources import load_accepted_tree_run

PREFLIGHT = {
    "preflight_run": (
        "artifacts/20260915-065803-601734__breast-cancer__lime__model-seed-42__training-preflight"
    ),
    "preflight_inventory_sha256": (
        "49b11bf5fef8b238f768e71951c0f23657c18c29853367bbdc3d70d11de82505"
    ),
    "preflight_config_sha256": ("bf6c361aadafeddf57326bfc72cb1ba9a520f728269fef0e4d84e3066c92d3a6"),
    "preflight_commit": "dfa0327d3f14fef34f75f5ca24ceff15e10ee757",
    "anchor_training_ids": [460, 165, 122, 455, 412, 397, 182, 429],
}
SHAP_INVENTORY = "a53b5b759d74435d9e62a5441fb1e1edb67c9ab4deb6fef5d0316dbbf5349964"


def load_test_sources(config: dict, hashes: dict) -> dict:
    """Validate all inputs before model deserialization or new test explanations."""
    validate_lime_config(config)
    _, preflight, bundle, _ = load_preflight(PREFLIGHT, hashes)
    for module, expected in preflight["lime_source_sha256"].items():
        path = Path(importlib.import_module(module).__file__).resolve()
        if file_sha256(path) != expected:
            raise ValueError("active LIME implementation differs from accepted preflight")
        hashes[str(path)] = expected
    source_path = Path(config["source_config"])
    if file_sha256(source_path) != config["source_config_sha256"]:
        raise ValueError("frozen source config changed")
    hashes[str(source_path.resolve())] = file_sha256(source_path)
    sources = load_sources(yaml.safe_load(source_path.read_text()))
    hashes.update(sources["file_hashes"])
    shap_root = Path(config["accepted_shap_run"])
    digest, inventory = inventory_digest(shap_root)
    if digest != SHAP_INVENTORY or len(inventory) != 39:
        raise ValueError("accepted TreeSHAP inventory changed or incomplete")
    hashes.update({str((shap_root / p).resolve()): h for p, h in inventory.items()})
    shap = load_accepted_tree_run(shap_root, source_path, hashes)
    if (
        shap["seed"] != 42
        or tuple(shap["feature_names"]) != bundle.feature_names
        or not np.array_equal(shap["sample_ids"], bundle.test_indices)
        or not np.array_equal(shap["labels"], bundle.y_test)
        or bundle.x_train.shape != (397, 30)
        or bundle.x_test.shape != (86, 30)
    ):
        raise ValueError("test/preflight/TreeSHAP identity mismatch")
    return {**sources, "shap": shap, "preflight_manifest": preflight}


def verify_predictions(probabilities: np.ndarray, expected: dict) -> float:
    """Existing reload tolerance; require identical predicted targets as well."""
    if (
        probabilities.shape != expected["probabilities"].shape
        or not np.isfinite(probabilities).all()
        or not np.array_equal(probabilities.argmax(axis=1), expected["predictions"])
        or not np.allclose(probabilities, expected["probabilities"], atol=1e-7, rtol=1e-6)
    ):
        raise ValueError("LIME probabilities or targets differ from the frozen model reference")
    return float(np.max(abs(probabilities - expected["probabilities"])))
