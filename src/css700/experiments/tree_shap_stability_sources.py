"""Revalidate accepted TreeSHAP bundles before comparing registered model seeds."""

import csv
import json
from pathlib import Path

import numpy as np
import yaml

from css700.experiments.tree_shap_sources import file_sha256, load_sources, named_feature_matrix


def scientific_settings(config: dict) -> dict:
    return {k: v for k, v in config.items() if k not in ("model_seed", "grid_run", "model_sha256")}


def load_accepted_tree_run(root: Path, expected_config_path: Path, hashes: dict[str, str]) -> dict:
    def source(relative: str) -> Path:
        path = (root / relative).resolve()
        hashes[str(path)] = file_sha256(path)
        return path

    def read_json(relative: str) -> dict:
        return json.loads(source(relative).read_text())

    def read_npz(relative: str) -> dict:
        with np.load(source(relative), allow_pickle=False) as data:
            return {k: v.copy() for k, v in data.items()}

    config_path = source("config.yaml")
    expected_config_path = expected_config_path.resolve()
    hashes[str(expected_config_path)] = file_sha256(expected_config_path)
    manifest, summary = read_json("manifest.json"), read_json("summary.json")
    if (
        manifest["status"] != "ok"
        or summary["status"] != "ok"
        or manifest["worktree_dirty_at_start"]
        or summary["errors"]
        or not summary["checks"]
        or not all(value is True for value in summary["checks"].values())
    ):
        raise ValueError("source TreeSHAP run was not accepted from a clean commit")
    if not (
        file_sha256(config_path) == manifest["config_sha256"] == file_sha256(expected_config_path)
    ):
        raise ValueError("source TreeSHAP config does not match the frozen seed config")
    config = yaml.safe_load(config_path.read_text())
    if config["protocol_version"] != "tree-shap-probability-v1.1":
        raise ValueError("stability requires the accepted v1.1 protocol")
    seed = config["model_seed"]
    if manifest.get("model_seed", seed) != seed:
        raise ValueError("source manifest model seed mismatch")
    original = load_sources(config)  # verifies named identities, saved models and original hashes
    if original["file_hashes"] != manifest["source_file_sha256"]:
        raise ValueError("original source files changed since the TreeSHAP run")
    hashes.update(original["file_hashes"])
    bundle = original["bundle"]
    if (
        tuple(manifest["feature_names"]) != bundle.feature_names
        or tuple(manifest["class_names"]) != bundle.target_names
        or manifest["split_fingerprint"] != config["expected_split_sha256"]
    ):
        raise ValueError("TreeSHAP feature, class or split identity mismatch")
    splits = read_npz("splits.npz")
    for name in ("train", "validation", "test"):
        if not np.array_equal(splits[name], getattr(bundle, f"{name}_indices")):
            raise ValueError("TreeSHAP split order mismatch")
    scaler = read_npz("preprocessing.npz")
    if not all(
        np.array_equal(scaler[k], getattr(bundle, f"scaler_{k}")) for k in ("mean", "scale")
    ):
        raise ValueError("TreeSHAP preprocessing mismatch")
    expected_conditions = {
        f"{family}__{bg}"
        for family in ("random_forest", "xgboost")
        for bg in ("train_all", "train_100")
    }
    if set(summary["conditions"]) != expected_conditions:
        raise ValueError("missing or unexpected model/background conditions")
    backgrounds, conditions = {}, {}
    for bg in ("train_all", "train_100"):
        data = read_npz(f"background_{bg}.npz")
        positions = (
            np.arange(len(bundle.x_train))
            if bg == "train_all"
            else np.sort(
                np.random.default_rng(config["background"]["sensitivity_seed"]).permutation(
                    len(bundle.x_train)
                )[:100]
            )
        )
        if not np.array_equal(data["values"], bundle.x_train[positions]) or not np.array_equal(
            data["sample_ids"], bundle.train_indices[positions]
        ):
            raise ValueError("background is not the declared ordered training-only selection")
        backgrounds[bg] = data
        for family in ("random_forest", "xgboost"):
            name = f"{family}__{bg}"
            report = read_json(f"{name}/checks.json")
            required_checks = {
                "probability_reconstruction",
                "base_matches_background_probability_mean",
                "binary_contributions_complementary",
            }
            if family == "random_forest":
                required_checks.add("copied_forest_matches_original_probabilities")
                audit = read_json(f"{name}/routing_audit.json")
                source(f"{name}/adapter_thresholds.npz")
                if (
                    audit["passed"] is not True
                    or audit["adapted_background_routing_mismatches"] != 0
                ):
                    raise ValueError("source RF adapter routing audit failed")
            smoke_checks = required_checks | {"warmup_passed", "repeat_passed", "repeatable"}
            if (
                report != summary["conditions"][name]
                or report["status"] != "ok"
                or not all(report["test_checks"].get(k) is True for k in required_checks)
                or not all(report["smoke_checks"].get(k) is True for k in smoke_checks)
            ):
                raise ValueError("source condition quality checks failed or disagree")
            effective = read_npz(f"{name}/explainer_background.npz")
            expected_values = (
                data["values"].astype(np.float32).astype(float)
                if family == "random_forest"
                else data["values"]
            )
            if not np.array_equal(effective["values"], expected_values) or not np.array_equal(
                effective["sample_ids"], data["sample_ids"]
            ):
                raise ValueError("effective background representation mismatch")
            raw = read_npz(f"{name}/test_attributions.npz")
            n, d = len(bundle.test_indices), len(bundle.feature_names)
            shapes = {
                "values": (n, d, 2),
                "base": (2,),
                "probabilities": (n, 2),
                "residuals": (n, 2),
            }
            if any(
                raw[k].shape != shape or not np.isfinite(raw[k]).all()
                for k, shape in shapes.items()
            ):
                raise ValueError("invalid raw attribution shape or finite-value contract")
            predictions = original["tree_predictions"][family]
            probabilities = raw["probabilities"]
            residuals = raw["values"].sum(axis=1) + raw["base"] - probabilities
            settings = config["shap"]
            if (
                np.any(probabilities < 0)
                or np.any(probabilities > 1)
                or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6, rtol=0)
                or not np.allclose(
                    raw["values"][:, :, 0],
                    -raw["values"][:, :, 1],
                    atol=settings["reconstruction_atol"],
                    rtol=settings["reconstruction_rtol"],
                )
                or not np.allclose(
                    probabilities, predictions["probabilities"], atol=1e-7, rtol=1e-6
                )
                or not np.array_equal(probabilities.argmax(axis=1), predictions["predictions"])
                or not np.array_equal(raw["residuals"], residuals)
                or not np.all(
                    abs(residuals)
                    <= settings["reconstruction_atol"]
                    + settings["reconstruction_rtol"] * abs(probabilities)
                )
            ):
                raise ValueError("raw prediction or reconstruction verification failed")
            with source(f"{name}/target_features.csv").open(newline="") as handle:
                rows = [{**r, "predicted_label": r["target_class"]} for r in csv.DictReader(handle)]
            scores = named_feature_matrix(
                rows,
                bundle.test_indices,
                bundle.feature_names,
                predictions["predictions"],
                bundle.y_test,
                value_field="signed_attribution",
            )
            selected = raw["values"][np.arange(n), :, predictions["predictions"]]
            if not np.array_equal(scores, selected):
                raise ValueError("named feature scores differ from the signed class arrays")
            conditions[name] = {
                "scores": scores,
                "predictions": predictions["predictions"],
                "probabilities": probabilities,
            }
    return {
        "seed": seed,
        "config": config,
        "manifest": manifest,
        "conditions": conditions,
        "backgrounds": backgrounds,
        "sample_ids": bundle.test_indices,
        "labels": bundle.y_test,
        "feature_names": bundle.feature_names,
    }
