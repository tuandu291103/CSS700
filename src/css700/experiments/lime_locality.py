"""Append-only, training-only LIME mechanism check on frozen preflight outputs."""

import hashlib
import json
import platform
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml
from threadpoolctl import threadpool_limits

from css700.evaluation.lime_locality import (
    VARIANTS,
    descriptive_summaries,
    fit_locality_variant,
    paired_contrasts,
    variant_inputs,
    verify_reference,
)
from css700.experiments.lime_preflight import validate_lime_config
from css700.experiments.som_mapping_audit import _git_state
from css700.experiments.tree_shap import _json
from css700.experiments.tree_shap_sources import file_sha256, load_sources
from css700.interpretability.lime_tabular import build_lime_design


def inventory_digest(root: Path) -> tuple[str, dict]:
    paths = sorted(p for p in root.rglob("*") if p.is_file())
    if any(p.is_symlink() for p in paths):
        raise ValueError("symlinks are not accepted in the preflight inventory")
    inventory = {str(p.relative_to(root)): file_sha256(p) for p in paths}
    digest = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return digest, inventory


def validate_config(config: dict) -> None:
    expected = {
        "protocol_version": "lime-locality-diagnostic-v1",
        "preflight_run": (
            "artifacts/20260915-065803-601734__breast-cancer__lime__model-seed-42__training-preflight"
        ),
        "preflight_inventory_sha256": (
            "49b11bf5fef8b238f768e71951c0f23657c18c29853367bbdc3d70d11de82505"
        ),
        "preflight_config_sha256": (
            "bf6c361aadafeddf57326bfc72cb1ba9a520f728269fef0e4d84e3066c92d3a6"
        ),
        "preflight_commit": "dfa0327d3f14fef34f75f5ca24ceff15e10ee757",
        "model_seed": 42,
        "data_split_seed": 42,
        "anchor_training_ids": [460, 165, 122, 455, 412, 397, 182, 429],
        "explainer_seeds": [11, 29, 47],
        "families": ["random_forest", "xgboost"],
        "variants": list(VARIANTS),
        "reference_width_multiplier": 0.75,
        "narrow_width_multiplier": 0.25,
        "ridge_alpha": 1.0,
        "ridge_solver": "cholesky",
        "reference_evaluation": "saved_v1_evaluation_rows_and_weights",
        "selection_rule": "none_diagnostic_only",
        "reporting": "paired_descriptive_no_intervals",
        "new_model_calls": 0,
        "new_test_explanations": 0,
    }
    if json.dumps(config, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("unsupported or changed frozen locality diagnostic")


def require_accepted_preflight(manifest: dict, summary: dict, config: dict) -> None:
    required = {
        "clean_worktree",
        "no_cnn_runtime",
        "all_48_training_fits_completed",
        "six_same_seed_repeats_passed",
        "only_registered_training_anchors",
        "source_files_unchanged",
    }
    if (
        manifest.get("status") != "ok"
        or summary.get("status") != "ok"
        or manifest.get("worktree_dirty_at_start") is not False
        or manifest.get("commit_sha") != config["preflight_commit"]
        or manifest.get("config_sha256") != config["preflight_config_sha256"]
        or manifest.get("anchor_training_ids") != config["anchor_training_ids"]
        or manifest.get("scope") != "training_anchors_and_synthetic_neighbors_only"
        or summary.get("errors") != []
        or summary.get("completed_training_fits") != 48
        or summary.get("test_explanations_generated") != 0
        or not required.issubset(summary.get("checks", {}))
        or not all(v is True for v in summary["checks"].values())
    ):
        raise ValueError("preflight is not the registered accepted training-only source")


def load_preflight(config: dict, hashes: dict) -> tuple:
    root = Path(config["preflight_run"])
    digest, inventory = inventory_digest(root)
    if digest != config["preflight_inventory_sha256"] or len(inventory) != 148:
        raise ValueError("preflight artifact inventory changed or incomplete")
    hashes.update({str((root / p).resolve()): h for p, h in inventory.items()})
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    require_accepted_preflight(manifest, summary, config)
    if file_sha256(root / "config.yaml") != config["preflight_config_sha256"]:
        raise ValueError("preflight config hash mismatch")
    frozen = yaml.safe_load((root / "config.yaml").read_text())
    validate_lime_config(frozen)
    for path, expected in manifest["source_file_sha256"].items():
        if file_sha256(Path(path)) != expected:
            raise ValueError(f"preflight source input changed: {path}")
        hashes[path] = expected
    for package, expected in manifest["packages"].items():
        if version(package) != expected:
            raise ValueError(f"preflight package version changed: {package}")
    source_config = yaml.safe_load(Path(frozen["source_config"]).read_text())
    sources = load_sources(source_config)  # Integrity reads only; no deserialization/inference.
    hashes.update(sources["file_hashes"])
    bundle = sources["bundle"]
    if bundle.train_indices[:8].tolist() != config["anchor_training_ids"]:
        raise ValueError("registered anchors are not the first eight training rows")
    aggregate = json.loads((root / "training_diagnostics.json").read_text())
    indexed = {(r["sample_id"], r["explainer_seed"], r["family"]): r for r in aggregate}
    if len(aggregate) != 48 or len(indexed) != 48:
        raise ValueError("missing or duplicate preflight fits")
    return root, manifest, bundle, indexed


def read_saved_case(root: Path, bundle, position: int, seed: int, indexed: dict) -> tuple:
    sample_id = int(bundle.train_indices[position])
    case = root / f"sample-{sample_id}__explainer-seed-{seed}"
    design = build_lime_design(
        bundle.x_train,
        bundle.train_indices,
        np.concatenate((bundle.validation_indices, bundle.test_indices)),
        bundle.x_train[position],
        bundle.feature_names,
        sample_id=sample_id,
        explainer_seed=seed,
    )
    if json.loads((case / "metadata.json").read_text()) != design.metadata:
        raise ValueError("preflight neighborhood metadata mismatch")
    with np.load(case / "neighborhoods.npz", allow_pickle=False) as saved:
        for role in ("fit", "evaluation"):
            for field in ("design", "inverse", "distances", "weights"):
                if not np.array_equal(
                    saved[f"{role}_{field}"], getattr(getattr(design, role), field)
                ):
                    raise ValueError("preflight neighborhood no longer reproduces from training")
    outputs = {}
    for family in ("random_forest", "xgboost"):
        diag = json.loads((case / f"{family}.json").read_text())
        if diag != indexed[(sample_id, seed, family)] or diag["partition"] != "training":
            raise ValueError("preflight fit identity or aggregate mismatch")
        with np.load(case / f"{family}.npz", allow_pickle=False) as data:
            arrays = {key: value.copy() for key, value in data.items()}
        if (
            arrays["coefficients"].shape != (30,)
            or arrays["probabilities"].shape != (7000, 2)
            or arrays["fit_surrogate_predictions"].shape != (5000,)
            or arrays["evaluation_surrogate_predictions"].shape != (2000,)
        ):
            raise ValueError("preflight feature/class/sample axes mismatch")
        outputs[family] = (arrays, diag)
    return design, outputs


def run_lime_locality(config_path: Path, output_dir: Path) -> dict:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text())
    validate_config(config)
    commit, dirty = _git_state()
    root = output_dir / (
        datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
        + "__breast-cancer__lime__training-locality-diagnostic"
    )
    root.mkdir(parents=True, exist_ok=False)
    (root / "config.yaml").write_bytes(config_path.read_bytes())
    hashes = {str(config_path.resolve()): file_sha256(config_path)}
    manifest = dict(
        run_id=root.name,
        commit_sha=commit,
        worktree_dirty_at_start=dirty,
        config_sha256=file_sha256(config_path),
        python=platform.python_version(),
        created_at_utc=datetime.now(UTC).isoformat(),
        status="running",
        committee_approval_claimed=False,
        source_run=config["preflight_run"],
    )
    _json(root / "manifest.json", manifest)
    checks = {"clean_worktree": not dirty, "no_cnn_runtime": "torch" not in sys.modules}
    errors, records, pairs = [], [], []
    try:
        if not all(checks.values()):
            raise ValueError("locality diagnostic requires a clean commit and no CNN runtime")
        source_root, source_manifest, bundle, indexed = load_preflight(config, hashes)
        manifest.update(
            packages=source_manifest["packages"],
            source_commit=source_manifest["commit_sha"],
            threadpool_limit=1,
        )
        scale = bundle.x_train.std(axis=0, ddof=0)
        np.save(root / "training_feature_scale.npy", scale)
        reference_count = 0
        with threadpool_limits(limits=1):
            for position, sample_id in enumerate(config["anchor_training_ids"]):
                for seed in config["explainer_seeds"]:
                    design, outputs = read_saved_case(source_root, bundle, position, seed, indexed)
                    for variant in VARIANTS:
                        case = root / f"sample-{sample_id}__explainer-seed-{seed}" / variant
                        case.mkdir(parents=True, exist_ok=False)
                        inputs = variant_inputs(design, scale, variant)
                        np.savez_compressed(case / "design_weights.npz", **inputs)
                        for family, (arrays, original) in outputs.items():
                            result = fit_locality_variant(
                                design, arrays["probabilities"], scale, variant
                            )
                            if variant == "reference":
                                verify_reference(result, arrays, original)
                                reference_count += 1
                            record = dict(
                                sample_id=sample_id,
                                explainer_seed=seed,
                                family=family,
                                variant=variant,
                                partition="training",
                                **result["diagnostics"],
                            )
                            records.append(record)
                            _json(case / f"{family}.json", record)
                            np.savez_compressed(
                                case / f"{family}.npz",
                                **{k: v for k, v in result.items() if k != "diagnostics"},
                            )
        pairs = paired_contrasts(records)
        checks.update(
            all_192_fits_completed=len(records) == 192,
            all_48_references_match=reference_count == 48,
            all_144_paired_contrasts=len(pairs) == 144,
            all_24_neighborhoods_revalidated=True,
        )
        _json(root / "descriptive_summary.json", descriptive_summaries(records, pairs))
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    try:
        checks["source_files_unchanged"] = all(file_sha256(Path(p)) == h for p, h in hashes.items())
    except OSError as error:
        checks["source_files_unchanged"] = False
        errors.append(str(error))
    checks["no_cnn_runtime"] = "torch" not in sys.modules
    summary = dict(
        status="ok" if all(checks.values()) and not errors else "quality_gate_failed",
        output=str(root),
        checks=checks,
        errors=errors,
        completed_fits=len(records),
        paired_contrasts=len(pairs),
        new_model_calls=0,
        test_explanations_generated=0,
        selected_variant=None,
        runtime_seconds=time.perf_counter() - started,
        interpretation="Bounded training-only diagnostic; no automatic protocol selection.",
    )
    manifest.update(
        status=summary["status"],
        completed_at_utc=datetime.now(UTC).isoformat(),
        source_file_sha256=hashes,
    )
    _json(root / "fit_diagnostics.json", records)
    _json(root / "paired_contrasts.json", pairs)
    _json(root / "manifest.json", manifest)
    _json(root / "summary.json", summary)
    return summary
