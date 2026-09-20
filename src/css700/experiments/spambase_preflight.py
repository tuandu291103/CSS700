"""Training-only scaling, fixed Grid identity checks and bounded CPU compatibility."""

import platform
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml

from css700.data.spambase import (
    fit_training_standardizer,
    load_reviewed_spambase_training,
)
from css700.data.spambase_audit import FEATURE_NAMES
from css700.encoding import GridEncoder
from css700.encoding.mapping import audit_feature_mapping
from css700.experiments.som_mapping_audit import _git_state
from css700.experiments.spambase_audit import sha256, write_json
from css700.interpretability.tracing import trace_attributions_to_features

PREFLIGHT_POLICY = {
    "fit_scope": "accepted_training_rows_only",
    "heldout_transform": False,
    "scaler": "sklearn_standard_scaler_population_variance",
    "feature_selection": "none",
    "grid": {"rows": 8, "columns": 8, "padding_value": 0.0, "order": "source"},
    "statistics_rtol": 1e-10,
    "statistics_atol": 1e-12,
    "standardized_moment_atol": 1e-10,
    "smoke": {"device": "cpu", "batch_size": 32, "seed": 42, "threads": 1, "optimizer_steps": 0},
}


def cpu_compatibility_check(images: np.ndarray, labels: np.ndarray) -> dict:
    """One fresh untrained model, one forward/backward pass, zero parameter updates."""
    import torch
    from torch import nn

    from css700.models import SmallTabularImageCNN

    old_threads = torch.get_num_threads()
    old_determinism = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    started = time.perf_counter()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(42)
            model = SmallTabularImageCNN(input_channels=1, num_classes=2).cpu()
            before = {k: v.detach().clone() for k, v in model.state_dict().items()}
            inputs = torch.from_numpy(images[:32].copy()).requires_grad_(True)
            targets = torch.from_numpy(labels[:32].copy())
            logits = model(inputs)
            loss = nn.CrossEntropyLoss()(logits, targets)
            loss.backward()
            checks = {
                "output_shape": list(logits.shape) == [len(inputs), 2],
                "finite_logits_and_loss": bool(
                    torch.isfinite(logits).all() and torch.isfinite(loss)
                ),
                "finite_input_gradient": inputs.grad is not None
                and bool(torch.isfinite(inputs.grad).all()),
                "finite_parameter_gradients": all(
                    p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for p in model.parameters()
                ),
                "no_parameter_updates": all(
                    torch.equal(before[k], v) for k, v in model.state_dict().items()
                ),
            }
            return {
                "checks": checks,
                "device": "cpu",
                "seed": 42,
                "threads": 1,
                "batch_rows": len(inputs),
                "input_shape": list(inputs.shape),
                "output_shape": list(logits.shape),
                "parameters": sum(p.numel() for p in model.parameters()),
                "optimizer_steps": 0,
                "checkpoint_saved": False,
                "elapsed_seconds": time.perf_counter() - started,
                "interpretation": (
                    "Compatibility only, not predictive performance or a throughput benchmark."
                ),
            }
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_determinism, warn_only=old_warn_only)


def prepare_training_grid(source, root: Path) -> tuple[dict, list[str]]:
    scaler, standardized, info = fit_training_standardizer(source.x_train_raw)
    raw = source.x_train_raw
    constant = np.ptp(raw, axis=0) == 0
    guard = np.asarray(info["numerical_scale_guard_indices"], dtype=int)
    normal = ~constant
    normal[guard] = False
    policy = PREFLIGHT_POLICY
    checks = {
        "scaler_saw_training_rows_only": info["fit_rows"] == len(source.train_indices),
        "mean_matches_training_reference": bool(
            np.allclose(
                scaler.mean_,
                raw.mean(axis=0),
                rtol=policy["statistics_rtol"],
                atol=policy["statistics_atol"],
            )
        ),
        "variance_matches_training_reference": bool(
            np.allclose(
                scaler.var_,
                raw.var(axis=0),
                rtol=policy["statistics_rtol"],
                atol=policy["statistics_atol"],
            )
        ),
        "constant_features_retained_scale_one": bool(
            np.all(scaler.scale_[constant] == 1) and np.all(standardized[:, constant] == 0)
        ),
        "standardized_training_mean": bool(
            np.all(np.abs(standardized.mean(axis=0)) <= policy["standardized_moment_atol"])
        ),
        "standardized_training_variance": bool(
            np.allclose(
                standardized[:, normal].var(axis=0),
                1,
                rtol=0,
                atol=policy["standardized_moment_atol"],
            )
        ),
        "finite_training_arrays": bool(np.isfinite(standardized).all()),
    }
    encoder = GridEncoder(rows=8, columns=8, padding_value=0).fit(source.feature_names)
    images = encoder.transform(standardized)
    mask = encoder.padding_mask()
    mapping = audit_feature_mapping(encoder.mapping or (), (1, 8, 8))
    # These are synthetic tracing probes, NOT CNN-generated attribution maps.
    probe = images.copy()
    probe[:, ~mask] = 1e9
    reordered = tuple(reversed(encoder.mapping or ()))
    checks.update(
        all_57_feature_identities=(
            source.feature_names == FEATURE_NAMES
            and mapping["feature_count"] == 57
            and mapping["one_to_one"]
        ),
        seven_zero_padding_pixels=bool(int((~mask).sum()) == 7 and np.all(images[:, ~mask] == 0)),
        finite_float32_grid=bool(images.dtype == np.float32 and np.isfinite(images).all()),
        grid_values_preserved=bool(
            np.array_equal(images[:, mask], standardized.astype(np.float32))
        ),
        signed_tracing_excludes_padding=bool(
            np.array_equal(
                trace_attributions_to_features(probe, reordered), standardized.astype(np.float32)
            )
        ),
        absolute_tracing_excludes_padding=bool(
            np.array_equal(
                trace_attributions_to_features(probe, reordered, absolute=True),
                np.abs(standardized.astype(np.float32)),
            )
        ),
        target_names_correct=source.target_names == ("non-spam", "spam"),
    )
    np.savez_compressed(
        root / "training_arrays.npz",
        train_indices=source.train_indices,
        y_train=source.y_train,
        x_standardized=standardized,
        images=images,
        occupied_mask=mask,
        scaler_mean=scaler.mean_,
        scaler_var=scaler.var_,
        scaler_scale=scaler.scale_,
    )
    write_json(root / "feature_map.json", encoder.mapping_records())
    write_json(
        root / "training_feature_checks.json",
        [
            {
                "feature_index": j,
                "feature_name": name,
                "mean": float(scaler.mean_[j]),
                "variance": float(scaler.var_[j]),
                "scale": float(scaler.scale_[j]),
                "constant": bool(constant[j]),
                "numerical_scale_guard": j in guard,
                "raw_zero_count": int(np.sum(raw[:, j] == 0)),
            }
            for j, name in enumerate(source.feature_names)
        ],
    )
    smoke = cpu_compatibility_check(images, source.y_train)
    smoke["source_row_ids"] = source.train_indices[:32].tolist()
    write_json(root / "cpu_smoke.json", smoke)
    checks.update({f"cpu_{name}": value for name, value in smoke["checks"].items()})
    return {
        "checks": checks,
        "scaling": info,
        "mapping": mapping,
        "target_names": list(source.target_names),
        "training_image_shape": list(images.shape),
        "array_storage_bytes": {
            "standardized_training": standardized.nbytes,
            "training_images": images.nbytes,
        },
        "max_float32_cast_absolute_error": float(
            np.max(np.abs(standardized - standardized.astype(np.float32).astype(np.float64)))
        ),
        "cpu_smoke": smoke,
    }, ["training_arrays.npz", "feature_map.json", "training_feature_checks.json", "cpu_smoke.json"]


def run_spambase_preflight(config_path: Path, output_dir: Path) -> dict:
    started = time.perf_counter()
    commit, dirty = _git_state()
    root = output_dir / (
        datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f") + "__spambase__training-preflight-v1"
    )
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "commit_sha": commit,
        "worktree_dirty_at_start": dirty,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "running",
        "scope": "training_only_preprocessing_grid_and_cpu_compatibility",
        "python": platform.python_version(),
        "packages": {n: version(n) for n in ("numpy", "scikit-learn", "torch")},
    }
    write_json(root / "manifest.json", manifest)
    checks, hashes, errors, result, outputs = {"clean_worktree": not dirty}, {}, [], {}, []
    try:
        if dirty:
            raise ValueError("preflight requires a clean frozen code/config commit")
        payload = config_path.read_bytes()
        hashes[str(config_path.resolve())] = sha256(payload)
        (root / "config.yaml").write_bytes(payload)
        outputs.append("config.yaml")
        config = yaml.safe_load(payload)
        if (
            set(config) != {"protocol", "review_path", "review_sha256", "policy"}
            or config["protocol"] != "spambase_training_preflight_v1"
            or config["policy"] != PREFLIGHT_POLICY
        ):
            raise ValueError("frozen preflight schema or policy changed")
        source = load_reviewed_spambase_training(
            Path(config["review_path"]), config["review_sha256"], hashes=hashes
        )
        checks["reviewed_v2_source_and_assignments_verified"] = True
        result = {
            "split_fingerprint": source.split_fingerprint,
            "partition_counts": {p: d["rows"] for p, d in source.partition_counts.items()},
        }
        prepared, files = prepare_training_grid(source, root)
        checks.update(prepared.pop("checks"))
        result.update(prepared)
        outputs.extend(files)
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    try:
        checks["all_inputs_unchanged"] = all(
            sha256(Path(p).read_bytes()) == h for p, h in hashes.items()
        )
    except OSError as error:
        checks["all_inputs_unchanged"] = False
        errors.append(str(error))
    accepted = all(checks.values()) and not errors
    summary = {
        "status": "ok" if accepted else "quality_gate_failed",
        "output": str(root),
        "checks": checks,
        "errors": errors,
        **result,
        "decision": "go_to_frozen_grid_training_protocol" if accepted else "hold_preflight_review",
        "heldout_rows_transformed_or_encoded": 0,
        "optimizer_steps": 0,
        "models_trained": 0,
        "explanations_generated": 0,
        "rows_removed": 0,
        "labels_changed": 0,
        "test_evaluation_ready": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(root / "summary.json", summary)
    outputs.append("summary.json")
    # Also pin any partial diagnostic files retained before an exception.
    outputs = sorted(
        set(outputs) | {p.name for p in root.iterdir() if p.is_file() and p.name != "manifest.json"}
    )
    manifest.update(
        status=summary["status"],
        source_file_sha256=hashes,
        output_file_sha256={n: sha256((root / n).read_bytes()) for n in outputs},
        completed_at_utc=datetime.now(UTC).isoformat(),
    )
    write_json(root / "manifest.json", manifest)
    return summary
