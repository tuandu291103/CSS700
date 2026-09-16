"""Append-only execution of the registered seed-42 LIME test pilot."""

import csv
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import yaml
from threadpoolctl import threadpool_limits

from css700.evaluation.lime_comparison import (
    comparison_tables,
    feature_rows,
    fit_quality_summary,
)
from css700.evaluation.local_agreement import summarize_rank_rows
from css700.experiments.lime_preflight import _save_design, validate_lime_config
from css700.experiments.lime_test_sources import PREFLIGHT, load_test_sources, verify_predictions
from css700.experiments.som_mapping_audit import _git_state
from css700.experiments.tree_shap import _csv, _json
from css700.experiments.tree_shap_sources import file_sha256, named_feature_matrix
from css700.interpretability.lime_tabular import (
    build_lime_design,
    fit_diagnostics,
    fit_probability_lime,
)
from css700.interpretability.tree_shap import binary_probabilities


def verify_fit(result: dict, design, expected: dict, target: int) -> None:
    """Independently solve the surrogate and check all stored fit/evaluation metrics."""
    diag = result["diagnostics"]
    p = result["probabilities"]
    if diag["target_class"] != target:
        raise ValueError("LIME surrogate target changed")
    verify_predictions(p[:1], expected)
    x, w = design.fit.design, design.fit.weights
    y = p[: len(x), target]
    xm, ym = np.average(x, weights=w, axis=0), np.average(y, weights=w)
    xc = x - xm
    beta = np.linalg.solve(xc.T @ (w[:, None] * xc) + np.eye(x.shape[1]), xc.T @ (w * (y - ym)))
    intercept = ym - xm @ beta
    if not np.allclose(beta, result["coefficients"], atol=1e-10, rtol=1e-8) or not np.isclose(
        intercept, diag["intercept"], atol=1e-10, rtol=1e-8
    ):
        raise ValueError("independent weighted Ridge reconstruction failed")
    for role, actual in (("fit", y), ("evaluation", p[len(x) :, target])):
        neighborhood = getattr(design, role)
        prediction = result[f"{role}_surrogate_predictions"]
        if (
            not np.allclose(
                intercept + neighborhood.design @ beta, prediction, atol=1e-10, rtol=1e-8
            )
            or fit_diagnostics(actual, prediction, neighborhood.weights) != diag[role]
        ):
            raise ValueError("independent LIME predictions/diagnostics mismatch")
    anchor = float(result["fit_surrogate_predictions"][0])
    if (
        diag["anchor_model_probability"] != float(p[0, target])
        or diag["anchor_surrogate_prediction"] != anchor
        or diag["anchor_absolute_error"] != abs(anchor - float(p[0, target]))
    ):
        raise ValueError("LIME anchor diagnostic mismatch")


def run_lime_test(config_path: Path, output_dir: Path) -> dict:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text())
    validate_lime_config(config)
    commit, dirty = _git_state()
    root = output_dir / (
        datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
        + "__breast-cancer__lime__model-seed-42__test-pilot"
    )
    root.mkdir(parents=True, exist_ok=False)
    (root / "config.yaml").write_bytes(config_path.read_bytes())
    hashes = {str(config_path.resolve()): file_sha256(config_path)}
    manifest = {
        "run_id": root.name,
        "commit_sha": commit,
        "worktree_dirty_at_start": dirty,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_sha256": file_sha256(config_path),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "scope": "frozen_v1_test_pilot_no_tuning_or_fit_quality_filter",
        "committee_approval_claimed": False,
        "status": "running",
    }
    _json(root / "manifest.json", manifest)
    checks = {"clean_worktree": not dirty, "no_cnn_runtime": "torch" not in sys.modules}
    errors, records, features, rows = [], [], [], []
    generated = verified = neighborhoods = 0
    try:
        if not all(checks.values()):
            raise ValueError("LIME test pilot requires a clean commit and no CNN runtime")
        if file_sha256(config_path) != PREFLIGHT["preflight_config_sha256"]:
            raise ValueError("test settings differ from the original frozen LIME config")
        sources = load_test_sources(config, hashes)
        bundle, shap = sources["bundle"], sources["shap"]
        checks["accepted_preflight_and_shap_revalidated"] = True
        manifest.update(
            packages=sources["preflight_manifest"]["packages"],
            lime_source_sha256=sources["preflight_manifest"]["lime_source_sha256"],
            source_preflight=PREFLIGHT["preflight_run"],
            source_shap=config["accepted_shap_run"],
            source_commits={
                "preflight": sources["preflight_manifest"]["commit_sha"],
                "shap": shap["manifest"]["commit_sha"],
                "tree": sources["tree_manifest"]["commit_sha"],
            },
            feature_names=bundle.feature_names,
            class_names=bundle.target_names,
            threadpool_limit=1,
            expected_test_fits=516,
            expected_neighborhood_pairs=258,
        )
        _json(root / "manifest.json", manifest)
        np.savez_compressed(
            root / "splits.npz",
            train=bundle.train_indices,
            validation=bundle.validation_indices,
            test=bundle.test_indices,
        )
        np.savez_compressed(
            root / "preprocessing.npz", mean=bundle.scaler_mean, scale=bundle.scaler_scale
        )
        with threadpool_limits(limits=1):
            models = {f: joblib.load(sources["model_paths"][f]) for f in config["families"]}
            reload_checks = {}
            for family, model in models.items():
                probabilities = binary_probabilities(model, bundle.x_test)
                reload_checks[family] = verify_predictions(
                    probabilities, sources["tree_predictions"][family]
                )
            _json(root / "model_reload.json", reload_checks)
            checks["original_test_predictions_reproduced"] = True
            for i, sample in enumerate(bundle.test_indices):
                sample_id = int(sample)
                for seed in config["explainer_seeds"]:
                    design = build_lime_design(
                        bundle.x_train,
                        bundle.train_indices,
                        np.concatenate((bundle.validation_indices, bundle.test_indices)),
                        bundle.x_test[i],
                        bundle.feature_names,
                        sample_id=sample_id,
                        explainer_seed=seed,
                        fit_samples=config["fit_samples"],
                        evaluation_samples=config["evaluation_samples"],
                    )
                    case = root / f"sample-{sample_id}__explainer-seed-{seed}"
                    _save_design(case, design)
                    neighborhoods += 1
                    for family, model in models.items():
                        fit_started = time.perf_counter()
                        saved = sources["tree_predictions"][family]
                        target = int(saved["predictions"][i])
                        result = fit_probability_lime(model, design, target=target)
                        generated += 1
                        # Persist raw evidence before the independent verification gate.
                        np.savez_compressed(
                            case / f"{family}.npz",
                            **{k: v for k, v in result.items() if k != "diagnostics"},
                        )
                        record = {
                            "sample_id": sample_id,
                            "family": family,
                            "explainer_seed": seed,
                            "partition": "test",
                            "true_label": int(bundle.y_test[i]),
                            "runtime_seconds_including_export": time.perf_counter() - fit_started,
                            **result["diagnostics"],
                        }
                        _json(case / f"{family}.json", record)
                        verify_fit(
                            result,
                            design,
                            {k: v[i : i + 1] for k, v in saved.items()},
                            target,
                        )
                        verified += 1
                        records.append(record)
                        features.extend(
                            feature_rows(record, result["coefficients"], design.metadata)
                        )
                _json(
                    root / "progress.json",
                    {
                        "completed_test_samples": i + 1,
                        "verified_fits": verified,
                        "expected_test_samples": len(bundle.test_indices),
                    },
                )
            checks["all_516_fits_verified"] = generated == verified == len(records) == 516
            checks["all_258_shared_neighborhood_pairs"] = neighborhoods == 258
            if not all(checks.values()):
                raise ValueError("incomplete LIME test pilot")
            _csv(root / "target_features.csv", features)
            # Re-read the named export to check serialized ID/order/value fidelity.
            with (root / "target_features.csv").open(newline="") as handle:
                exported = list(csv.DictReader(handle))
            for family in config["families"]:
                for seed in config["explainer_seeds"]:

                    def select(r, f=family, s=seed):
                        return r["family"] == f and int(r["explainer_seed"]) == s

                    matrices = [
                        named_feature_matrix(
                            [r for r in collection if select(r)],
                            bundle.test_indices,
                            bundle.feature_names,
                            sources["tree_predictions"][family]["predictions"],
                            bundle.y_test,
                            value_field="signed_coefficient",
                        )
                        for collection in (features, exported)
                    ]
                    if not np.array_equal(*matrices):
                        raise ValueError("named-feature CSV round-trip mismatch")
            checks["named_feature_export_round_trip"] = len(exported) == 15480
            rows = comparison_tables(features, records, shap)
            summaries = summarize_rank_rows(
                rows,
                resamples=config["bootstrap_resamples"],
                seed=config["bootstrap_seed"],
                min_group=config["minimum_interval_group"],
            )
            checks["all_15_comparisons_complete"] = len(rows) == 1290 and len(summaries) == 90
            _csv(root / "paired_agreement.csv", rows)
            _json(root / "paired_agreement.json", rows)
            _json(root / "agreement_summary.json", summaries)
            _json(root / "fit_quality_summary.json", fit_quality_summary(records))
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    try:
        checks["source_files_unchanged"] = all(file_sha256(Path(p)) == h for p, h in hashes.items())
    except OSError as error:
        checks["source_files_unchanged"] = False
        errors.append(str(error))
    checks["no_cnn_runtime"] = "torch" not in sys.modules
    summary = {
        "status": "ok" if all(checks.values()) and not errors else "quality_gate_failed",
        "output": str(root),
        "checks": checks,
        "errors": errors,
        "generated_test_fits": generated,
        "verified_test_fits": verified,
        "shared_neighborhood_pairs": neighborhoods,
        "named_feature_rows": len(features),
        "paired_comparison_rows": len(rows),
        "runtime_seconds": time.perf_counter() - started,
        "interpretation": (
            "Technical acceptance only; report all fit errors, separate sampling seeds, "
            "no superiority claim."
        ),
    }
    manifest.update(
        status=summary["status"],
        completed_at_utc=datetime.now(UTC).isoformat(),
        source_file_sha256=hashes,
    )
    _json(root / "test_diagnostics.json", records)
    _json(root / "manifest.json", manifest)
    _json(root / "summary.json", summary)
    return summary
