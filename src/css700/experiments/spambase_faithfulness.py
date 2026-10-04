"""One frozen validation-only Grad-CAM faithfulness experiment, independent of effect sign."""

import platform
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml

from css700.evaluation.grouped_deletion import grouped_summaries, plot_deletion_curves
from css700.experiments.som_mapping_audit import _git_state
from css700.experiments.spambase_audit import sha256, write_json
from css700.experiments.spambase_explanation_source import load_deletion_source
from css700.interpretability.spambase_deletion import deletion_rankings, evaluate_deletions

POLICY = {
    "split": "validation",
    "attribution": "accepted_grad_cam_v2_raw_feature_scores",
    "targets": "frozen_predicted_class_logits",
    "rankings": ["descending_raw", "ascending_raw", "group_shared_random"],
    "tie_break": "original_feature_index",
    "deletion_steps": [0, 1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 57],
    "replacements": ["zero", "training_median"],
    "random_repeats": 100,
    "random_seed": 20261004,
    "bootstrap_resamples": 1000,
    "bootstrap_seed": 20261005,
    "confidence_level": 0.95,
    "minimum_interval_groups": 10,
    "uncertainty": "whole_group_paired_conditional_on_fixed_random_controls",
    "group_weighting": "inverse_group_size_within_analysis_subset",
    "batch_size": 32,
    "threads": 1,
    "device": "cpu",
    "nnpack_enabled": True,
    "max_seconds": 300.0,
    "original_logits": "bitwise_exact",
    "padding": "never_mask",
    "test_evaluation": False,
    "refit_or_retrain": False,
}


def validate_config(config):
    if (
        set(config) != {"protocol", "grad_cam_review", "grad_cam_review_sha256", "policy"}
        or config["protocol"] != "spambase_validation_grad_cam_deletion_v1"
        or config["policy"] != POLICY
    ):
        raise ValueError("frozen validation deletion policy changed")


def run_spambase_faithfulness(config_path, output_dir):
    started, (commit, dirty) = time.perf_counter(), _git_state()
    root = Path(output_dir) / (
        datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
        + "__spambase__grid__seed-42__validation-faithfulness-v1"
    )
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "commit_sha": commit,
        "worktree_dirty_at_start": dirty,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": "validation_grad_cam_deletion_test_closed",
        "packages": {n: version(n) for n in ("numpy", "torch", "matplotlib")},
        "platform": platform.platform(),
    }
    write_json(root / "manifest.json", manifest)
    hashes, checks, errors, details = {}, {"clean_worktree": not dirty}, [], {}
    try:
        if dirty:
            raise ValueError("faithfulness requires a clean prospective code/config commit")
        payload = Path(config_path).read_bytes()
        hashes[str(Path(config_path).resolve())] = sha256(payload)
        (root / "config.yaml").write_bytes(payload)
        config = yaml.safe_load(payload)
        validate_config(config)
        source, scores, coordinates, references, groups = load_deletion_source(
            config["grad_cam_review"], config["grad_cam_review_sha256"], hashes
        )
        checks["accepted_checkpoint_and_explanations_verified"] = True
        if not all(sha256(Path(p).read_bytes()) == h for p, h in hashes.items()):
            raise ValueError("source changed before deletion")
        policy = config["policy"]
        rankings = deletion_rankings(
            scores, groups, policy["random_repeats"], policy["random_seed"]
        )
        checks["random_orders_exact_replay"] = np.array_equal(
            rankings,
            deletion_rankings(scores, groups, policy["random_repeats"], policy["random_seed"]),
        )
        steps = policy["deletion_steps"]
        np.savez_compressed(
            root / "design.npz",
            rankings=rankings,
            coordinates=coordinates,
            references=references,
            group_ids=groups,
            sample_ids=source.predictions["sample_ids"],
            labels=source.predictions["labels"],
            targets=source.predictions["predictions"],
            original_logits=source.predictions["logits"],
            feature_scores=scores,
            steps=np.array(steps, dtype=np.int64),
        )
        logits, original, result_checks, counts = evaluate_deletions(
            source,
            rankings,
            coordinates,
            references,
            steps,
            partial_path=root / "perturbed_logits.npy",
            batch_size=policy["batch_size"],
            max_seconds=policy["max_seconds"],
        )
        checks.update(result_checks)
        curves, summaries, aopc = grouped_summaries(
            logits,
            original,
            source.predictions["predictions"],
            source.predictions["labels"],
            groups,
            steps,
            resamples=policy["bootstrap_resamples"],
            seed=policy["bootstrap_seed"],
            minimum_groups=policy["minimum_interval_groups"],
        )
        np.savez_compressed(root / "per_row_aopc.npz", **aopc)
        write_json(root / "curves.json", curves)
        write_json(root / "aopc_summaries.json", summaries)
        plot_deletion_curves(
            root / "deletion_curves.png",
            curves,
            rows=len(groups),
            random_repeats=policy["random_repeats"],
        )
        checks["all_aopc_values_finite"] = all(np.isfinite(v).all() for v in aopc.values())
        ordered_scores = np.take_along_axis(scores, rankings[0], axis=1)
        details.update(
            source_training_run=source.source_run,
            split_fingerprint=source.split_fingerprint,
            evaluated_rows=len(groups),
            feature_groups=len(np.unique(groups)),
            feature_count=57,
            model_seed=42,
            checkpoint_epoch=source.checkpoint["best_epoch"],
            primary_logit_summaries=[
                r for r in summaries if r["group"] == "all" and r["metric"] == "logit"
            ],
            reference_max_abs_difference=float(np.abs(references[1] - references[0]).max()),
            zero_score_rows=int(np.any(scores == 0, axis=1).sum()),
            positive_top5_boundary_ties=int(
                np.sum((ordered_scores[:, 4] == ordered_scores[:, 5]) & (ordered_scores[:, 5] > 0))
            ),
            positive_top10_boundary_ties=int(
                np.sum(
                    (ordered_scores[:, 9] == ordered_scores[:, 10]) & (ordered_scores[:, 10] > 0)
                )
            ),
            **counts,
        )
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    try:
        checks["all_inputs_unchanged"] = all(
            sha256(Path(p).read_bytes()) == h for p, h in hashes.items()
        )
    except OSError as error:
        checks["all_inputs_unchanged"] = False
        errors.append(str(error))
    passed = not errors and all(checks.values())
    summary = {
        "status": "ok" if passed else "quality_gate_failed",
        "output": str(root),
        "checks": checks,
        "errors": errors,
        **details,
        "decision": "await_independent_faithfulness_review"
        if passed
        else "hold_faithfulness_review",
        "effect_direction_is_acceptance_gate": False,
        "test_evaluation_ready": False,
        "test_rows_transformed_or_scored": 0,
        "models_trained": 0,
        "scalers_refitted": 0,
        "optimizer_steps": 0,
        "new_explanations_generated": 0,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(root / "summary.json", summary)
    manifest.update(
        status=summary["status"],
        completed_at_utc=datetime.now(UTC).isoformat(),
        source_file_sha256=hashes,
        output_file_sha256={
            p.name: sha256(p.read_bytes())
            for p in sorted(root.iterdir())
            if p.is_file() and p.name != "manifest.json"
        },
    )
    write_json(root / "manifest.json", manifest)
    return summary
