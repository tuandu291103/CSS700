"""One prospective, validation-only Occlusion run and fixed Grad-CAM comparison."""

import io
import json
import platform
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml

from css700.evaluation.grouped_occlusion import grouped_rank_summary, plot_agreement, rank_records
from css700.experiments.som_mapping_audit import _git_state
from css700.experiments.spambase_audit import sha256, write_json
from css700.experiments.spambase_explanation_source import load_deletion_source
from css700.experiments.spambase_faithfulness import validate_config as validate_deletion_config
from css700.experiments.spambase_split import verified_read
from css700.interpretability.activation_boundary import bitwise_equal
from css700.interpretability.spambase_occlusion import evaluate_occlusion, signed_effects

POLICY = {
    "split": "validation",
    "targets": "frozen_predicted_class_logits",
    "feature_unit": "one_complete_mapped_feature",
    "replacements": ["zero", "training_median"],
    "primary_score": "signed_target_logit_drop",
    "secondary_score": "signed_target_probability_drop_float64",
    "ranking": "absolute_logit_drop",
    "tie_break": "original_feature_index",
    "constant_vectors": "retain_rows_with_undefined_agreement",
    "top_k": [5, 10],
    "bootstrap_resamples": 1000,
    "bootstrap_seed": 20261006,
    "confidence_level": 0.95,
    "minimum_interval_groups": 10,
    "uncertainty": "paired_whole_feature_groups_fixed_model",
    "batch_size": 32,
    "threads": 1,
    "device": "cpu",
    "nnpack_enabled": True,
    "max_seconds": 120.0,
    "original_and_deletion_k1_logits": "bitwise_exact",
    "padding": "never_mask",
    "test_evaluation": False,
    "refit_or_retrain": False,
}
DELETION_FILES = {
    "aopc_summaries.json",
    "config.yaml",
    "curves.json",
    "deletion_curves.png",
    "design.npz",
    "manifest.json",
    "per_row_aopc.npz",
    "perturbed_logits.npy",
    "summary.json",
}


def validate_config(config):
    if (
        set(config)
        != {
            "protocol",
            "grad_cam_review",
            "grad_cam_review_sha256",
            "faithfulness_review",
            "faithfulness_review_sha256",
            "policy",
        }
        or config["protocol"] != "spambase_validation_occlusion_v1"
        or config["policy"] != POLICY
    ):
        raise ValueError("frozen validation Occlusion policy changed")


def load_deletion_consistency(config, source, cam, coordinates, references, groups, hashes):
    review = yaml.safe_load(
        verified_read(
            Path(config["faithfulness_review"]), config["faithfulness_review_sha256"], hashes
        )
    )
    if (
        review.get("review_version") != "spambase_validation_grad_cam_deletion_v1_review_1"
        or review.get("acceptance") is not True
        or review.get("acceptance_scope")
        != "seed_42_validation_grad_cam_conditional_deletion_evidence_only"
        or review.get("technical_status") != "ok"
        or review.get("decision") != "go_to_prospective_input_level_explanation_protocol"
        or review.get("test_evaluation_ready") is not False
        or review.get("independent_calculation", {}).get("status") != "passed"
        or set(review.get("file_sha256", {})) != DELETION_FILES
        or any(review.get(k) != config[k] for k in ("grad_cam_review", "grad_cam_review_sha256"))
    ):
        raise ValueError("affirmative pinned deletion review with same Grad-CAM source required")
    root = Path(review["run_dir"])
    saved = {n: verified_read(root / n, h, hashes) for n, h in review["file_sha256"].items()}
    manifest, summary = (json.loads(saved[n]) for n in ("manifest.json", "summary.json"))
    if (
        manifest["status"] != "ok"
        or manifest["worktree_dirty_at_start"]
        or manifest["commit_sha"] != review["execution_commit"]
        or summary["status"] != "ok"
        or summary["errors"]
        or not all(summary["checks"].values())
        or summary["source_training_run"] != source.source_run
        or summary["test_rows_transformed_or_scored"] != 0
        or summary["test_evaluation_ready"] is not False
        or manifest["output_file_sha256"]
        != {n: h for n, h in review["file_sha256"].items() if n != "manifest.json"}
    ):
        raise ValueError("deletion review/run identity mismatch")
    for path, digest in manifest["source_file_sha256"].items():
        verified_read(Path(path), digest, hashes)
    prior_config = verified_read(
        Path(review["faithfulness_config"]), review["faithfulness_config_sha256"], hashes
    )
    validate_deletion_config(yaml.safe_load(prior_config))
    if prior_config != saved["config.yaml"]:
        raise ValueError("deletion config mismatch")
    with np.load(io.BytesIO(saved["design.npz"]), allow_pickle=False) as f:
        design = {k: f[k] for k in f.files}
    for key, values in (
        ("sample_ids", source.predictions["sample_ids"]),
        ("labels", source.predictions["labels"]),
        ("targets", source.predictions["predictions"]),
        ("original_logits", source.predictions["logits"]),
        ("feature_scores", cam),
        ("coordinates", coordinates),
        ("references", references),
        ("group_ids", groups),
    ):
        if not bitwise_equal(design[key], values):
            raise ValueError(f"deletion design identity mismatch: {key}")
    if np.sum(design["steps"] == 1) != 1:
        raise ValueError("prior deletion must contain exactly one k=1 step")
    return root, design, np.load(io.BytesIO(saved["perturbed_logits.npy"]), allow_pickle=False)


def check_deletion_k1(logits, design, previous):
    j = int(np.flatnonzero(design["steps"] == 1)[0])
    return all(
        bitwise_equal(logits[r, np.arange(logits.shape[1]), order[:, 0]], previous[r, i, j])
        for r in range(2)
        for i, order in enumerate(design["rankings"])
    )


def run_spambase_occlusion(config_path, output_dir):
    started, (commit, dirty) = time.perf_counter(), _git_state()
    root = Path(output_dir) / (
        datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
        + "__spambase__grid__seed-42__validation-occlusion-v1"
    )
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "commit_sha": commit,
        "worktree_dirty_at_start": dirty,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": "validation_occlusion_rank_comparison_test_closed",
        "packages": {n: version(n) for n in ("numpy", "torch", "scipy", "matplotlib")},
        "platform": platform.platform(),
    }
    write_json(root / "manifest.json", manifest)
    hashes, checks, errors, details = {}, {"clean_worktree": not dirty}, [], {}
    try:
        if dirty:
            raise ValueError("Occlusion requires a clean prospective code/config commit")
        payload = Path(config_path).read_bytes()
        hashes[str(Path(config_path).resolve())] = sha256(payload)
        (root / "config.yaml").write_bytes(payload)
        config = yaml.safe_load(payload)
        validate_config(config)
        source, cam, coordinates, references, groups = load_deletion_source(
            config["grad_cam_review"], config["grad_cam_review_sha256"], hashes
        )
        deletion_root, prior_design, prior_logits = load_deletion_consistency(
            config, source, cam, coordinates, references, groups, hashes
        )
        checks["accepted_sources_verified"] = True
        if not all(sha256(Path(p).read_bytes()) == h for p, h in hashes.items()):
            raise ValueError("source changed before Occlusion")
        p = source.predictions
        np.savez_compressed(
            root / "design.npz",
            sample_ids=p["sample_ids"],
            labels=p["labels"],
            targets=p["predictions"],
            original_logits=p["logits"],
            group_ids=groups,
            coordinates=coordinates,
            references=references,
            grad_cam_scores=cam,
        )
        write_json(root / "feature_map.json", source.feature_map)
        policy = config["policy"]
        logits, original, result_checks, counts = evaluate_occlusion(
            source,
            coordinates,
            references,
            partial_path=root / "perturbed_logits.npy",
            batch_size=policy["batch_size"],
            max_seconds=policy["max_seconds"],
        )
        checks.update(result_checks)
        checks["every_prior_deletion_k1_logit_exact"] = check_deletion_k1(
            logits, prior_design, prior_logits
        )
        effects = signed_effects(logits, original, p["predictions"])
        np.savez_compressed(root / "effects.npz", **effects)
        records = rank_records(
            effects["logit"], cam, p["predictions"], p["labels"], p["sample_ids"]
        )
        summaries = grouped_rank_summary(
            records,
            groups,
            p["labels"],
            p["predictions"],
            resamples=policy["bootstrap_resamples"],
            seed=policy["bootstrap_seed"],
            minimum_groups=policy["minimum_interval_groups"],
        )
        write_json(root / "rank_records.json", records)
        write_json(root / "rank_summaries.json", summaries)
        plot_agreement(root / "agreement_histograms.png", records)
        checks["all_signed_effects_finite"] = all(np.isfinite(v).all() for v in effects.values())
        details.update(
            source_training_run=source.source_run,
            source_deletion_run=str(deletion_root),
            split_fingerprint=source.split_fingerprint,
            evaluated_rows=len(groups),
            feature_groups=len(np.unique(groups)),
            feature_count=57,
            model_seed=42,
            checkpoint_epoch=source.checkpoint["best_epoch"],
            all_row_rank_summaries=[r for r in summaries if r["group"] == "all"],
            effect_sign_counts={
                ref: {
                    name: int(f(effects["logit"][r]).sum())
                    for name, f in (
                        ("negative", lambda a: a < 0),
                        ("zero", lambda a: a == 0),
                        ("positive", lambda a: a > 0),
                    )
                }
                for r, ref in enumerate(("zero", "training_median"))
            },
            unchanged_input_feature_values=[
                int(
                    (
                        source.arrays["validation_images"].reshape(len(groups), 64)[:, coordinates]
                        == ref.reshape(64)[coordinates]
                    ).sum()
                )
                for ref in references
            ],
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
        "decision": "await_independent_occlusion_review" if passed else "hold_occlusion_review",
        "agreement_magnitude_is_acceptance_gate": False,
        "test_evaluation_ready": False,
        "test_rows_transformed_or_scored": 0,
        "models_trained": 0,
        "scalers_refitted": 0,
        "optimizer_steps": 0,
        "integrated_gradients_rows": 0,
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
