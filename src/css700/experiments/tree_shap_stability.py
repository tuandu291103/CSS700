"""Separate seed-pair TreeSHAP stability estimates; no pooled independence claim."""

import platform
import sys
import time
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

from css700.evaluation.local_agreement import paired_rank_rows, summarize_rank_rows
from css700.experiments.som_mapping_audit import _git_state
from css700.experiments.tree_shap import _csv, _json
from css700.experiments.tree_shap_sources import file_sha256
from css700.experiments.tree_shap_stability_sources import (
    load_accepted_tree_run,
    scientific_settings,
)


def validate_analysis_config(config: dict) -> None:
    required = {
        "analysis": "tree_shap_fixed_split_training_seed_stability_v1",
        "model_seeds": [7, 42, 2026],
        "families": ["random_forest", "xgboost"],
        "backgrounds": ["train_all", "train_100"],
        "primary_background": "train_all",
        "primary_target": "same_predicted_class",
        "ranking": "absolute",
        "bootstrap_resamples": 1000,
        "bootstrap_seed": 2026,
        "minimum_interval_group": 10,
        "cross_seed_inference": "separate_seed_pairs_no_pooled_intervals",
    }
    if any(config.get(k) != v for k, v in required.items()) or set(config["seed_configs"]) != {
        7,
        42,
        2026,
    }:
        raise ValueError("unsupported or incomplete registered stability settings")


def stability_tables(runs: dict[int, dict], config: dict) -> tuple[list, list, list]:
    if set(runs) != {7, 42, 2026}:
        raise ValueError("stability requires exactly the three registered seeds")
    rows, predictions, consensus = [], [], []
    for family in config["families"]:
        primary = [runs[s]["conditions"][f"{family}__train_all"] for s in config["model_seeds"]]
        matrix = np.stack([c["predictions"] for c in primary])
        consensus.append(
            {
                "family": family,
                "total_samples": matrix.shape[1],
                "all_three_same_prediction": int(np.all(matrix == matrix[0], axis=0).sum()),
            }
        )
        for left_seed, right_seed in combinations(config["model_seeds"], 2):
            left, right = runs[left_seed], runs[right_seed]
            if not np.array_equal(left["sample_ids"], right["sample_ids"]) or not np.array_equal(
                left["labels"], right["labels"]
            ):
                raise ValueError("cross-seed sample/label identity mismatch")
            for bg in config["backgrounds"]:
                a, b = left["conditions"][f"{family}__{bg}"], right["conditions"][f"{family}__{bg}"]
                paired = paired_rank_rows(
                    a["scores"],
                    b["scores"],
                    a["predictions"],
                    b["predictions"],
                    left["labels"],
                    left["sample_ids"],
                    comparison=f"{family}__{bg}__seeds-{left_seed}-{right_seed}",
                )
                rows.extend(
                    [
                        {
                            **r,
                            "family": family,
                            "background": bg,
                            "left_seed": left_seed,
                            "right_seed": right_seed,
                        }
                        for r in paired
                    ]
                )
                if bg == "train_all":
                    same = a["predictions"] == b["predictions"]
                    predictions.append(
                        {
                            "family": family,
                            "left_seed": left_seed,
                            "right_seed": right_seed,
                            "total_samples": len(same),
                            "same_prediction_samples": int(same.sum()),
                            "prediction_agreement": float(same.mean()),
                            "max_abs_probability_difference": float(
                                abs(a["probabilities"] - b["probabilities"]).max()
                            ),
                        }
                    )
    return rows, predictions, consensus


def _figure(path: Path, summaries: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    rows = [r for r in summaries if r["group"] == "all" and "__train_all__" in r["comparison"]]
    fig, axis = plt.subplots(figsize=(10, 4.5))
    labels = []
    for i, row in enumerate(rows):
        name = (
            row["comparison"]
            .replace("random_forest__train_all__seeds-", "RF: ")
            .replace("xgboost__train_all__seeds-", "XGB: ")
        )
        labels.append(f"{name} (n={row['eligible_samples']})")
        metric = row["spearman"]
        if metric["mean"] is not None:
            axis.plot(metric["mean"], i, "o", color="tab:blue")
            if metric["mean_95_percentile_interval"] is not None:
                axis.plot(metric["mean_95_percentile_interval"], [i, i], color="tab:blue")
    axis.set_yticks(range(len(rows)), labels)
    axis.set(
        xlim=(-1.05, 1.05),
        xlabel="Mean absolute-score Spearman; 95% paired-sample bootstrap interval",
        title=(
            "Local TreeSHAP training-seed stability\n"
            "Fixed split and 397-row background; separate seed-pair estimates"
        ),
    )
    axis.invert_yaxis()
    axis.axvline(0, color="gray", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run_tree_shap_stability(run_dirs: list[Path], config_path: Path, output_dir: Path) -> dict:
    started = time.perf_counter()
    commit, dirty = _git_state()
    config = yaml.safe_load(config_path.read_text())
    validate_analysis_config(config)
    root = output_dir / (
        datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
        + "__breast-cancer__tree-shap__seeds-7-42-2026__stability"
    )
    root.mkdir(parents=True, exist_ok=False)
    (root / "config.yaml").write_bytes(config_path.read_bytes())
    hashes = {str(config_path.resolve()): file_sha256(config_path)}
    checks = {"clean_worktree": not dirty, "no_cnn_runtime": "torch" not in sys.modules}
    manifest = {
        "run_id": root.name,
        "commit_sha": commit,
        "worktree_dirty_at_start": dirty,
        "config_sha256": file_sha256(config_path),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "status": "running",
        "committee_approval_claimed": False,
        "input_run_dirs": [str(p.resolve()) for p in run_dirs],
    }
    _json(root / "manifest.json", manifest)
    errors, runs = [], {}
    try:
        if not all(checks.values()):
            raise ValueError("analysis requires a clean commit and no CNN runtime")
        if len(run_dirs) != 3 or len({p.resolve() for p in run_dirs}) != 3:
            raise ValueError("provide exactly three distinct accepted TreeSHAP runs")
        for path in run_dirs:
            source_config = yaml.safe_load((path / "config.yaml").read_text())
            seed = source_config["model_seed"]
            if seed not in config["model_seeds"] or seed in runs:
                raise ValueError("missing, duplicate or unregistered model seed")
            if seed == 42 and path.resolve() != Path(config["reference_run"]).resolve():
                raise ValueError("seed-42 run is not the registered accepted reference")
            runs[seed] = load_accepted_tree_run(path, Path(config["seed_configs"][seed]), hashes)
        reference = runs[42]
        for run in runs.values():
            if scientific_settings(run["config"]) != scientific_settings(reference["config"]):
                raise ValueError("scientific settings changed between model seeds")
            if run["manifest"]["packages"] != reference["manifest"]["packages"]:
                raise ValueError("source explanation package versions differ")
            for bg in config["backgrounds"]:
                if any(
                    not np.array_equal(run["backgrounds"][bg][k], reference["backgrounds"][bg][k])
                    for k in ("values", "sample_ids")
                ):
                    raise ValueError("background rows/values/order differ across seeds")
        rows, predictive, consensus = stability_tables(runs, config)
        summaries = summarize_rank_rows(
            rows,
            resamples=config["bootstrap_resamples"],
            seed=config["bootstrap_seed"],
            min_group=config["minimum_interval_group"],
        )
        checks["all_twelve_conditions_revalidated"] = (
            sum(len(r["conditions"]) for r in runs.values()) == 12
        )
        checks["all_twelve_seed_pair_comparisons"] = len({r["comparison"] for r in rows}) == 12
        _csv(root / "paired_seed_agreement.csv", rows)
        _csv(root / "prediction_seed_agreement.csv", predictive)
        _json(root / "prediction_consensus.json", consensus)
        _json(root / "agreement_summary.json", summaries)
        manifest["source_commits"] = {s: r["manifest"]["commit_sha"] for s, r in runs.items()}
        manifest["source_packages"] = reference["manifest"]["packages"]
        _figure(root / "seed_stability.png", summaries)
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    try:
        checks["input_files_unchanged"] = all(
            file_sha256(Path(p)) == value for p, value in hashes.items()
        )
    except OSError as error:
        checks["input_files_unchanged"] = False
        errors.append(f"input file recheck failed: {error}")
    checks["no_cnn_runtime"] = "torch" not in sys.modules
    summary = {
        "status": "ok" if all(checks.values()) and not errors else "quality_gate_failed",
        "output": str(root),
        "checks": checks,
        "errors": errors,
        "runtime_seconds": time.perf_counter() - started,
        "interpretation": (
            "Separate same-target seed-pair estimates; no pooled sample/seed "
            "independence or explainer-superiority claim."
        ),
    }
    manifest.update(
        status=summary["status"],
        source_file_sha256=hashes,
        completed_at_utc=datetime.now(UTC).isoformat(),
    )
    _json(root / "manifest.json", manifest)
    _json(root / "summary.json", summary)
    return summary
