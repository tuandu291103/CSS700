"""Read-only independent Occlusion replay, manual tied ranks and group statistics audit."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from verify_spambase_grad_cam_v2 import Replica, digest, exact


def compare(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            compare(actual[key], expected[key])
    elif isinstance(expected, (float, np.floating)):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            compare(a, b)
    else:
        assert actual == expected, (actual, expected)


def probability(x):
    x = x.astype(np.float64)
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def manual_ranks(values):
    # Independent average ranks from counts, not scipy's ranking/correlation routines.
    return np.array([1 + np.sum(values < v) + (np.sum(values == v) - 1) / 2 for v in values])


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    config = yaml.safe_load((root / "config.yaml").read_text())
    policy = config["policy"]
    for path, expected in manifest["source_file_sha256"].items():
        assert digest(Path(path)) == expected, path
    for name, expected in manifest["output_file_sha256"].items():
        assert digest(root / name) == expected, name
    assert summary["status"] == "ok" and not summary["errors"] and all(summary["checks"].values())
    with np.load(root / "design.npz", allow_pickle=False) as f:
        a = {k: f[k] for k in f.files}
    training = Path(summary["source_training_run"])
    with np.load(training / "inputs.npz", allow_pickle=False) as f:
        images, train, ids, labels = (
            f["validation_images"],
            f["train_images"],
            f["validation_indices"],
            f["y_validation"],
        )
    with np.load(training / "validation_predictions.npz", allow_pickle=False) as f:
        original, targets = f["logits"], f["predictions"]
    mapping = json.loads((training / "feature_map.json").read_text())
    assert json.loads((root / "feature_map.json").read_text()) == mapping
    coordinates = np.array(
        [r["row"] * 8 + r["column"] for r in sorted(mapping, key=lambda r: r["feature_index"])]
    )
    refs = np.stack([np.zeros((1, 8, 8), np.float32), np.median(train, axis=0)])
    row_path = next(
        Path(p) for p in manifest["source_file_sha256"] if Path(p).name == "row_assignments.json"
    )
    rows = {
        r["row_id"]: r for r in json.loads(row_path.read_text()) if r["partition"] == "validation"
    }
    groups = np.array([rows[int(i)]["group_id"] for i in ids])
    cam_review = yaml.safe_load(Path(config["grad_cam_review"]).read_text())
    with np.load(Path(cam_review["run_dir"]) / "grad_cam_arrays.npz", allow_pickle=False) as f:
        cam = f["feature_scores"]
    for key, expected in (
        ("sample_ids", ids),
        ("labels", labels),
        ("targets", targets),
        ("original_logits", original),
        ("coordinates", coordinates),
        ("references", refs),
        ("group_ids", groups),
        ("grad_cam_scores", cam),
    ):
        exact(a[key], expected)
    logits = np.load(root / "perturbed_logits.npy", mmap_mode="r", allow_pickle=False)
    assert logits.shape == (2, len(ids), 57, 2)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    net = Replica().eval()
    net.load_state_dict(
        torch.load(training / "best_checkpoint.pt", weights_only=True, map_location="cpu")[
            "model_state_dict"
        ]
    )
    calls = 0

    def forward(x):
        nonlocal calls
        parts = []
        for start in range(0, len(x), policy["batch_size"]):
            parts.append(
                net.head(
                    net.features[:3](
                        torch.from_numpy(x[start : start + policy["batch_size"]].copy())
                    )
                ).numpy()
            )
            calls += 1
        return np.concatenate(parts)

    with torch.no_grad(), torch.backends.nnpack.flags(enabled=True):
        exact(forward(images), original)
        for r, ref in enumerate(refs):
            for j, coordinate in enumerate(coordinates):
                mask = (np.arange(64) == coordinate).reshape(1, 8, 8)
                assert mask.sum() == 1 and np.flatnonzero(mask)[0] in coordinates
                values = np.where(mask, ref, images)
                exact(values[:, ~mask], images[:, ~mask])
                exact(forward(values), logits[r, :, j])
    assert calls == summary["forward_calls"]
    deletion_review = yaml.safe_load(Path(config["faithfulness_review"]).read_text())
    deletion = Path(deletion_review["run_dir"])
    assert str(deletion) == summary["source_deletion_run"]
    with np.load(deletion / "design.npz", allow_pickle=False) as f:
        old = {k: f[k] for k in f.files}
    for key, expected in (
        ("sample_ids", ids),
        ("labels", labels),
        ("targets", targets),
        ("original_logits", original),
        ("coordinates", coordinates),
        ("references", refs),
        ("group_ids", groups),
        ("feature_scores", cam),
    ):
        exact(old[key], expected)
    prior = np.load(deletion / "perturbed_logits.npy", mmap_mode="r", allow_pickle=False)
    k1 = old["steps"].tolist().index(1)
    for r in range(2):
        for i, order in enumerate(old["rankings"]):
            exact(logits[r, np.arange(len(ids)), order[:, 0]], prior[r, i, k1])
    effects = {
        "logit": original[np.arange(len(ids)), targets].astype(np.float64)[None, :, None]
        - np.take_along_axis(logits.astype(np.float64), targets[None, :, None, None], -1)[..., 0],
        "probability": probability(original)[np.arange(len(ids)), targets][None, :, None]
        - np.take_along_axis(probability(logits), targets[None, :, None, None], -1)[..., 0],
    }
    with np.load(root / "effects.npz", allow_pickle=False) as f:
        for key, value in effects.items():
            np.testing.assert_allclose(f[key], value, rtol=1e-12, atol=1e-12)
    names = (
        "occlusion_zero_vs_grad_cam",
        "occlusion_median_vs_grad_cam",
        "occlusion_zero_vs_median",
    )
    records = json.loads((root / "rank_records.json").read_text())
    expected_records = []
    for name, left, right in zip(
        names,
        (effects["logit"][0], effects["logit"][1], effects["logit"][0]),
        (cam, cam, effects["logit"][1]),
        strict=True,
    ):
        for i, (x, y) in enumerate(zip(abs(left), abs(right), strict=True)):
            constant = bool(np.all(x == x[0]) or np.all(y == y[0]))
            u, v = manual_ranks(x), manual_ranks(y)
            u, v = u - u.mean(), v - v.mean()
            row = {
                "comparison": name,
                "sample_id": int(ids[i]),
                "true_label": int(labels[i]),
                "left_prediction": int(targets[i]),
                "right_prediction": int(targets[i]),
                "same_target": True,
                "constant_vector": constant,
                "primary_eligible": not constant,
                "exclusion_reason": "constant_vector" if constant else "",
                "correctness_group": "both_correct"
                if targets[i] == labels[i]
                else "neither_correct",
                "spearman": None if constant else float(u @ v / np.sqrt((u @ u) * (v @ v))),
            }
            for k in (5, 10):
                order_x = np.lexsort((np.arange(len(x)), -x))
                order_y = np.lexsort((np.arange(len(y)), -y))
                intersection = len(set(order_x[:k]) & set(order_y[:k]))
                row.update(
                    {
                        f"left_top_{k}_cutoff_tie": bool(x[order_x[k - 1]] == x[order_x[k]]),
                        f"right_top_{k}_cutoff_tie": bool(y[order_y[k - 1]] == y[order_y[k]]),
                        f"top_{k}_intersection": None if constant else intersection,
                        f"top_{k}_overlap_fraction": None if constant else intersection / k,
                        f"top_{k}_jaccard": None
                        if constant
                        else intersection / (2 * k - intersection),
                    }
                )
            expected_records.append(row)
    compare(records, expected_records)
    summaries = json.loads((root / "rank_summaries.json").read_text())
    subsets = {
        "all": np.ones(len(ids), bool),
        "true_0": labels == 0,
        "true_1": labels == 1,
        "predicted_0": targets == 0,
        "predicted_1": targets == 1,
        "correct": targets == labels,
        "incorrect": targets != labels,
    }
    rng, checked = np.random.default_rng(policy["bootstrap_seed"]), 0
    for name, subset in subsets.items():
        keys, codes = np.unique(groups[subset], return_inverse=True)
        if not len(keys):
            continue
        draws = rng.integers(len(keys), size=(policy["bootstrap_resamples"], len(keys)))
        for entry in (r for r in summaries if r["group"] == name):
            pair = [r for r in expected_records if r["comparison"] == entry["comparison"]]
            selected = [r for r, keep in zip(pair, subset, strict=True) if keep]
            eligible = np.array([r["primary_eligible"] for r in selected])
            counts = np.array([np.sum((codes == g) & eligible) for g in range(len(keys))])
            assert entry["total_rows"] == subset.sum() and entry["total_feature_groups"] == len(
                keys
            )
            assert entry["eligible_rows"] == eligible.sum() and entry[
                "eligible_feature_groups"
            ] == np.sum(counts > 0)
            assert entry["constant_vector_rows"] == len(selected) - eligible.sum()
            compare(
                entry["cutoff_tie_rows_eligible"],
                {
                    f"{side}_top_{k}": sum(
                        r["primary_eligible"] and r[f"{side}_top_{k}_cutoff_tie"] for r in selected
                    )
                    for side in ("left", "right")
                    for k in (5, 10)
                },
            )
            for metric in ("spearman", "top_5_overlap_fraction", "top_10_overlap_fraction"):
                sums = np.array(
                    [
                        sum(
                            r[metric]
                            for i, r in enumerate(selected)
                            if eligible[i] and codes[i] == g
                        )
                        for g in range(len(keys))
                    ],
                    dtype=float,
                )
                mass = counts.copy().astype(float)
                if entry["weighting"] == "group_equal":
                    sums = np.array(
                        [
                            value / count if count else 0
                            for value, count in zip(sums, counts, strict=True)
                        ]
                    )
                    mass = (counts > 0).astype(float)
                den = mass[draws].sum(1)
                valid = den > 0
                distribution = sums[draws[valid]].sum(1) / den[valid]
                compare(
                    entry[metric],
                    {
                        "mean": float(sums.sum() / mass.sum()) if mass.sum() else None,
                        "ci95": np.quantile(distribution, [0.025, 0.975]).tolist()
                        if np.sum(counts > 0) >= policy["minimum_interval_groups"] and valid.all()
                        else None,
                        "valid_bootstrap_draws": int(valid.sum()),
                    },
                )
            checked += 1
    assert checked == len(summaries)
    compare(summary["all_row_rank_summaries"], [r for r in summaries if r["group"] == "all"])
    compare(
        summary["effect_sign_counts"],
        {
            ref: {
                "negative": int((effects["logit"][r] < 0).sum()),
                "zero": int((effects["logit"][r] == 0).sum()),
                "positive": int((effects["logit"][r] > 0).sum()),
            }
            for r, ref in enumerate(("zero", "training_median"))
        },
    )
    compare(
        summary["unchanged_input_feature_values"],
        [
            int(
                (images.reshape(len(ids), 64)[:, coordinates] == ref.reshape(64)[coordinates]).sum()
            )
            for ref in refs
        ],
    )
    return {
        "status": "independent_full_replay_passed",
        "input_hashes_verified": len(manifest["source_file_sha256"]),
        "manifest_output_hashes_verified": len(manifest["output_file_sha256"]),
        "forward_calls_replayed": calls,
        "all_perturbation_and_k1_logits_bitwise_exact": True,
        "rank_records_verified": len(records),
        "group_summary_records_verified": checked,
        "metric_means_and_intervals_verified": checked * 3,
        "execution_commit": manifest["commit_sha"],
        "file_sha256": {p.name: digest(p) for p in sorted(root.iterdir()) if p.is_file()},
    }


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1])), indent=2))
