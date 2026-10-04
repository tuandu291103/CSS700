"""Independent read-only full perturbation replay and cluster-statistics audit.

Usage: .venv/bin/python scripts/verify_spambase_faithfulness.py artifacts/<run>
Imports the independently reconstructed CNN, never css700 model/deletion/statistics code.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from verify_spambase_grad_cam_v2 import Replica, digest, exact


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


def probabilities(x):
    values = x.astype(np.float64)
    exp = np.exp(values - np.max(values, axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


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
        design = {k: f[k] for k in f.files}
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
    for k, values in (
        ("sample_ids", ids),
        ("labels", labels),
        ("targets", targets),
        ("original_logits", original),
    ):
        exact(design[k], values)
    refs = np.stack((np.zeros((1, 8, 8), np.float32), np.median(train, axis=0)))
    exact(refs, design["references"])
    mapping = json.loads((training / "feature_map.json").read_text())
    coordinates = np.array(
        [r["row"] * 8 + r["column"] for r in sorted(mapping, key=lambda r: r["feature_index"])]
    )
    exact(coordinates, design["coordinates"])
    row_path = next(
        Path(p) for p in manifest["source_file_sha256"] if Path(p).name == "row_assignments.json"
    )
    rows = {
        r["row_id"]: r for r in json.loads(row_path.read_text()) if r["partition"] == "validation"
    }
    groups = np.array([rows[int(i)]["group_id"] for i in ids])
    exact(groups, design["group_ids"])
    review = yaml.safe_load(Path(config["grad_cam_review"]).read_text())
    with np.load(Path(review["run_dir"]) / "grad_cam_arrays.npz") as f:
        scores = f["feature_scores"]
    exact(scores, design["feature_scores"])
    unique, codes = np.unique(groups, return_inverse=True)
    random = np.argsort(
        np.random.Generator(np.random.PCG64(policy["random_seed"])).random(
            (policy["random_repeats"], len(unique), 57)
        ),
        axis=-1,
        kind="stable",
    )[:, codes]
    orders = np.concatenate(
        (
            np.argsort(-scores, axis=1, kind="stable")[None],
            np.argsort(scores, axis=1, kind="stable")[None],
            random,
        )
    ).astype(np.int16)
    exact(orders, design["rankings"])
    steps = np.array(policy["deletion_steps"], dtype=np.int64)
    exact(steps, design["steps"])
    stored = np.load(root / "perturbed_logits.npy", mmap_mode="r", allow_pickle=False)
    assert stored.shape == (2, len(orders), len(steps), len(images), 2)
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
        chunks = []
        for start in range(0, len(x), policy["batch_size"]):
            tensor = torch.from_numpy(x[start : start + policy["batch_size"]].copy())
            chunks.append(net.head(net.features[:3](tensor)).numpy())
            calls += 1
        return np.concatenate(chunks)

    padding = np.ones(64, bool)
    padding[coordinates] = False
    with torch.no_grad(), torch.backends.nnpack.flags(enabled=True):
        exact(forward(images), original)
        for r, ref in enumerate(refs):
            for i, order in enumerate(orders):
                exact(stored[r, i, 0], original)
                positions = np.argsort(order, axis=1)
                for j, k in enumerate(steps[1:], 1):
                    mask = np.zeros((len(images), 64), bool)
                    mask[:, coordinates] = positions < k
                    assert np.all(mask.sum(1) == k) and not mask[:, padding].any()
                    perturbed = np.where(mask.reshape(images.shape), ref, images)
                    exact(
                        perturbed.reshape(len(images), 64)[:, padding],
                        images.reshape(len(images), 64)[:, padding],
                    )
                    exact(forward(perturbed), stored[r, i, j])
    assert calls == summary["forward_calls"]
    base_prob = probabilities(original)[np.arange(len(targets)), targets]
    take = targets[None, None, None, :, None]
    drops = {
        "logit": original[np.arange(len(targets)), targets].astype(np.float64)
        - np.take_along_axis(stored.astype(np.float64), take, axis=-1)[..., 0],
        "probability": base_prob - np.take_along_axis(probabilities(stored), take, axis=-1)[..., 0],
    }
    aopc = {
        k: ((v[:, :, :-1] + v[:, :, 1:]) * np.diff(steps)[None, None, :, None] / 2).sum(2) / 57
        for k, v in drops.items()
    }
    with np.load(root / "per_row_aopc.npz") as f:
        for k in aopc:
            close(aopc[k], f[k])
    summaries = json.loads((root / "aopc_summaries.json").read_text())
    curves = json.loads((root / "curves.json").read_text())
    pred = stored.argmax(-1)
    subsets = {
        "all": np.ones(len(ids), bool),
        "true_0": labels == 0,
        "true_1": labels == 1,
        "predicted_0": targets == 0,
        "predicted_1": targets == 1,
        "correct": targets == labels,
        "incorrect": targets != labels,
    }
    rng = np.random.default_rng(policy["bootstrap_seed"])
    checked_summaries, checked_curves = 0, 0
    for name, subset in subsets.items():
        keys, subgroup_codes, sizes = np.unique(
            groups[subset], return_inverse=True, return_counts=True
        )
        if not len(keys):
            continue
        draws = rng.integers(len(keys), size=(policy["bootstrap_resamples"], len(keys)))
        for entry in (r for r in summaries if r["group"] == name):
            r = 0 if entry["replacement"] == "zero" else 1
            a = aopc[entry["metric"]][r][:, subset]
            vals = {"top": a[0], "bottom": a[1], "random": a[2:].mean(0)}
            vals.update(
                top_minus_random=vals["top"] - vals["random"],
                top_minus_bottom=vals["top"] - vals["bottom"],
            )
            vector = vals[entry["order_or_contrast"]]
            sums = np.array([vector[subgroup_codes == g].sum() for g in range(len(keys))])
            if entry["weighting"] == "row_primary":
                point, distribution = vector.mean(), sums[draws].sum(1) / sizes[draws].sum(1)
            else:
                point, distribution = (sums / sizes).mean(), (sums / sizes)[draws].mean(1)
            close(point, entry["mean"])
            assert entry["rows"] == int(subset.sum()) and entry["feature_groups"] == len(keys)
            if len(keys) >= policy["minimum_interval_groups"]:
                close(np.quantile(distribution, [0.025, 0.975]), entry["ci95"])
            else:
                assert entry["ci95"] is None
            checked_summaries += 1
        for entry in (r for r in curves if r["group"] == name):
            r = 0 if entry["replacement"] == "zero" else 1
            j = steps.tolist().index(entry["deleted_features"])
            ix = {"top": slice(0, 1), "bottom": slice(1, 2), "random": slice(2, None)}[
                entry["order"]
            ]
            weights = (
                np.ones(subset.sum())
                if entry["weighting"] == "row_primary"
                else 1 / sizes[subgroup_codes]
            )
            selected_pred = pred[r, ix, j][:, subset]
            for metric in ("logit", "probability"):
                value = (drops[metric][r, ix, j][:, subset] * weights).sum(
                    -1
                ).mean() / weights.sum()
                close(value, entry[f"mean_{metric}_drop"])
            close(
                ((selected_pred == labels[subset]) * weights).sum(-1).mean() / weights.sum(),
                entry["accuracy"],
            )
            close(
                ((selected_pred != targets[subset]) * weights).sum(-1).mean() / weights.sum(),
                entry["flip_rate"],
            )
            f1 = []
            for row in selected_pred:
                for c in (0, 1):
                    tp = weights[(row == c) & (labels[subset] == c)].sum()
                    fp = weights[(row == c) & (labels[subset] != c)].sum()
                    fn = weights[(row != c) & (labels[subset] == c)].sum()
                    f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
            close(np.mean(f1), entry["macro_f1"])
            checked_curves += 1
    assert checked_summaries == len(summaries) and checked_curves == len(curves)
    return {
        "status": "independent_full_replay_passed",
        "input_hashes_verified": len(manifest["source_file_sha256"]),
        "manifest_output_hashes_verified": len(manifest["output_file_sha256"]),
        "forward_calls_replayed": calls,
        "all_perturbation_logits_bitwise_exact": True,
        "aopc_summaries_and_intervals_verified": checked_summaries,
        "curves_verified": checked_curves,
        "execution_commit": manifest["commit_sha"],
        "file_sha256": {p.name: digest(p) for p in sorted(root.iterdir()) if p.is_file()},
    }


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1])), indent=2))
