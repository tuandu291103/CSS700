"""Read-only independent v2 audit; no imports from the css700 implementation.

Usage: .venv/bin/python scripts/verify_spambase_grad_cam_v2.py artifacts/<v2-run>
Prints evidence to stdout; never edits a run or changes its acceptance decision.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


class Replica(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.grad_cam_layer = self.features[2]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, 2)

    def head(self, activation):
        return self.classifier(self.pool(self.features[3](activation)).flatten(1))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    numerical = json.loads((root / "numerical_diagnostics.json").read_text())
    for path, expected in manifest["source_file_sha256"].items():
        assert digest(Path(path)) == expected, path
    for name, expected in manifest["output_file_sha256"].items():
        assert digest(root / name) == expected, name
    training = Path(summary["source_training_run"])
    with np.load(training / "inputs.npz", allow_pickle=False) as source:
        images = source["validation_images"]
    with np.load(training / "validation_predictions.npz", allow_pickle=False) as source:
        predictions = {k: source[k] for k in source.files}
    with np.load(root / "grad_cam_arrays.npz", allow_pickle=False) as source:
        a = {k: source[k] for k in source.files}
    with np.load(root / "numerical_logits.npz", allow_pickle=False) as source:
        modes = {k: source[k] for k in source.files}
    for k, expected in (
        ("sample_ids", predictions["sample_ids"]),
        ("labels", predictions["labels"]),
        ("predicted_classes", predictions["predictions"]),
        ("target_classes", predictions["predictions"]),
        ("group_equal_weights", predictions["group_equal_weights"]),
        ("inference_logits", predictions["logits"]),
    ):
        exact(a[k], expected)
    exact(modes["sample_ids"], a["sample_ids"])
    model = Replica().eval()
    checkpoint = torch.load(training / "best_checkpoint.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    references, candidates = [], []
    names = ("nnpack_on_no_grad", "nnpack_on_grad", "nnpack_off_no_grad", "nnpack_off_grad")
    replay = {name: [] for name in names}
    for start in range(0, len(images), 32):
        x = torch.from_numpy(images[start : start + 32].copy())
        targets = a["target_classes"][start : start + 32]
        for name, enabled, grad in zip(
            names, (True, True, False, False), (False, True, False, True), strict=True
        ):
            with torch.backends.nnpack.flags(enabled=enabled), torch.set_grad_enabled(grad):
                result = model.head(model.features[:3](x))
            replay[name].append(result.detach().numpy())
        with torch.backends.nnpack.flags(enabled=True), torch.no_grad():
            activation = model.features[:3](x)
        leaf = activation.detach().requires_grad_(True)
        logits = model.head(leaf)
        candidates.append(logits.detach().numpy())
        values = activation.numpy().astype(np.float64)
        weights = model.classifier.weight.detach().numpy().astype(np.float64)[targets]
        derivative = weights[:, :, None, None] * (values > 0) / 64
        references.append(
            np.maximum((derivative.mean((2, 3), keepdims=True) * values).sum(1, keepdims=True), 0)
        )
    for name in names:
        exact(np.concatenate(replay[name]), modes[name])
    exact(np.concatenate(candidates), a["logits"])
    exact(a["logits"], predictions["logits"])
    reference = np.concatenate(references)
    np.testing.assert_allclose(a["raw_maps"], reference, rtol=1e-5, atol=1e-7)
    normalized = a["raw_maps"] / np.maximum(a["raw_maps"].max((2, 3), keepdims=True), 1e-12)
    exact(normalized, a["normalized_maps"])
    mapping = json.loads((root / "feature_map.json").read_text())
    assert mapping == json.loads((training / "feature_map.json").read_text())
    mapping.sort(key=lambda r: r["feature_index"])
    assert [r["feature_index"] for r in mapping] == list(range(57))
    mask = np.zeros((1, 8, 8), dtype=bool)
    for record in mapping:
        k, row, column = record["feature_index"], record["row"], record["column"]
        mask[0, row, column] = True
        np.testing.assert_array_equal(a["feature_scores"][:, k], a["raw_maps"][:, 0, row, column])
        np.testing.assert_array_equal(
            a["normalized_feature_scores"][:, k], normalized[:, 0, row, column]
        )
    exact(mask, a["occupied_mask"])
    scores = a["feature_scores"]
    records = json.loads((root / "per_sample.json").read_text())
    assert len(records) == len(images)
    fractions = []
    for i, record in enumerate(records):
        positive = [int(k) for k in np.argsort(-scores[i], kind="stable") if scores[i, k] > 0]
        total = float(a["raw_maps"][i].sum(dtype=np.float64))
        padding = float(a["raw_maps"][i][~mask].sum(dtype=np.float64))
        fraction = padding / total if total else None
        expected = {
            "sample_id": int(a["sample_ids"][i]),
            "true_label": int(a["labels"][i]),
            "predicted_class": int(a["predicted_classes"][i]),
            "target_class": int(a["target_classes"][i]),
            "correct": bool(a["labels"][i] == a["predicted_classes"][i]),
            "raw_map_mass": total,
            "feature_mass": float(scores[i].sum()),
            "padding_mass": padding,
            "padding_fraction": fraction,
            "zero_raw_map": total == 0,
            "zero_feature_map": not positive,
            "positive_feature_count": len(positive),
        }
        for k in (5, 10):
            expected[f"positive_top_{k}"] = [
                {
                    "feature_index": j,
                    "feature_name": mapping[j]["feature_name"],
                    "raw_score": float(scores[i, j]),
                }
                for j in positive[:k]
            ]
            expected[f"positive_tie_at_{k}"] = bool(
                len(positive) > k and scores[i, positive[k - 1]] == scores[i, positive[k]]
            )
        assert record == expected, i
        if fraction is not None:
            fractions.append(fraction)
    aggregates = json.loads((root / "aggregate_features.json").read_text())
    groups = {"all_predicted_classes": np.ones(len(images), dtype=bool)}
    groups.update({f"predicted_class_{c}": a["predicted_classes"] == c for c in (0, 1)})
    for name, selected in groups.items():
        assert aggregates[name]["rows"] == int(selected.sum())
        for weighting in ("row_primary", "group_equal"):
            expected = []
            if selected.any():
                weights = (
                    np.ones(selected.sum())
                    if weighting == "row_primary"
                    else a["group_equal_weights"][selected]
                )
                means = np.average(scores[selected], axis=0, weights=weights)
                expected = [
                    {
                        "rank": rank,
                        "feature_index": int(k),
                        "feature_name": mapping[k]["feature_name"],
                        "mean_raw_score": float(means[k]),
                    }
                    for rank, k in enumerate(np.argsort(-means, kind="stable"), 1)
                ]
            assert aggregates[name][weighting] == expected
    examples = []
    for actual, predicted in ((0, 0), (1, 1), (0, 1), (1, 0)):
        selected = np.flatnonzero((a["labels"] == actual) & (a["predicted_classes"] == predicted))
        if len(selected):
            examples.append(int(a["sample_ids"][selected[0]]))
    assert summary["example_sample_ids"] == examples
    return {
        "status": "independent_checks_passed_not_automatic_acceptance",
        "input_hashes_verified": len(manifest["source_file_sha256"]),
        "manifest_output_hashes_verified": len(manifest["output_file_sha256"]),
        "all_four_modes_and_candidate_replayed_exactly": True,
        "candidate_logits_bitwise_equal_frozen_inference": True,
        "analytic_reference_max_abs_error": float(np.abs(reference - a["raw_maps"]).max()),
        "analytic_reference_failed_cells": int(
            (~np.isclose(a["raw_maps"], reference, rtol=1e-5, atol=1e-7)).sum()
        ),
        "all_rows_features_records_aggregates_examples_verified": True,
        "rows": len(images),
        "feature_scores": scores.size,
        "padding_fraction_mean": float(np.mean(fractions)),
        "execution_commit": manifest["commit_sha"],
        "numerical": numerical,
        "file_sha256": {p.name: digest(p) for p in sorted(root.iterdir()) if p.is_file()},
    }


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1])), indent=2))
