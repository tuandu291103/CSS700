"""Named-feature exports and separate, fit-aware comparisons for frozen LIME v1."""

from itertools import combinations

import numpy as np

from css700.evaluation.local_agreement import paired_rank_rows
from css700.experiments.tree_shap_sources import named_feature_matrix

FAMILIES = ("random_forest", "xgboost")
SEEDS = (11, 29, 47)


def feature_rows(record: dict, coefficients: np.ndarray, metadata: dict) -> list[dict]:
    names = metadata["feature_names"]
    if (
        coefficients.shape != (len(names),)
        or not np.isfinite(coefficients).all()
        or len(set(names)) != len(names)
        or metadata["sample_id"] != record["sample_id"]
        or metadata["explainer_seed"] != record["explainer_seed"]
    ):
        raise ValueError("LIME feature export identity mismatch")
    ranks = np.argsort(np.argsort(-abs(coefficients), kind="stable"), kind="stable") + 1
    return [
        {
            "sample_id": record["sample_id"],
            "family": record["family"],
            "explainer_seed": record["explainer_seed"],
            "true_label": record["true_label"],
            "predicted_label": record["target_class"],
            "target_class": record["target_class"],
            "feature_index": j,
            "feature_name": name,
            "signed_coefficient": float(coefficients[j]),
            "absolute_rank": int(ranks[j]),
            "anchor_bin_id": metadata["anchor_bin_ids"][j],
            "intercept": record["intercept"],
            "anchor_model_probability": record["anchor_model_probability"],
            "anchor_surrogate_prediction": record["anchor_surrogate_prediction"],
            "anchor_absolute_error": record["anchor_absolute_error"],
            "evaluation_weighted_r2": record["evaluation"]["weighted_r2"],
            "evaluation_weighted_rmse": record["evaluation"]["weighted_rmse"],
        }
        for j, name in enumerate(names)
    ]


def index_diagnostics(records: list[dict], sample_ids, labels, predictions) -> dict:
    ids = np.asarray(sample_ids)
    if ids.ndim != 1 or len(set(ids.tolist())) != len(ids) or len(labels) != len(ids):
        raise ValueError("invalid sample/label identities")
    expected = {(f, s, int(i)) for f in FAMILIES for s in SEEDS for i in ids}
    indexed = {(r["family"], r["explainer_seed"], r["sample_id"]): r for r in records}
    if len(indexed) != len(records) or set(indexed) != expected:
        raise ValueError("missing, duplicate or unexpected LIME fits")
    for family in FAMILIES:
        targets = predictions[family]
        if np.asarray(targets).shape != ids.shape or not np.isin(targets, [0, 1]).all():
            raise ValueError("invalid target vector")
        for seed in SEEDS:
            for i, sample_id in enumerate(ids):
                r = indexed[(family, seed, int(sample_id))]
                if (
                    r["partition"] != "test"
                    or r["true_label"] != int(labels[i])
                    or r["target_class"] != int(targets[i])
                ):
                    raise ValueError("LIME sample/label/target identity mismatch")
    return indexed


def comparison_tables(features: list[dict], records: list[dict], shap: dict) -> list[dict]:
    ids, labels, names = shap["sample_ids"], shap["labels"], shap["feature_names"]
    predictions = {f: shap["conditions"][f"{f}__train_all"]["predictions"] for f in FAMILIES}
    indexed = index_diagnostics(records, ids, labels, predictions)
    if len(features) != len(records) * len(names) or any(
        r["family"] not in FAMILIES or r["explainer_seed"] not in SEEDS for r in features
    ):
        raise ValueError("missing or unexpected named-feature exports")
    matrices = {
        (f, s): named_feature_matrix(
            [r for r in features if r["family"] == f and r["explainer_seed"] == s],
            ids,
            names,
            predictions[f],
            labels,
            value_field="signed_coefficient",
        )
        for f in FAMILIES
        for s in SEEDS
    }
    rows = []

    def compare(kind, left, right):
        family, seed = left
        right_family, right_seed = right
        right_scores = (
            shap["conditions"][f"{right_family}__train_all"]["scores"]
            if right_seed is None
            else matrices[right]
        )
        right_name = "shap-train-all" if right_seed is None else f"lime-{right_seed}"
        paired = paired_rank_rows(
            matrices[left],
            right_scores,
            predictions[family],
            predictions[right_family],
            labels,
            ids,
            comparison=f"{kind}__{family}-lime-{seed}__{right_family}-{right_name}",
        )
        for row in paired:
            row.update(
                analysis=kind,
                left_family=family,
                right_family=right_family,
                left_explainer_seed=seed,
                right_explainer_seed=right_seed,
                primary_lime_shap=(kind == "same_model_shap" and seed == 11),
            )
            for side, key in (("left", left), ("right", right)):
                diag = indexed[(*key, row["sample_id"])] if key[1] is not None else None
                row[f"{side}_anchor_absolute_error"] = (
                    diag["anchor_absolute_error"] if diag else None
                )
                for metric in ("weighted_rmse", "weighted_r2"):
                    row[f"{side}_evaluation_{metric}"] = (
                        diag["evaluation"][metric] if diag else None
                    )
            rows.append(row)

    for family in FAMILIES:
        for seed in SEEDS:
            compare("same_model_shap", (family, seed), (family, None))
        for a, b in combinations(SEEDS, 2):
            compare("sampling_stability", (family, a), (family, b))
    for seed in SEEDS:
        compare("cross_model", (FAMILIES[0], seed), (FAMILIES[1], seed))
    return rows


def fit_quality_summary(records: list[dict]) -> list[dict]:
    """Descriptive distributions, never pooled over sampling seeds or fit-filtered."""
    summaries = []
    for family in FAMILIES:
        for seed in SEEDS:
            selected = [r for r in records if r["family"] == family and r["explainer_seed"] == seed]
            groups = {"all": selected}
            for target in (0, 1):
                groups[f"predicted_class_{target}"] = [
                    r for r in selected if r["target_class"] == target
                ]
            for correct in (True, False):
                groups["correct" if correct else "incorrect"] = [
                    r for r in selected if (r["target_class"] == r["true_label"]) == correct
                ]
            for group, members in groups.items():
                result = {
                    "family": family,
                    "explainer_seed": seed,
                    "group": group,
                    "samples": len(members),
                    "anchor_out_of_bounds_count": sum(
                        not 0 <= r["anchor_surrogate_prediction"] <= 1 for r in members
                    ),
                }
                fields = {"anchor_absolute_error": [r["anchor_absolute_error"] for r in members]}
                for role in ("fit", "evaluation"):
                    result[f"{role}_undefined_r2_count"] = sum(
                        r[role]["weighted_r2"] is None for r in members
                    )
                    for metric in (
                        "weighted_r2",
                        "weighted_rmse",
                        "weighted_mae",
                        "effective_sample_size",
                        "surrogate_out_of_probability_bounds_fraction",
                    ):
                        fields[f"{role}_{metric}"] = [r[role][metric] for r in members]
                for metric, raw in fields.items():
                    values = np.asarray([v for v in raw if v is not None], dtype=float)
                    if not np.isfinite(values).all():
                        raise ValueError("nonfinite fit quality summary input")
                    result[metric] = {
                        "defined_samples": len(values),
                        "mean": float(values.mean()) if len(values) else None,
                        "median": float(np.median(values)) if len(values) else None,
                        "quartiles": np.quantile(values, [0.25, 0.75]).tolist()
                        if len(values)
                        else None,
                        "min": float(values.min()) if len(values) else None,
                        "max": float(values.max()) if len(values) else None,
                    }
                summaries.append(result)
    return summaries
