"""Deterministic whole-feature-group allocation, without model or preprocessing fits."""

import hashlib
import math

import numpy as np

from css700.data.splits import split_fingerprint

PARTITIONS = ("train", "validation", "test")
SPLIT_POLICY = {
    "algorithm": "largest_first_balanced_greedy_v1",
    "seed": 42,
    "fractions": [0.7, 0.15, 0.15],
    "max_row_fraction_error": 0.01,
    "max_class_proportion_error": 0.02,
    "row_order": "ascending_source_row_id",
    "conflicting_labels": "retain_original_labels_and_keep_group_together",
}


def allocate_groups(memberships: list[dict], seed: int = 42) -> tuple[dict, list[dict]]:
    """Largest groups first; hash-seeded ties; no repeated seeds or post-allocation swaps."""
    if not memberships or type(seed) is not int or seed < 0:
        raise ValueError("nonempty memberships and a nonnegative integer seed required")
    groups = {}
    seen = set()
    for member in memberships:
        row_id, label = member["row_id"], member["target_class"]
        if type(row_id) is not int or row_id < 0 or row_id in seen:
            raise ValueError("row IDs must be unique nonnegative integers")
        if type(label) is not int or label not in (0, 1):
            raise ValueError("binary integer labels required")
        seen.add(row_id)
        group = groups.setdefault(
            member["feature_vector_sha256"], {"row_ids": [], "class_counts": np.zeros(2, int)}
        )
        group["row_ids"].append(row_id)
        group["class_counts"][label] += 1
    if seen != set(range(len(memberships))):
        raise ValueError("memberships must cover every zero-based source row exactly once")
    totals = sum((g["class_counts"] for g in groups.values()), start=np.zeros(2, int))
    if not np.all(totals > 0):
        raise ValueError("both classes required")
    # Exact targets q/20, with a common denominator for the squared objective.
    # Integer comparison avoids platform-dependent floating-point near-tie decisions.
    numerators = (14, 3, 3)
    scaled_targets = [q * int(total) for q in numerators for total in totals]
    scaled_targets += [q * len(memberships) for q in numerators]
    common_denominator = math.lcm(*(target**2 for target in scaled_targets))
    weights = [common_denominator // target**2 for target in scaled_targets]

    def order(item):
        digest, group = item
        key = hashlib.sha256(
            b"css700-group-order-v1\0" + str(seed).encode("ascii") + b"\0" + digest.encode("ascii")
        ).hexdigest()
        return -len(group["row_ids"]), key, digest

    def objective(counts):
        actual = counts.flatten().tolist() + counts.sum(axis=1).tolist()
        return sum(
            weight * (20 * count - target) ** 2
            for weight, count, target in zip(weights, actual, scaled_targets, strict=True)
        )

    counts = np.zeros((3, 2), dtype=np.int64)
    row_ids = {name: [] for name in PARTITIONS}
    allocations = []
    for digest, group in sorted(groups.items(), key=order):
        scores = []
        for partition in range(3):
            candidate = counts.copy()
            candidate[partition] += group["class_counts"]
            scores.append(objective(candidate))
        chosen = min(range(3), key=scores.__getitem__)  # Ties use partition order.
        counts[chosen] += group["class_counts"]
        ids = sorted(group["row_ids"])
        row_ids[PARTITIONS[chosen]].extend(ids)
        allocations.append(
            {
                "allocation_order": len(allocations),
                "feature_vector_sha256": digest,
                "partition": PARTITIONS[chosen],
                "row_ids": ids,
                "class_counts": group["class_counts"].tolist(),
                "conflicting_labels": bool(np.all(group["class_counts"] > 0)),
            }
        )
    return {name: sorted(ids) for name, ids in row_ids.items()}, allocations


def verify_partitions(memberships: list[dict], row_ids: dict) -> dict:
    """Recheck row coverage and group disjointness separately from the allocator."""
    if set(row_ids) != set(PARTITIONS):
        raise ValueError("exactly train, validation and test partitions required")
    all_ids = [i for name in PARTITIONS for i in row_ids[name]]
    n = len(memberships)
    if any(type(i) is not int for i in all_ids):
        raise ValueError("partition row IDs must be integers")
    if sorted(all_ids) != list(range(n)):
        raise ValueError("partition coverage has omissions, duplicates or out-of-range IDs")
    by_id = {m["row_id"]: m for m in memberships}
    if set(by_id) != set(range(n)) or len(by_id) != len(memberships):
        raise ValueError("invalid source row memberships")
    global_counts = np.bincount([by_id[i]["target_class"] for i in range(n)], minlength=2)
    global_proportions = global_counts / n
    assigned, details = {}, {}
    for j, name in enumerate(PARTITIONS):
        ids = row_ids[name]
        if ids != sorted(ids):
            raise ValueError("partition rows must retain ascending source order")
        for i in ids:
            digest = by_id[i]["feature_vector_sha256"]
            if digest in assigned and assigned[digest] != name:
                raise ValueError("a feature group crosses partitions")
            assigned[digest] = name
        counts = np.bincount([by_id[i]["target_class"] for i in ids], minlength=2)
        details[name] = {
            "rows": len(ids),
            "groups": len({by_id[i]["feature_vector_sha256"] for i in ids}),
            "class_counts": {str(c): int(counts[c]) for c in (0, 1)},
            "fraction": len(ids) / n,
            "spam_proportion": float(counts[1] / len(ids)) if ids else None,
            "row_fraction_error": abs(len(ids) / n - SPLIT_POLICY["fractions"][j]),
            "max_class_proportion_error": (
                float(np.max(np.abs(counts / len(ids) - global_proportions))) if ids else None
            ),
        }
    checks = {
        "complete_disjoint_row_coverage": True,
        "zero_cross_partition_feature_group_overlap": True,
        "both_classes_in_every_partition": all(
            min(d["class_counts"].values()) > 0 for d in details.values()
        ),
        "row_fraction_tolerance": all(
            d["row_fraction_error"] <= SPLIT_POLICY["max_row_fraction_error"]
            for d in details.values()
        ),
        "class_proportion_tolerance": all(
            d["max_class_proportion_error"] is not None
            and d["max_class_proportion_error"] <= SPLIT_POLICY["max_class_proportion_error"]
            for d in details.values()
        ),
    }
    return {
        "checks": checks,
        "partitions": details,
        "split_fingerprint": split_fingerprint(
            *(np.asarray(row_ids[name], dtype=np.int64) for name in PARTITIONS)
        ),
    }
