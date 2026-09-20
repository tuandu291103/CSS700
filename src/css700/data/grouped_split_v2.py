"""Prospective v2: target-weighted whole-group allocation and multiplicity gates."""

import hashlib
import math
from collections import Counter

import numpy as np

from css700.data.grouped_split import PARTITIONS, SPLIT_POLICY, verify_partitions

SPLIT_POLICY_V2 = {
    **SPLIT_POLICY,
    "algorithm": "largest_first_target_weighted_group_balance_v2",
    "objective_components": ["class_0_rows", "class_1_rows", "rows", "groups", "duplicate_rows"],
    "max_group_fraction_error": 0.02,
    "max_duplicate_row_proportion_error": 0.03,
    "required_group_types": "singleton_and_duplicate_in_every_partition",
}
SIZE_BINS = {"1": (1, 1), "2": (2, 2), "3-5": (3, 5), "6-10": (6, 10), "11+": (11, None)}


def allocate_groups_v2(memberships: list[dict], seed: int = 42) -> tuple[dict, list[dict]]:
    """One largest-first pass; exact rational objective; the v1 ordering is unchanged."""
    if not memberships or type(seed) is not int or seed < 0:
        raise ValueError("nonempty memberships and a nonnegative integer seed required")
    groups, seen = {}, set()
    for member in memberships:
        i, label, digest = member["row_id"], member["target_class"], member["feature_vector_sha256"]
        if type(i) is not int or i < 0 or i in seen:
            raise ValueError("row IDs must be unique nonnegative integers")
        if type(label) is not int or label not in (0, 1):
            raise ValueError("binary integer labels required")
        seen.add(i)
        group = groups.setdefault(digest, {"row_ids": [], "class_counts": [0, 0]})
        group["row_ids"].append(i)
        group["class_counts"][label] += 1
    if seen != set(range(len(memberships))):
        raise ValueError("memberships must cover every zero-based source row exactly once")
    for group in groups.values():
        size = len(group["row_ids"])
        group["components"] = [*group["class_counts"], size, 1, size if size > 1 else 0]
    totals = [sum(g["components"][j] for g in groups.values()) for j in range(5)]
    if min(totals[:2]) <= 0:
        raise ValueError("both classes required")
    # Objective = sum((count - target)^2 / target). Remove the common positive
    # factor 1/20, then multiply by an integer common denominator. Zero-total
    # components contribute zero, not a division by zero or a fitted epsilon.
    targets = [q * total for q in (14, 3, 3) for total in totals]
    denominator = math.lcm(*(target for target in targets if target))
    weights = [denominator // target if target else 0 for target in targets]

    def order(item):
        digest, group = item
        key = hashlib.sha256(
            b"css700-group-order-v1\0" + str(seed).encode("ascii") + b"\0" + digest.encode("ascii")
        ).hexdigest()
        return -len(group["row_ids"]), key, digest

    counts = np.zeros((3, 5), dtype=np.int64)
    row_ids, allocations = {p: [] for p in PARTITIONS}, []
    for digest, group in sorted(groups.items(), key=order):
        scores = []
        for partition in range(3):
            candidate = counts.copy()
            candidate[partition] += group["components"]
            scores.append(
                sum(
                    weight * (20 * actual - target) ** 2
                    for weight, actual, target in zip(
                        weights, candidate.flatten().tolist(), targets, strict=True
                    )
                )
            )
        chosen = min(range(3), key=scores.__getitem__)
        counts[chosen] += group["components"]
        ids = sorted(group["row_ids"])
        row_ids[PARTITIONS[chosen]].extend(ids)
        allocations.append(
            {
                "allocation_order": len(allocations),
                "feature_vector_sha256": digest,
                "partition": PARTITIONS[chosen],
                "row_ids": ids,
                "class_counts": list(group["class_counts"]),
                "conflicting_labels": min(group["class_counts"]) > 0,
            }
        )
    return {p: sorted(ids) for p, ids in row_ids.items()}, allocations


def verify_partitions_v2(memberships: list[dict], row_ids: dict) -> dict:
    """Retain v1 row/class/disjointness gates; add prospectively fixed group-size gates."""
    result = verify_partitions(memberships, row_ids)
    by_id = {m["row_id"]: m for m in memberships}
    sizes = Counter(m["feature_vector_sha256"] for m in memberships)
    duplicate_rows = sum(n for n in sizes.values() if n > 1)
    total_rows = len(memberships)
    overall_duplicate_proportion = duplicate_rows / total_rows
    groups = {}
    for member in memberships:
        groups.setdefault(member["feature_vector_sha256"], set()).add(member["target_class"])
    for j, partition in enumerate(PARTITIONS):
        ids = row_ids[partition]
        digests = {by_id[i]["feature_vector_sha256"] for i in ids}
        local_sizes = [sizes[d] for d in digests]
        local_duplicate_rows = sum(n for n in local_sizes if n > 1)
        bins = {
            name: {
                "groups": sum(lower <= n and (upper is None or n <= upper) for n in local_sizes),
                "rows": sum(n for n in local_sizes if lower <= n and (upper is None or n <= upper)),
            }
            for name, (lower, upper) in SIZE_BINS.items()
        }
        details = result["partitions"][partition]
        details.update(
            singleton_groups=sum(n == 1 for n in local_sizes),
            duplicate_groups=sum(n > 1 for n in local_sizes),
            rows_in_duplicate_groups=local_duplicate_rows,
            duplicate_row_proportion=local_duplicate_rows / len(ids) if ids else None,
            duplicate_row_proportion_error=(
                abs(local_duplicate_rows / len(ids) - overall_duplicate_proportion) if ids else None
            ),
            group_fraction=len(digests) / len(sizes),
            group_fraction_error=abs(len(digests) / len(sizes) - SPLIT_POLICY_V2["fractions"][j]),
            largest_group=max(local_sizes, default=0),
            group_size_bins=bins,
            conflicting_groups=sum(len(groups[d]) == 2 for d in digests),
        )
    result["checks"].update(
        group_fraction_tolerance=all(
            d["group_fraction_error"] <= SPLIT_POLICY_V2["max_group_fraction_error"]
            for d in result["partitions"].values()
        ),
        duplicate_row_proportion_tolerance=all(
            d["duplicate_row_proportion_error"] is not None
            and d["duplicate_row_proportion_error"]
            <= SPLIT_POLICY_V2["max_duplicate_row_proportion_error"]
            for d in result["partitions"].values()
        ),
        both_group_types_in_every_partition=all(
            d["singleton_groups"] > 0 and d["duplicate_groups"] > 0
            for d in result["partitions"].values()
        ),
    )
    result["source_group_structure"] = {
        "rows": total_rows,
        "groups": len(sizes),
        "singleton_groups": sum(n == 1 for n in sizes.values()),
        "duplicate_groups": sum(n > 1 for n in sizes.values()),
        "rows_in_duplicate_groups": duplicate_rows,
        "duplicate_row_proportion": overall_duplicate_proportion,
    }
    return result
