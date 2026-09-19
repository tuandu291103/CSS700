"""Pre-split Spambase integrity checks; no preprocessing, splits or model fitting."""

import csv
import hashlib
from collections import Counter

import numpy as np

WORDS = (
    "make address all 3d our over remove internet order mail receive will people report "
    "addresses free business email you credit your font 000 money hp hpl george 650 lab "
    "labs telnet 857 data 415 85 technology 1999 parts pm direct cs meeting original "
    "project re edu table conference"
).split()
FEATURE_NAMES = (
    tuple(f"word_freq_{word}" for word in WORDS)
    + tuple(f"char_freq_{char}" for char in ";([!$#")
    + ("capital_run_length_average", "capital_run_length_longest", "capital_run_length_total")
)
MISSING_TOKENS = {"", "?", "na", "n/a", "nan", "null", "none"}


def parse_names(payload: bytes) -> tuple[str, ...]:
    names, classes = [], []
    for raw in payload.decode("ascii").splitlines():
        line = raw.split("|", 1)[0].strip()
        if not line:
            continue
        if ":" in line:
            name, kind = line.rsplit(":", 1)
            if kind.strip() != "continuous.":
                raise ValueError("unexpected source feature declaration")
            names.append(name.strip())
        else:
            classes.append(line.replace(" ", ""))
    if tuple(names) != FEATURE_NAMES or classes != ["1,0."]:
        raise ValueError("source feature order/names or class declaration mismatch")
    return tuple(names)


def parse_data(payload: bytes) -> tuple[np.ndarray, list[dict]]:
    """Keep every physical line as a zero-based row ID; malformed rows are findings."""
    lines = payload.decode("ascii").splitlines()
    values = np.full((len(lines), 58), np.nan)
    issues = []
    for row_id, line in enumerate(lines):
        fields = next(csv.reader([line], strict=True))
        if len(fields) != 58:
            issues.append(
                {
                    "row_id": row_id,
                    "line_number": row_id + 1,
                    "column_index": None,
                    "issue": "wrong_field_count",
                    "observed_fields": len(fields),
                }
            )
            continue
        for j, field in enumerate(fields):
            token = field.strip()
            issue = None
            if token.lower() in MISSING_TOKENS:
                issue = "missing_token"
            else:
                try:
                    value = float(token)
                    if np.isfinite(value):
                        values[row_id, j] = value
                    else:
                        issue = "nonfinite_token"
                except ValueError:
                    issue = "nonnumeric_token"
            if issue:
                issues.append(
                    {
                        "row_id": row_id,
                        "line_number": row_id + 1,
                        "column_index": j,
                        "issue": issue,
                        "token": token,
                    }
                )
    return values, issues


def duplicate_groups(values: np.ndarray) -> tuple[dict, list[dict], list[dict]]:
    """Exact float64 feature equality (including +0 == -0), ignoring the target."""
    if values.ndim != 2 or values.shape[1] != 58 or not np.isfinite(values).all():
        raise ValueError("duplicate grouping requires complete finite 58-column rows")
    if not np.isin(values[:, -1], [0, 1]).all():
        raise ValueError("duplicate grouping requires binary target codes")
    groups = {}
    for row_id, row in enumerate(values):
        groups.setdefault(tuple(row[:-1]), []).append(row_id)
    duplicates, memberships = [], []
    conflicting = []
    for vector, ids in groups.items():
        canonical = np.asarray(vector, dtype="<f8").copy()
        canonical[canonical == 0] = 0  # Normalize signed zero for the stable hash.
        digest = hashlib.sha256(
            b"css700-spambase-float64-vector-v1\0" + canonical.tobytes()
        ).hexdigest()
        counts = Counter(str(int(values[i, -1])) for i in ids)
        record = {
            "group_id": f"feature-vector-{ids[0]:06d}",
            "feature_vector_sha256": digest,
            "first_row_id": ids[0],
            "row_ids": ids,
            "line_numbers": [i + 1 for i in ids],
            "size": len(ids),
            "class_counts": {str(c): counts.get(str(c), 0) for c in (0, 1)},
            "conflicting_labels": len(counts) > 1,
        }
        if len(ids) > 1:
            duplicates.append(record)
        if len(counts) > 1:
            conflicting.append(record)
        memberships.extend(
            {
                "row_id": i,
                "line_number": i + 1,
                "group_id": record["group_id"],
                "feature_vector_sha256": digest,
                "target_class": int(values[i, -1]),
            }
            for i in ids
        )
    memberships.sort(key=lambda row: row["row_id"])
    summary = {
        "definition": (
            "exact parsed float64 feature equality; target excluded; signed zero normalized"
        ),
        "unique_feature_vectors": len(groups),
        "duplicate_feature_groups": len(duplicates),
        "rows_in_duplicate_feature_groups": sum(r["size"] for r in duplicates),
        "redundant_feature_rows_beyond_first": len(values) - len(groups),
        "unique_labeled_rows": len(np.unique(values, axis=0)),
        "redundant_labeled_rows_beyond_first": len(values) - len(np.unique(values, axis=0)),
        "conflicting_feature_groups": len(conflicting),
        "rows_in_conflicting_feature_groups": sum(r["size"] for r in conflicting),
        "largest_feature_group": max((len(ids) for ids in groups.values()), default=0),
    }
    return summary, duplicates, memberships


def inspect_data(payload: bytes, names_payload: bytes) -> dict:
    names = parse_names(names_payload)
    values, issues = parse_data(payload)
    profiles = []
    for j, name in enumerate(names):
        column = values[:, j]
        valid = np.isfinite(column)
        finite = column[valid]
        lower, upper = (0, 100) if j < 54 else (1, None)
        bad_range = valid & ((column < lower) | ((column > upper) if upper is not None else False))
        bad_integer = (
            valid & (column != np.floor(column))
            if j in (55, 56)
            else np.zeros(len(column), dtype=bool)
        )
        profiles.append(
            {
                "feature_index": j,
                "feature_name": name,
                "finite_count": int(valid.sum()),
                "nonfinite_or_unparsed_count": int((~valid).sum()),
                "min": float(finite.min()) if len(finite) else None,
                "max": float(finite.max()) if len(finite) else None,
                "zero_count": int(np.sum(finite == 0)),
                "unique_finite_values": len(np.unique(finite)),
                "constant_in_audited_release": bool(len(finite) and np.ptp(finite) == 0),
                "expected_lower_bound": lower,
                "expected_upper_bound": upper,
                "requires_integer": j in (55, 56),
                "out_of_range_row_ids": np.flatnonzero(bad_range).tolist(),
                "noninteger_row_ids": np.flatnonzero(bad_integer).tolist(),
            }
        )
    labels = values[:, -1]
    valid_labels = np.isin(labels, [0, 1])
    class_counts = {str(c): int(np.sum(labels == c)) for c in (0, 1)}
    counts = Counter(i["issue"] for i in issues)
    checks = {
        "expected_4601_rows": len(values) == 4601,
        "all_rows_have_58_fields": counts["wrong_field_count"] == 0,
        "all_values_finite_numeric_without_missing_tokens": not issues
        and bool(np.isfinite(values).all()),
        "binary_labels_only": bool(valid_labels.all()),
        "published_class_counts": class_counts == {"0": 2788, "1": 1813},
        "documented_feature_ranges": not any(r["out_of_range_row_ids"] for r in profiles),
        "integer_count_features": not any(r["noninteger_row_ids"] for r in profiles),
    }
    grouping = {
        "performed": False,
        "reason": "invalid values or labels; no rows silently discarded",
    }
    duplicates, memberships = [], []
    if not issues and np.isfinite(values).all() and valid_labels.all():
        stats, duplicates, memberships = duplicate_groups(values)
        grouping = {"performed": True, **stats}
    return {
        "schema_checks": checks,
        "rows": len(values),
        "features": len(names),
        "class_counts": class_counts,
        "invalid_label_row_ids": np.flatnonzero(~valid_labels).tolist(),
        "parse_issue_counts": {
            k: counts[k]
            for k in ("wrong_field_count", "missing_token", "nonfinite_token", "nonnumeric_token")
        },
        "parse_issues": issues,
        "feature_profiles": profiles,
        "duplicates": grouping,
        "duplicate_groups": duplicates,
        "row_memberships": memberships,
    }
