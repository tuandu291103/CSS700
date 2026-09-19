# Spambase grouped split v1: prospective partition-only contract

**Post-execution review:** v1 is diagnostic-only and **not accepted for preprocessing or
modeling**. Although technical gates passed, all repeated-feature groups went to held-out
partitions. See `configs/spambase_split_v1_review.yaml` and
[`the session report`](logs/2026-09-19-spambase-grouped-split.md).
The frozen pre-execution protocol below is retained unchanged in substance; its original
version is at commit `d9b17a82cbe30d02f05e01dc4a08dc62f89f12f0`.

Date: **2026-09-19**, following the completed source-integrity audit. Yu's continuation
authorizes proceeding with Spambase as the working second dataset and implementing the
recommended grouped split with original labels retained. This is not committee approval
or permission to reinterpret the original labels. Commit this protocol, configuration,
code and synthetic tests **before the first real-data allocation**.

## Inclusion and scope

Retain all 4,601 source rows, all 57 input features and every original target label. The
three conflicting-label pairs stay unchanged and together. Do not deduplicate, majority-vote,
exclude conflicting rows, balance by resampling, or inspect model outcomes. The accepted
audit has already observed the full release for integrity; the split construction now also
uses group sizes and target counts for balance. Disclose this, not a never-seen-test claim.

Group identity is exact parsed numerical feature equality, excluding the label, as defined
in `SPAMBASE_AUDIT_PROTOCOL.md`. Verify raw bytes, the committed source lock and all seven
accepted audit files against `configs/spambase_split_v1.yaml`. Recompute the integrity
report/memberships from the source and require agreement before using them for allocation.
Keep the original Breast Cancer split, configurations and artifacts unchanged.

## One deterministic allocation

Target **row** fractions in train / validation / test order: **0.70 / 0.15 / 0.15**.
The fixed split seed is **42**, separate from future model-training seeds 7, 42 and 2026.
All future encoders/models on this source version must share the same accepted row IDs;
never create a separate split for each model seed.

The project-defined `largest_first_balanced_greedy_v1` heuristic is:

1. Assemble whole feature groups and their two original-label counts.
2. Sort by descending group size; break size ties by ascending SHA-256 of the exact bytes
   `b"css700-group-order-v1\0" + b"42" + b"\0" + feature_vector_sha256.encode("ascii")`.
   Use the feature-vector digest itself as the final tie key. This makes ordering seeded
   and independent of the NumPy random-number generator or dictionary iteration order.
3. Start all allocated row/class counts at zero. For each group, tentatively add it to
   each of the three partitions and choose the lowest global objective:

   `sum_pc ((allocated_pc - fraction_p * total_c) / (fraction_p * total_c))**2`
   `+ sum_p ((allocated_rows_p - fraction_p * total_rows) / (fraction_p * total_rows))**2`.

   Every normalized cell has coefficient one. Candidate counts include this group and
   prior allocations, not later groups. Compare this objective exactly: use fractions
   `(14, 3, 3) / 20`, multiply squared rational terms by a common positive denominator,
   and compare Python integers. Exact ties use train, validation, test order. No
   floating-point near-tie decision is involved. Record runtime versions and preserve IDs.
4. Do not swap groups, retry seeds or optimize scores after this one pass. Export group
   allocation order and ascending original source row IDs within each partition.

This is a transparent balance heuristic, **not** a globally optimal allocation, IID random
sample or standard stratified-row split. It balances row totals and target counts; it does
not fit feature distributions or use prediction/explanation quality. Large groups can
make nominal fractions unattainable, which must cause a review rather than silent relaxation.

## Predeclared acceptance gates

- Each source row appears exactly once across three nonempty partitions.
- Each exact feature group appears in exactly one partition, including mixed-label groups.
- Both classes occur in each partition; every original label is preserved.
- For each partition, absolute observed row fraction minus target fraction is at most
  **0.01** (one percentage point of the full source row count).
- For each partition and class, its within-partition class proportion differs from the
  full-release class proportion by at most **0.02** (two percentage points).
- Source/receipt/config/audit bytes remain unchanged throughout execution.

The tolerances are prospective engineering gates, not confidence intervals or guarantees
of population representativeness. Failure retains the attempted allocation and failed
summary; stop for a documented, versioned policy amendment. Do not search candidates and
report only the most favorable split. A replay for verification is not a new seed search.

## Frozen preprocessing and evaluation principles

No preprocessing or models are fitted in this milestone. In the next training-only
preflight, use per-feature StandardScaler-style training mean and population standard
deviation; retain constant training columns with scale one and flag them. Keep all 57
features in source order, with no imputation or feature selection. Fail on invalid values;
never estimate anything from validation/test data to repair training preprocessing.
Training gives each original row weight one; retaining duplicates changes their influence
relative to deduplication and must be disclosed.

Primary evaluation retains one weight per original row for continuity with the pilot.
Also require a **group-equal sensitivity** using weight `1 / group_size` for each member
row: each feature group has total weight one, including conflicting pairs. Keep original
labels; do not reduce conflicting groups to a single invented target. Report row and group
counts together; repeated rows must not be described as independent observations.

When uncertainty estimates are introduced, resample whole feature groups and preserve
paired group draws across methods/encoders. Freeze the detailed interval procedure, budget,
seeds and undefined-metric handling in its own protocol before test evaluation; no intervals
or tests are computed here. Exact feature groups are an operational dependence unit, not
proof that different groups are independent emails or sender/thread groups.

## Execution, evidence and next gate

Run `make split-spambase PYTHON=.venv/bin/python` from a clean source/config commit.
Create a new ignored directory with exact config, manifest, partition row IDs, all-row
assignments, group allocation order and summary. Include source/output hashes, per-partition
row/group/class counts, tolerances, mixed-label allocation and the existing versioned
split fingerprint. Do not overwrite attempts. Independently verify numerical group
disjointness against raw values, all memberships/labels/weights and the fingerprint.

An accepted split permits a **training-only preprocessing/encoding/compute preflight**,
not an automatic predictive or explanation experiment. Freeze dataset-aware loader and
model/explainer contracts without weakening the Breast Cancer guards. SOM2IMG still requires
authoritative source/collision guidance. Committee scope review and artifact backup remain
separate; neither a local commit nor a push backs up ignored run bundles.
