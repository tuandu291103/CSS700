# Spambase grouped split v2: correction verified and accepted for preflight

Session date: **2026-09-19, America/Los_Angeles**. Run timestamp: September 20 at
06:27:33 UTC (September 19 at 23:27:33 PDT). Yu authorized the prospective v2 correction;
this is an internal split-structure acceptance, not committee approval or a model result.

## Brief summary

- Corrected the allocation objective and froze explicit group-size balance gates before
  the first real v2 run. Kept the same source, all original rows/labels and split seed 42.
- Accepted **3,220 training / 691 validation / 690 test rows**, with no numerical feature
  group crossing partitions and repeated-group rows at approximately **12.5% in each**.
- Independently verified every allocation decision, row identity, label, weight and
  structural statistic. **150 tests pass**; the original v1 remains diagnostic-only.
- No preprocessing, model or explanation was fitted. Next: training-only data/encoding
  preflight using this exact accepted split, not another allocation.

## Prospective correction and frozen execution

V1's squared-relative-deficit objective favored filling smaller partitions early. V2
uses target-weighted squared deficits, with explicit count components for both classes,
all rows, feature groups and repeated-group rows. Largest-first ordering, hash tie order,
seed and exact arithmetic remain fixed. Additional gates limit group-fraction error to
two percentage points and repeated-row composition error to three percentage points;
each partition must contain singleton and repeated groups. Row/class gates are unchanged.
These settings were fixed without Spambase model outcomes; no tolerances or seeds were
changed after v2's first allocation. This is a transparent project heuristic, not a claim
of a new optimal splitter or a random IID sample.

Protocol: `docs/SPAMBASE_SPLIT_V2_PROTOCOL.md`.
Config: `configs/spambase_split_v2.yaml`.
Frozen code/config commit: `9723173cf978ee9b9e2e5e21b4dd430b3538239a`.
Run: `artifacts/20260920-062733-688767__spambase__grouped-split-v2/`.
Fingerprint: `6d0b7926d502366e3f4dff8e905d1ffaa513f44d086bdf37d4509b314aabd39d`.

There was **one real v2 allocation**, followed by an independent replay for verification,
not a seed/candidate search. The automatic summary deliberately awaits scientific review.
The separate tracked decision in `configs/spambase_split_v2_review.yaml` pins all six run
files and accepts the split for training-only preflight; the original summary is unchanged.
Future consumers must verify that affirmative review and exact identities, not merely
an automatic `status: ok`. The v1 hold and all six pinned v1 run files remain unchanged.

## Partition results

| Partition | Rows | Feature groups | Non-spam / spam | Repeated groups | Rows in repeated groups | Repeated-row share |
|---|---:|---:|---:|---:|---:|---:|
| Training | 3,220 | 2,946 | 1,951 / 1,269 | 130 | 404 | 12.55% |
| Validation | 691 | 629 | 419 / 272 | 25 | 87 | 12.59% |
| Test | 690 | 632 | 418 / 272 | 28 | 86 | 12.46% |

The source repeated-row share is 12.54%. The largest deviation from it is **0.077
percentage points**, below the prospective three-point limit. Maximum group-fraction,
row-fraction and class-proportion errors are respectively about **0.049**, **0.019** and
**0.041 percentage points**, below their two-, one- and two-point limits. All eleven
technical checks passed. All 4,601 rows and original labels remain present exactly once.

The three conflicting pairs remain intact: `[63, 3108]` in validation, `[148, 1985]` in
training and `[478, 3213]` in test (zero-based row IDs). One pair landing in each partition
was an observed result, not a forced rule, relabeling step or reason to select a seed.

We also reviewed the predeclared group-size bins. Group counts by size:

| Partition | 1 | 2 | 3-5 | 6-10 | 11+ | Largest group |
|---|---:|---:|---:|---:|---:|---:|
| Training | 2,816 | 96 | 28 | 3 | 3 | 69 |
| Validation | 604 | 20 | 4 | 0 | 1 | 35 |
| Test | 604 | 21 | 5 | 1 | 1 | 19 |

Rare sizes are not perfectly proportional: validation has no size-6-to-10 group, and the
single largest group is in training. Unlike v1, all partitions contain singleton/repeated
groups and broadly matched repeated-row shares. No requirement that every rare bin occur
everywhere was added after allocation. These residual differences must remain visible;
group disjointness and balance do not establish sender/thread independence or population
representativeness. Retain row-primary and group-equal sensitivity evaluation.

## Verification and next milestone

Before real allocation, **150 tests passed in 27.90 seconds**, including 19 new synthetic
cases. Lint, formatting and whitespace checks passed. Tests cover the v1 counterexample,
independent rational objective calculations, mixed group sizes/labels, class-size
correlation, membership corruption, infeasible allocations, frozen policy changes,
source/audit tampering and inputs changing during execution. No real cohort was allocated
by tests; v1's production files/config/hold were not modified.

The separate real-run verification used `numpy.loadtxt` / `numpy.unique`, with no project
imports, and exact `fractions.Fraction` arithmetic. It verified all **4,207 choices**, the
unchanged hash ordering, all **4,601 memberships/labels/weights**, every partition statistic
and size bin, disjointness, fingerprint, **14 input hashes**, five manifest-listed output
hashes and six preserved v1-file hashes. The review additionally pins the v2 manifest.

The split-correction milestone is complete. Next build a dataset-aware loader that requires
the accepted review and exact row IDs, then verify training-only scaling, retained feature
order, constant-column handling, Grid mapping/padding and compute feasibility. Freeze new
model/explainer settings and budgets before test evaluation; preserve Breast Cancer guards.
No Spambase prediction or explanation outcome has been inspected. The CNN matrix remains
**3/12**, not a thesis-completion percentage. SOM2IMG authority/collision guidance and
committee scope review remain open. Raw artifacts are local/Git-ignored and need separate
backup. No Git push or committee message was sent.
