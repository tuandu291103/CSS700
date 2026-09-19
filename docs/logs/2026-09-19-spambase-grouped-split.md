# Spambase grouped split v1: verified execution, scientific hold

Date: **2026-09-19**. Scope: continue with Spambase as the working second dataset;
freeze and inspect a whole-feature-group split while preserving all rows and labels.
No committee approval is claimed. **No preprocessing or model fitting occurred.**

## Brief summary

- Implemented and prospectively froze one deterministic 70/15/15 grouped allocation
  with split seed 42, exact-arithmetic decisions and unchanged conflicting labels.
- The first allocation produced 3,221 training / 690 validation / 690 test rows, with
  complete coverage, both classes, balanced class proportions and zero feature-group overlap.
- Independent verification reproduced every one of 4,207 allocation decisions and all
  4,601 row memberships/labels/weights. Source and saved-output hashes match.
- Scientific review found a design defect: **all 183 repeated-feature groups were held
  out**, and training contains only unique vectors. The split is **diagnostic-only**;
  technical checks passing does not authorize preprocessing or training.

## Frozen execution and identity

Code, protocol, source/audit pins, policy and synthetic tests were committed before the
first real allocation at `d9b17a82cbe30d02f05e01dc4a08dc62f89f12f0`.
Configuration: `configs/spambase_split_v1.yaml`.
Protocol: `docs/SPAMBASE_SPLIT_PROTOCOL.md` at that commit.
Run: `artifacts/20260919-093236-104613__spambase__grouped-split-v1/`.

The run retained exact config, manifest, partition row IDs, all-row assignments with
evaluation weights, whole-group allocation order and summary. The original source and
accepted readiness audit remain unchanged. This was one allocation, not a seed search.
The later independent replay verifies that same allocation; it does not select an alternative.

Fingerprint: `6253e124d4963c3e6a53ff11e29c3b0ce4dc7b6195915cb361bd19c56d9fa510`.
All six run-file hashes and the overriding scientific hold are tracked in
`configs/spambase_split_v1_review.yaml`. Original outputs have **not** been edited to hide
the technical runner's `go_to_training_only_preflight` result. That automatic result is
superseded by this scientific review. Future loaders must require affirmative scientific
acceptance, not merely `status: ok`; v1 must not become a modeling source.

## Observed partition structure

| Partition | Rows | Feature groups | Non-spam / spam | Repeated-feature groups | Rows in repeated groups |
|---|---:|---:|---:|---:|---:|
| Training | 3,221 | 3,221 | 1,952 / 1,269 | 0 | 0 |
| Validation | 690 | 493 | 418 / 272 | 91 | 288 |
| Test | 690 | 493 | 418 / 272 | 92 | 289 |

Class proportions differ from the full release by less than 0.016 percentage points;
row fractions are extremely close to 70/15/15. All original labels are unchanged. The
three conflicting pairs are all in test, still intact. The largest group (69 rows) is
in validation; the largest test group has 35 rows.

However, duplicate-group rows constitute **0% of training**, **41.74% of validation**
and **41.88% of test**, versus **12.54% of the source release**. Exact disjointness is
necessary but insufficient: this allocation systematically separates group multiplicity
between training and held-out data. It is not accepted as our intended general benchmark
split. It is also not evidence of poor predictive performance or an unusable dataset.

## Diagnosis, not an outcome-driven revision

The v1 objective sums squared relative deficits, dividing each squared count difference
by its target **squared**. Starting from empty partitions, adding a group generally reduces
this objective faster in a smaller partition, until those deficits close. Largest-first
group ordering therefore routes the early repeated groups into validation/test. The
code matches the protocol; the protocol's balancing objective and acceptance gates were
insufficient, rather than there being a hidden data-loader error.

An independent synthetic counterexample isolates the mechanism: in a population of 1,000
rows with 600/400 labels, a ten-row group containing six/ four labels changes the initial
objective by `-417/4900` (about -0.0851) in training, but `-29/75` (about -0.3867) in
validation or test. V1 therefore chooses validation, the exact tie winner, even though
training has the largest quota. A post-run synthetic regression documents this behavior;
it was not part of the pre-execution acceptance gate.

No scaler, model, prediction, explanation, performance metric or test outcome informed
this review. The additional observation concerns split structure only. Preserve v1;
do not retrospectively call multiplicity balance a preregistered gate or alter its results.

## Verification and status

Before execution: **130 tests passed in 20.92 seconds**, including 16 new synthetic cases;
lint, formatting and whitespace checks passed. The suite covered exact objective comparison
against independent Fraction arithmetic, deterministic order, conflicting groups, corrupted
memberships, infeasible balance, source/audit tampering and changes during execution.

Real-run verification used `numpy.loadtxt` / `numpy.unique` without importing the splitter.
It checked all 4,601 memberships, labels, group IDs, primary/group-equal weights; all 4,207
groups and their order; all choices using exact `fractions.Fraction` arithmetic; all partition
counts; zero numerical feature-group overlap; fingerprint serialization; **14 input hashes
and five manifest-listed output hashes**. All checks passed. The later review pins the
manifest itself as the sixth output identity.

After adding the post-run diagnostic regression: **131 tests passed in 21.10 seconds**;
lint, formatting and whitespace checks remain clean. Passing tests establish implementation
behavior, not the scientific suitability of the retained v1 allocation.
Raw sources and run bundles remain local and ignored by Git; hash pins are not backups.
No Git push or committee message was sent.

## Next bounded milestone and on-track check

Recommend a **prospective v2 allocation policy**, keeping the same source, inclusion,
original labels, fixed split seed and nominal fractions. Correct the early small-partition
preference, test mixed group-size fixtures and explicitly freeze group-multiplicity
diagnostics/acceptance criteria before its first real allocation. A target-weighted squared
deficit objective is a candidate to validate synthetically, not an already approved or run
replacement. Do not change seeds to obtain a favorable partition.

Review v2's structural audit before the training-only preprocessing/encoding/compute
preflight, then freeze the Grid model/explainer execution contract. The v1 policy/review
milestone is complete, but **an accepted Spambase split is still outstanding**. This caught
a problem before model evaluation and does not add a research novelty or trained condition.
The CNN matrix remains **3/12**; SOM2IMG authority/collision guidance and committee scope
review are still open. No claim about meeting the defense calendar follows from this step.
