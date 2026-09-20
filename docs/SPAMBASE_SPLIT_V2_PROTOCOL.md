# Spambase grouped split v2: prospective correction before model outcomes

Date: **2026-09-19**. Yu authorized correcting the v1 allocation rule and adding group-size
balance checks. V1 remains diagnostic-only, with its code, config, original outputs and
scientific hold preserved. This amendment is informed by v1's structural defect, not by
prediction or explanation outcomes; no Spambase model has been fitted or evaluated.
Freeze this document, v2 code/config and synthetic tests before the first real v2 allocation.

## Unchanged contract

Use the same pinned source and accepted readiness audit; retain all 4,601 rows, all 57
features and all original labels, including the three conflicting pairs. Group by exact
parsed numerical features, target excluded. Keep each group wholly within one partition.
The same seed **42**, nominal row fractions **70/15/15**, largest-first group ordering and
hash-based size-tie order from v1 apply. No group swaps, seed searches or best-of-many selection.
Within each partition retain ascending source row IDs. All future models/encoders must use
the single accepted split and distinguish split seed from model-training seeds.

Source/audit checks, training-only preprocessing principles, row-primary/group-equal
evaluation and group-level uncertainty principles remain exactly as specified in v1.
No preprocessing, encoding, training or explanation runs are included in this milestone.
Whole-release values and targets have been inspected for integrity and stratification;
do not describe eventual test rows as never previously inspected.

## Corrected allocation objective

For each partition, maintain five count components in this order:

1. class-0 rows;
2. class-1 rows;
3. all rows;
4. feature groups (each group counts once);
5. rows belonging to groups of size greater than one (all members, not excess rows only).

Each component's target is `partition_fraction * full_release_component_total`.
Tentatively add the next whole group to each partition and minimize, over all partitions
and nonzero-total components, the sum:

`(allocated_count - target_count)**2 / target_count`.

All components have coefficient one. This is a target-weighted squared relative deficit,
not v1's unweighted squared relative deficit. At zero allocation the partition-dependent
cost of adding a group is proportional to its squared counts divided by the target, so
the largest quota is preferred initially; the defective early small-quota preference is
removed. Group counts and repeated-row counts are explicit terms rather than inferred
from balanced row/class totals. This is a project-defined greedy heuristic, not a guarantee
of global optimality or an IID sample.

Use exact arithmetic: write targets as `q * component_total / 20` with `q = 14, 3, 3`.
Compare sums of `(20 * allocated_count - q * component_total)**2 / (q * component_total)`
after multiplying by a common positive integer denominator. The omitted common factor
`1/20` cannot change ordering. Zero-total components contribute zero; never add an estimated
epsilon. Exact score ties use train, validation, test order. Keep the v1 hash ordering
domain and seed unchanged; this revision changes the objective, not the seed or tie order.

## Acceptance gates fixed before real v2 allocation

Retain all v1 checks: complete/disjoint row coverage; zero feature-group overlap; both
classes in every partition; original labels unchanged; row fractions within **0.01** of
their targets; per-class within-partition proportions within **0.02** of the source class
proportions; unchanged pinned inputs and clean code/config.

Add these structural requirements for this Spambase cohort:

- Each partition's fraction of the **total number of feature groups** must be within
  **0.02** of 0.70 / 0.15 / 0.15.
- Within each partition, the fraction of rows belonging to repeated-feature groups must
  differ from that fraction in the entire source by no more than **0.03** (three percentage
  points). This concerns all repeated-group members, not redundant rows beyond the first.
- Every partition must contain both singleton and repeated-feature groups. A synthetic
  corpus lacking either type cannot pass this Spambase-specific gate; report it as infeasible.

The new tolerances are prospective engineering limits on structural distortion, not
statistical confidence bounds. They allow indivisible groups but would reject v1's
0% versus roughly 42% repeated-row composition. Do not loosen them after seeing v2.

Also report, without optimizing or imposing post-hoc thresholds, group/row totals for
size bins **1, 2, 3-5, 6-10, 11+**, largest group and conflicting-group counts per partition.
Rare large or conflicting groups cannot necessarily occur in every partition; do not
split or relabel them to force that. These diagnostics inform a separate scientific
review even if numerical gates pass. Different groups need not be independent messages.

## Validation, execution and review

Synthetic tests must reproduce the v1 first-group defect and show its correction, compare
every v2 decision to an independent exact Fraction objective, exercise mixed group sizes,
class/group-size correlation and mixed-label groups, and retain failures for infeasible
allocations. Test determinism without altering original memberships, full row/label
coverage, strict config rejection, source/audit tampering and end-of-run input changes.
Do not load the real cohort through the v2 allocator during development/tests.

Run `make split-spambase-v2 PYTHON=.venv/bin/python` from the frozen clean commit. Save
append-only config, manifest, exact row IDs, all-row assignments/weights, allocation order,
fingerprint, structure diagnostics and summary. One failed attempt stays failed: no silent
fallback, reroll or overwrite. The automatic pass means **await independent scientific
review**, never automatic authorization to fit models.

Independently reparse the raw source and group numerical values, replay every choice using
exact Fraction arithmetic, verify counts/bins, row identities/labels/weights, disjointness,
fingerprint and input/output hashes. Record a separate tracked review with artifact pins
and an explicit acceptance decision; do not edit the original summary after review.
If acceptable, the next gate is training-only preprocessing/encoding/compute preflight,
not immediate model evaluation. Future loaders must require affirmative scientific
acceptance of this exact split. V1 remains rejected. Committee scope and authoritative
SOM2IMG settings remain separate open decisions, and ignored artifacts still need backup.
