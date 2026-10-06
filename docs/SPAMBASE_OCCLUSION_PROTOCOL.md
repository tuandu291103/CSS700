# Spambase validation Occlusion and Grad-CAM comparison v1

Prospective freeze: **2026-10-05**, before new real-data Occlusion queries. This advances
the proposal's input-level explanations and feature-ranking agreement. It is one bounded
part of the planned IG/Occlusion comparison, not completion of both methods or a new full
CNN condition. The retained gradient-mode discrepancy still needs a separate prospective
input-gradient protocol; the Grad-CAM activation-boundary repair is not an IG repair.

## Sources and execution

Consume the affirmative pinned Grad-CAM v2 review through the existing source loader,
retaining its exact checkpoint, all 691 saved validation images, 57 named-feature coordinates,
frozen predicted-class logit targets, training-derived references and feature-group IDs.
Also require the accepted deletion-v1 review and all nine pinned files. Verify its config,
source identities, saved original logits, feature scores, references and group IDs. Read
upstream raw bytes only for hash verification; no fitting, new encoding or test evaluation.

For each of the **57 features separately**, replace its one mapped pixel in every row.
Keep all other feature values and all seven padding pixels unchanged. Primary replacement
is standardized **zero/training mean**; sensitivity is the median of saved float32 training
images, exactly as in deletion v1. Do not select the median because its deletion result was
favorable. A replacement can equal the original value; count that explicitly, not as an error.

Use the accepted no-gradient CPU prediction path, NNPACK enabled, one thread and deterministic
algorithms. Preserve source order and batches of **32** (including final 19), separately for
each replacement/feature. Never pack feature variants into a different batch shape. Reproduce
the original logits **bit-for-bit before perturbation**. Bound inference at **120 seconds**,
checked between batches and at completion. Restore RNG, threads, determinism and backend flags;
no parameter gradients or weight updates. Save NaN-initialized partial logits on failure;
one production run, no automatic retry. Pure diagnostic replay is not a second experiment.

Save both-class logits with axes `(replacement, sample, feature, class)`. Signed target
logit drop is original minus perturbed logit, computed in float64 from saved float32 outputs.
Secondary probability drop uses float64 softmax consistently for both endpoints. Positive
means replacing that feature reduces the frozen target score; negative values are retained.
Occlusion is a finite replacement effect, not an additive attribution/completeness identity.

Cross-check **every saved k=1 deletion query** (both references, all 102 prior orderings,
all rows) by selecting the corresponding single-feature Occlusion logits. Require bitwise
identity. This is consistency with an earlier computation, not independent faithfulness
evidence for Occlusion. Preserve the primary negative Grad-CAM deletion result unchanged.

## Frozen rank comparisons

Primary ranks use **absolute signed logit drops**, matching the Breast Cancer pilot's
Occlusion convention; retain the signed values separately. Grad-CAM is its existing raw
nonnegative support map. This contrasts effect magnitude with positive spatial support,
not two identical explanation semantics. No score-magnitude comparison across methods.

In fixed order compare: `occlusion_zero_vs_grad_cam`, `occlusion_median_vs_grad_cam`,
`occlusion_zero_vs_median`. For each original row calculate Spearman with average tied ranks
and top-5/top-10 set overlap fractions. Stable original feature index breaks top-k ties.
Retain exact boundary-tie flags for each side/k. If either magnitude vector is constant,
retain the row and mark all three agreement metrics undefined; do not impute zero correlation
or interpret arbitrary constant-vector top-k agreement. All targets are frozen and identical.
Reuse the tested per-row ranking code; do not reuse its independent-row bootstrap.

Summaries: all rows, true class 0/1, predicted class 0/1, correct and incorrect, in that order.
Report total/eligible rows and feature groups, exclusions, means and conditional 95% intervals
for each metric, with row-primary and equal-feature-group sensitivity. Equal-group weighting
uses the eligible member count **within each analysis subgroup/comparison**. No case selection
based on agreement or model correctness; the subgroups are overlapping descriptive summaries.

Use NumPy PCG64 seed **20261006**, **1,000 paired whole-feature-group bootstrap draws**.
Draw all represented subgroup groups with replacement, retaining their members; reuse the
same draw indices across all comparisons, metrics and weightings within that subgroup.
Undefined metric rows contribute neither value nor weight; do not resample only favorable
rows. Report valid-draw count; omit intervals if fewer than ten eligible groups or any draw
has no eligible members. Percentile intervals describe means, not medians. They are conditional
on the fixed model, validation split, references and eligibility, not training-seed uncertainty.
No p-values, formal superiority claim or multiplicity-adjusted subgroup inference.

## Gates, records and interpretation

Technical acceptance is independent of agreement magnitude or effect sign. Require verified
sources, exact original and k=1 replay, exactly-one-feature masks, unmodified other features/
padding, finite scores, exact logits for unchanged-input rows, unchanged weights/files and
no test outcomes. Save append-only config, manifest, feature mapping, design identities,
all perturbed logits, signed logit/probability drops, per-row rank/tie records, group summaries
and a three-panel descriptive agreement distribution. Inspect the figure visually.

Independent read-only verification must reconstruct the CNN without project model/Occlusion/
statistics imports, rebuild references and identities, replay all single-feature logits and
k=1 identities exactly, recompute signed drops, tie-aware ranks and every mean/interval.
Use exact identities/counts/logits, metric `rtol=atol=1e-12`. Pin all outputs in a separate
review even when rankings disagree. Generated runs stay local/Git-ignored.

Agreement is not truth, causality or faithfulness. One-feature replacement does not capture
all interactions; it may create unrealistic combinations and is not conditional imputation.
This model was validation-selected and prior results informed the workflow, so these are
development findings, not confirmatory test results or established novelty. Keep all **690
test rows closed**. Do not launch IG, new seeds, Occlusion cumulative deletion, SOM2IMG or
test evaluation as part of this run. Those require their own prospective designs/reviews.
