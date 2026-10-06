# Session summary: Spambase Occlusion and Grad-CAM largely disagree

Date: **2026-10-05** (America/Los_Angeles).
Outcome: **validation-only single-feature Occlusion and three ranking comparisons accepted**
after independent replay. Grad-CAM agreement is low under both replacements; Occlusion
rankings also change with replacement. This is development evidence, not ground-truth
importance, a method-superiority result, final-test evidence or established novelty.

## What we did

- Froze the [protocol](../SPAMBASE_OCCLUSION_PROTOCOL.md), config, implementation, tests
  and independent verifier in `b1ccc904d1e35c2a6a56caaf8f38506fd22f4907` before the real run.
  **284 tests passed in 98.23 seconds**, including 25 new tests; lint/formatting were clean.
- Completed the Occlusion part of the planned input-level comparison first. The retained
  numerical diagnostic shows a full-gradient/inference mismatch; the Grad-CAM activation-
  boundary repair cannot provide input gradients. IG remains gated on its own preflight.
- Reused the epoch-64 seed-42 CNN, all **691 validation rows / 629 feature groups**, frozen
  predicted-class logit targets and all **57 named features**. Kept both standardized
  zero/training-mean (primary) and saved-training-image median (sensitivity) replacements.
- Replaced exactly one mapped feature per query, keeping other values/padding unchanged.
  Preserved original row order and 32-row inference batches. Saved signed logit/probability
  effects and both-class logits; no training, scaler refit or test evaluation.

## Ranking result

Occlusion ranks use **absolute signed logit drops**, as in the Breast Cancer pilot;
Grad-CAM uses raw nonnegative support. These semantics differ: a large negative Occlusion
effect can rank highly by magnitude but is not positive support for the target.
All 691 rows were eligible in every comparison; no constant-vector exclusions occurred.

| Comparison | Mean Spearman [95% paired-group interval] | Mean top-5 overlap | Mean top-10 overlap |
|---|---:|---:|---:|
| Occlusion mean vs Grad-CAM | **−0.0777 [−0.0889, −0.0667]** | 6.34% | 13.62% |
| Occlusion median vs Grad-CAM | **+0.0071 [−0.0050, +0.0184]** | 8.63% | 18.23% |
| Occlusion mean vs median | **+0.3192 [+0.3040, +0.3355]** | 55.14% | 52.59% |

Overlap is the intersection size divided by k, averaged over rows, not the percentage
of samples with identical sets. Equal-feature-group Spearman means are respectively
−0.0805 [−0.0913, −0.0696], +0.0021 [−0.0095, +0.0132] and +0.3241 [+0.3115, +0.3367];
the interpretation does not depend on overweighting repeated feature vectors.

Intervals use **1,000 paired whole-feature-group bootstrap draws**, retaining group members,
with the same draws across comparisons/metrics/weightings within each analysis subgroup.
All draws had eligible members. These are conditional mean intervals for one model/split
and fixed references, not training-seed uncertainty, p-values or formal superiority tests.

Predeclared subgroup mean Spearman values (row-primary; full intervals retained in artifacts):

| Subgroup | Rows | Mean vs CAM | Median vs CAM | Mean vs median Occlusion |
|---|---:|---:|---:|---:|
| True non-spam | 419 | −0.0939 | −0.0346 | +0.3179 |
| True spam | 272 | −0.0528 | +0.0713 | +0.3212 |
| Predicted non-spam | 438 | −0.0913 | −0.0420 | +0.3113 |
| Predicted spam | 253 | −0.0540 | +0.0921 | +0.3330 |
| Originally correct | 628 | −0.0830 | +0.0071 | +0.3263 |
| Originally incorrect | 63 | −0.0244 | +0.0067 | +0.2487 |

The near-zero aggregate median–CAM result mixes slightly negative and positive class
summaries. Subgroups overlap and are not independent replications; no post hoc exclusions
or multiplicity-adjusted claims were made.

## Replacement and tie diagnostics

| Replacement | Negative logit effects | Exactly zero | Positive | Input value already equals replacement |
|---|---:|---:|---:|---:|
| Zero/training mean | 16,467 | 0 | 22,920 | 0 |
| Training median | 3,580 | 29,336 | 6,471 | 29,336 |

Each row summarizes **39,387 sample–feature pairs**, not independent observations.
Median replacement leaves **74.48%** of those values unchanged, yielding exactly zero
effects. **162/691 rows** have fewer than ten nonzero median effects and a zero-valued
top-10 boundary tie. Their top-10 sets include index-selected zero effects, not ten identified
important features. All ties were retained and flagged; there were no top-5 boundary ties.
This qualifies the median top-10 overlap; it does not select a new replacement or ranking rule.

## Verification and evidence

Run: `artifacts/20261006-021949-097583__spambase__grid__seed-42__validation-occlusion-v1/`.
Started **19:19:49 PDT on October 5** (directory timestamp is October 6 UTC) from clean
frozen code. Inference took 1.20 seconds; whole-run time 1.99 seconds, UTC span 1.96 seconds.
These are execution diagnostics, not comparative efficiency claims. The run made **2,530
batched forwards / 79,465 row evaluations**, including original replay, and no optimizer steps.

The [independent verifier](../../scripts/verify_spambase_occlusion.py) checked **74 input
hashes** and nine manifest-listed output hashes, reconstructed the CNN and replayed **every
single-feature logit bit-for-bit** without project model/Occlusion/statistics imports.
All **140,964 prior k=1 deletion row-queries** also match exactly (two replacements × 102
orderings × 691 rows). This consistency check is not independent faithfulness evidence for
Occlusion. Original logits, unmodified-input logits and padding checks passed.

Manual tied ranks and separately implemented group resampling reproduced all **2,073
per-row comparison records**, **42 group/weighting summaries** and **126 metric means and
intervals** at `rtol=atol=1e-12`. The three-panel figure was visually inspected. Source files
and model weights remained unchanged. Ten output files are pinned in
`configs/spambase_occlusion_v1_review.yaml`. To verify the retained run without a new study:

```bash
.venv/bin/python scripts/verify_spambase_occlusion.py \
  artifacts/20261006-021949-097583__spambase__grid__seed-42__validation-occlusion-v1
```

## Proposal alignment and next step

This advances the proposal's **feature-level explanation agreement** on the second dataset.
The methods do not identify interchangeable rankings on this model, and replacement affects
Occlusion itself. It does not establish which method is correct or why they disagree.
Finite marginal replacement ignores some interactions, can create unrealistic inputs and is
not conditional imputation. Agreement is not faithfulness; the previous negative primary
Grad-CAM deletion result and positive median sensitivity remain unchanged.

Next: freeze a bounded **input-gradient numerical preflight for IG**, with explicit predictor
identity, gradient checks, training-only baselines, completeness and compute limits before
new IG explanations. Do not reuse the Grad-CAM-specific repair or loosen old tolerances after
observing results. IG, Occlusion cumulative deletion, remaining seeds and final evaluation
were not performed here. The matrix remains **3/12 complete CNN conditions**, plus Spambase
seed-42 validation training/tracing/deletion/Occlusion progress; all **690 test rows remain
untransformed, unscored and unexplained**. SOM2IMG source/collision guidance and committee
scope remain separate gates. Source/reports are Git delivery, not backup of ignored local
raw data/model/run bundles; original Breast Cancer evidence is unchanged.
