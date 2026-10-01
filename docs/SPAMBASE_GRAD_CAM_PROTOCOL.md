# Spambase validation-only Grad-CAM v1

Draft begun 2026-09-27; prospective freeze **2026-10-01**, before viewing Spambase explanation outputs. This
extends the accepted seed-42 training checkpoint toward the proposal's named-feature
tracing goal. It does not open test evaluation or retrain/select another checkpoint.

## Fixed source and method

- Require the affirmative `configs/spambase_grid_seed_42_v1_review.yaml`, all eight pinned
  training files and 31 upstream input identities. Verify saved validation IDs against the
  accepted split, original feature names, padding, prediction metadata and checkpoint.
  Load only saved validation arrays for model inference. No scaler fit or new encoding.
- Explain **all 691 validation rows**, ascending source order, with their **saved predicted
  class logit** as target. Retain correct/incorrect cases and both predicted classes.
  No true-class retargeting or selection of visually attractive examples.
- Reuse existing Grad-CAM: `features.2`, the second convolution **before its ReLU**;
  spatially average each channel's logit gradient, take the weighted activation sum,
  then ReLU for positive evidence. Activations/maps are already 8×8, so no resize occurs.
  This is the same layer/sign convention as the Breast Cancer pilot, not standard CAM
  at a different post-ReLU layer. Negative/inhibitory evidence is not represented.
- CPU, one Torch thread, deterministic algorithms, batch size **32**, no warm-up or
  optimizer. One no-gradient identity-check forward pass and one batched Grad-CAM pass
  per validation row. A **120-second** limit checked
  between batches/after the last batch bounds execution; preserve failures without retry.
  Restore CPU RNG/thread/determinism settings and remove hooks. Do not accumulate parameter
  gradients or change model state. Independent verification is separate from this pass.
- Save raw maps and raw 691×57 traced feature scores. Display normalization uses each
  full 64-pixel map's maximum, clamped at `1e-12`, with zero maps kept zero. Normalized
  maps/scores are for display only, never primary ranking or quantitative comparisons.

## Padding, ranks and summaries

Use the exact one-to-one Grid feature mapping and boolean occupancy mask: 57 features,
seven padding pixels. A zero input padding value does **not** imply a zero Grad-CAM value
at that location, because activation receptive fields and biases span neighboring cells.
Never assign padding to a feature. Save its raw positive-map mass and fraction of total
raw map mass per row. This fraction is a spatial diagnostic, not a causal contribution or
a completeness identity with the class logit. For all-zero maps its denominator is zero:
record `null`, not a fabricated fraction, and retain the row.

Rank raw feature values descending, breaking exact ties by original feature index. Top-5
and top-10 lists contain **positive scores only**, so may be shorter/empty; retain all 57
numerical scores and flag zero maps, positive counts and positive ties across each boundary.
Do not claim arbitrary zero-score tie order represents feature evidence. Grad-CAM is a
spatial coarse method; tracing its pixels does not establish independent feature effects.

Save mean-raw-score ranks with original-row weights and `1/feature-group-size` weights,
separately by predicted target class. Also retain an explicitly pooled descriptive table;
mixing target classes is not a comparable global causal-importance scale. Keep all rows,
including zero maps and conflicting-label groups. No significance tests or confidence
intervals. No threshold/hyperparameter selection based on these rankings.

The example figure uses the first source-ordered row for each available true/predicted
pair, in fixed order **00, 11, 01, 10**. Omit absent pairs. Show display heatmaps with gray
padding and up to five positive raw traced features; label validation-only status. Heatmap
normalization includes padding even though padding is masked for display. Raw arrays retain
all padding values and per-sample diagnostics quantify them.

## Gates and verification

Before execution, commit protocol/config/code and pass synthetic tests. Require clean code,
unchanged hashes, finite arrays, nonnegative raw maps, display values in [0,1], input-sized
maps, all 57 feature identities, exact no-gradient inference logits and predicted-class
targets relative to frozen predictions, exact
mapping reads and padding-exclusion probes under reversed record order. Require unchanged
weights, no accumulated parameter gradients and no leaked hooks. Zero maps or large padding
fractions are **reported**, not technical failure or a reason to retune.

Synthetic testing before freeze found that enabling gradients changes forward rounding
(maximum 1.49e-8 in the small synthetic diagnostic), while no-gradient inference matches
stored logits bitwise. Therefore preserve **both** outputs: require exact no-gradient
inference replay, gradient-enabled logits within `rtol=1e-5, atol=1e-6` of the stored logits,
and exactly the same predicted classes/targets in both modes. This numerical gate was
fixed before any real Spambase Grad-CAM run; the checkpoint, layer, targets and attribution
formula are unchanged. Larger logit disagreement or a prediction flip fails the run.

After execution independently reconstruct the model without project Grad-CAM/tracing imports.
For pre-ReLU convolution activation `A[k,h,w]`, pooled head weight `W[c,k]`, and spatial area
`S=64`, use the closed-form gradient `W[c,k] * 1[A>0] / S`. Spatially average this gradient
and form the ReLU-weighted activation sum. Verify every map at `rtol=1e-5, atol=1e-7`,
with zero derivative at A=0. Check exact saved logits, direct feature-index extraction,
display normalization, padding/zero/tie diagnostics, sample/top-list identities and both
aggregate weightings. This tests implementation, **not faithfulness**.

Outputs are append-only: config/manifest, raw and display maps, traced scores, IDs/labels/
targets/group weights, feature map, per-sample diagnostics/top lists, aggregate ranks,
example PNG and summary. Preserve partial outputs on failure and all upstream files.
Record separate acceptance pins and a brief report; raw outputs remain local/Git-ignored.

All **690 test rows remain outside encoding, inference and explanations**. Full raw source
bytes are hash-checked for provenance, not used as test outcomes. Validation selected the
model, so these are development diagnostics. Next freeze a validation-only faithfulness
check and input-level explanation comparison before final-test and seed expansion; no
research novelty or reliable feature importance is established by this milestone alone.

```bash
make spambase-grad-cam PYTHON=.venv/bin/python
```
