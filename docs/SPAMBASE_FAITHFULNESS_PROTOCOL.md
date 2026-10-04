# Spambase validation Grad-CAM deletion faithfulness v1

Prospective freeze: **2026-10-04**, before executing real validation perturbations. This
advances proposal sections III.B and V.F: do highly ranked features affect the frozen
model more than random or low-ranked features? It evaluates **Grad-CAM v2 only**, not a
three-method comparison, causal effects, final-test performance or a new CNN condition.

## Immutable sources and scope

Require affirmative `spambase_grad_cam_validation_v2_review.yaml`, its ten output pins,
all upstream identities and the original accepted seed-42 epoch-64 training checkpoint.
Verify sample/feature/target identities and inference logits. Reuse all 691 saved validation
images and their frozen predicted-class targets. No exclusions based on classification,
map quality, effect sign or padding mass. No new attribution generation or retraining.

Read saved **training images only** to calculate the component-wise median in the existing
float32 standardized Grid representation; no scaler fit and no raw-data re-encoding.
Read accepted split metadata solely to recover validation feature-group IDs and verify
their saved `1/group-size` weights. Preserve conflicting-label rows. Upstream raw files are
hash-only provenance inputs. No test features are transformed and no test model outcomes
are computed. All **690 test rows remain closed**.

## Frozen perturbation design

- Rank all 57 raw nonnegative Grad-CAM feature scores descending for **top**, ascending
  for **bottom**, with original feature index breaking ties in both directions. Retain
  zero-score features to define complete permutations; their arbitrary tie order is not
  evidence of importance. Report zero-score and positive-boundary-tie counts.
- Generate **100 random orderings per exact-feature group**, using NumPy PCG64 seed
  **20261004**, shape `(100, number_of_groups, 57)`, sorted-group-ID order, independent
  uniform keys and stable argsort. Broadcast a group's orders to all its rows. Reuse
  exactly these controls for both replacements; save all rankings. This avoids treating
  duplicate rows as independently randomized observations.
- Cumulative deletion counts: **0, 1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 57**. Replace
  only mapped feature pixels, never the seven padding cells. Count selected feature
  indices, not numerically changed values (some features already equal the replacement).
- Replacements: standardized **zero/training mean** primary; saved-training-image
  **median** sensitivity. Same references for every validation row. Keep negative drops;
  do not clip scores/probabilities or retarget after perturbing a prediction.
- Fixed no-gradient CPU inference, NNPACK enabled, one Torch thread, deterministic
  algorithms. Always preserve original source order and **32-row batches** (including
  the original final partial batch) separately for each ranking/step/replacement. Never
  pack different ranking repeats together, which could change batch arithmetic.
- Compute original inference once and require **bitwise identity** with accepted logits
  before perturbation. Reuse this exact k=0 endpoint. Evaluate every nonzero step of all
  102 rankings for both references, including k=57 to check order-independent endpoints.
  No warm-up, fitting or optimizer. Bound model execution at **300 seconds**, checking
  between forward batches; preserve failure metadata/available outputs, no automatic retry.
  Restore CPU RNG, threads, determinism/warn mode and NNPACK flags even on failure.

## Measures and uncertainty

Primary per-row summary: trapezoidal area of **signed frozen-target logit drop** versus
deleted count, normalized by 57 (AOPC). The scientific comparisons are **top minus mean
random** and **top minus bottom**; positive means larger average target-score damage.
Secondary: target-probability-drop AOPC, mean drop curves, prediction-flip rate, accuracy
and binary macro F1 at each step. Random summaries average the 100 separate curves/metrics;
do not collapse their predictions by majority vote. Probabilities use float64 softmax
of the saved float32 logits; baseline and perturbed values use the same calculation.

Report row-primary and equal-feature-group sensitivity estimates for all rows and fixed
groups: true class 0/1, predicted class 0/1, originally correct/incorrect. Within each
subgroup the sensitivity gives each represented group equal total weight (recompute group
sizes in that subgroup). Group summaries overlap and are not independent replications.

Use **1,000 paired cluster bootstrap draws**, seed **20261005**, resampling complete
feature groups with replacement and retaining all selected-group rows. Generate one set
of draws per subgroup in the stated order, reused across both references, both metrics,
both weightings and all AOPC/contrast summaries. Report percentile 95% intervals for means,
not medians. Omit intervals below ten represented groups. Random controls are averaged
first: intervals are **conditional on these 100 controls**, one validation-selected model,
one split and fixed references, not training-seed or random-control Monte Carlo uncertainty.
No p-values, method-superiority test, replacement selection or multiple-testing claim.

## Gates, outputs and interpretation

Technical pass requires clean frozen code/config, affirmative source pins, complete
rankings and exact random replay, mapped-only masks with exactly k selected features,
unchanged padding, finite outputs, exact original replay, zero k=0 drops, identical k=57
logits across ranking orders for each reference, unchanged inputs/weights and no parameter
gradients. **Top > random/bottom is a scientific result, never a software acceptance gate.**

Save append-only config/manifest/summary, rankings, replacement images, group identities,
every two-class perturbed logit, per-row AOPC, grouped curves, paired cluster intervals
and a two-panel curve figure. Save the RNG/bootstrap specification and output hashes.
Independently reconstruct the CNN without project model/deletion imports, check the masks
and replay **all** perturbation logits bitwise, independently integrate the curves and
recompute summaries/intervals (float64 metric tolerance `rtol=1e-12, atol=1e-12`; counts,
identities and logits exact). Visually review the figure, then pin the run in a separate
scientific review and report regardless of effect direction. Generated evidence stays local
and Git-ignored; source/report push is not an artifact backup.

Deletion can create unrealistic feature combinations and can favor an explanation aligned
with its replacement. Neither zero nor a marginal median is conditional imputation. A
positive result supports only this perturbation-based definition of faithfulness; a negative
result should be retained and investigated through the other planned explanation methods,
not corrected with post hoc ranking/replacement changes. Validation selected the model and
the pilot informed this protocol, so these are development results, not confirmatory test
evidence or established novelty. SOM2IMG and committee scope remain separate requirements.
