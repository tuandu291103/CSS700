# Session summary: Spambase Grad-CAM implementation and numerical hold

Date: **2026-10-01** (America/Los_Angeles).
Outcome: **validation-only Grad-CAM implemented and independently checked; first real run
is diagnostic-only because its frozen gradient-logit equivalence gate failed**.
The accepted training checkpoint is unchanged. No test evaluation or retraining occurred.

## What we did

- Froze the [protocol](../SPAMBASE_GRAD_CAM_PROTOCOL.md), config and implementation in
  `3093032cd7d56f4918943c2bffb1f8c2f5a89203` before any Spambase explanation run. Added
  20 synthetic tests and Torch CI coverage: **210 tests passed in 55.00 seconds**, with
  clean lint, formatting and whitespace checks before execution. The final full-suite
  rerun passed all 210 tests in 52.32 seconds; lint, formatting and whitespace remained clean.
- Added a pinned-checkpoint loader and validation-only command. Reused the saved 691
  validation inputs, original labels, predicted-class logit targets, 57-feature Grid and
  model weights. Verified 41 input hashes; no scaler refit or new encoding was needed.
- Ran Grad-CAM once on the second convolution before ReLU. Retained 691 raw 8×8 maps,
  display-normalized maps and **39,387 named-feature scores** (691×57), alongside padding,
  zero-map, top-list and group-weighted diagnostics. Kept all validation rows.
- Independently reconstructed the CNN and analytic Grad-CAM formula without project
  model/explanation imports, verified the arrays and summaries, and visually checked the
  fixed four-example figure. All outputs remain preserved, including the failure status.

## Why the result is held

Before the real run, synthetic tests exposed a small difference between gradient-enabled
and no-gradient forward passes (maximum 1.49e-8). The prospective protocol therefore
required both **bitwise-exact no-gradient replay** and gradient-enabled logits within
`rtol=1e-5, atol=1e-6` of the saved inference logits, with identical predicted classes.
These thresholds were fixed before observing the real maps and have not been changed.

The real run passed exact no-gradient replay, class/target identity, finite/nonnegative
maps, all-feature tracing, padding exclusion, unchanged weights/inputs and hook/gradient
cleanup. It failed the gradient-logit numerical gate:

- **76 of 1,382 logit values**, affecting **74 of 691 rows**, exceeded the frozen bound.
- Maximum absolute difference: **4.279613494873047e-5**, at source row 1668, class 0.
- All **691 predicted classes stayed identical**. That does not override the failed gate.
- Run status: `quality_gate_failed`; review `acceptance: false`. No production retry,
  threshold relaxation, checkpoint replacement or model tuning followed.

The independent replica reproduced each mode's stored logits exactly. A layer-wise check
found discrepancies already at the first convolution (maximum 2.15948e-4), then at the
second pre-ReLU convolution (1.36232e-3), pooled features (2.38419e-5) and classifier
(4.27961e-5). The second convolution's ReLU gates did not change. This localizes the
mode-dependent discrepancy upstream of the classifier; **the specific backend cause is
not yet isolated**. It is not evidence that the feature mapping caused the failure.

## Independent implementation checks and diagnostic observations

Using pre-ReLU activations `A`, selected-class head weights `W` and spatial area 64,
the independent float64 reference used `W * (A > 0) / 64` for the activation gradient,
then its channel-wise spatial mean, weighted activation sum and ReLU. All saved raw maps
passed the prospectively fixed `rtol=1e-5, atol=1e-7` comparison: **zero failing cells**,
maximum absolute difference **5.09e-7**. Relative tolerance contributes to this bound;
the maximum absolute difference alone is not the acceptance test.

The separate audit also checked all 41 input pins and seven manifest-listed output pins,
saved sample/target/feature identities, display normalization, every per-row score/top list,
padding/zero/tie diagnostics and every aggregate ranking under both row and group-equal
weighting. Example IDs were exactly **1824, 6, 2014, 89**, chosen prospectively as the first
source row for each true/predicted pair 00, 11, 01, 10. The figure was legible and unclipped.

In these **held diagnostic maps**, 685/691 rows had positive signal on the seven padding
pixels. Padding's fraction of raw positive map mass averaged **10.75%** (median 8.32%,
maximum 49.37%). No padding score was assigned to any named feature. All rows had nonzero
maps and at least ten positive feature scores; no positive top-5/top-10 boundary ties occurred.
This motivates explicit padding accounting, **not** a causal contribution, faithfulness
result, accepted feature ranking or claim of research novelty. The analytic check validates
the implemented formula, not whether its heatmaps explain the model faithfully.

## Retained evidence and scope

Run: `artifacts/20261001-070727-132501__spambase__grid__seed-42__validation-grad-cam-v1/`.
It began at `2026-10-01T07:07:27.132723+00:00` (00:07:27 PDT) from clean frozen code.
The whole-run performance counter measured 1.107 seconds; no cross-method speed claim follows.
All eight run-file pins, including the manifest, and the separate negative acceptance
decision are in `configs/spambase_grad_cam_validation_v1_review.yaml`.
The preserved example PNG itself has no failure watermark; treat it as diagnostic-only
under this review, not a thesis-ready accepted-results figure.

All **690 test rows remain untransformed, unscored and unexplained**. Full-source bytes
were hash-checked for provenance only in this milestone. Validation previously selected
the checkpoint, so even a repaired explanation run would be development evidence.
Original Breast Cancer, split, preprocessing and training artifacts are unchanged.

## Are we on track, and what is next?

This advances the proposal's reliable feature-tracing goal and exposes a concrete
reproducibility issue before final evaluation. The Grad-CAM milestone is **not accepted
yet**. The matrix remains **3/12 fully completed CNN conditions**, plus the accepted
Spambase training/validation checkpoint and this held explanation diagnostic. This is
not a percentage of thesis completion or a guarantee about the unconfirmed defense date.

Next: freeze a **versioned numerical-equivalence diagnostic/repair plan** for this exact
checkpoint, isolating CPU execution differences between inference and gradient modes.
Prefer a validated consistent computation path; do not simply enlarge v1's threshold
to fit observed residuals. Preserve v1, disclose any method change, and independently
review a new run before accepting explanations. Only then move to validation-only
faithfulness/input-level comparisons and the remaining seed/test protocol. Do not use
feature rankings, labels or accuracy gains to choose the numerical repair.

Committee scope and authoritative SOM2IMG collision/allocation choices remain separate
open decisions. Source/config/tests and this report are prepared for the user-requested
GitHub push; delivery confirmation belongs to the session handoff. Raw data, models and
run bundles remain local/Git-ignored: pushing source does **not** back up those artifacts.
