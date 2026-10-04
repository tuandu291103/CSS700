# Session summary: Spambase Grad-CAM faithfulness depends on replacement

Date: **2026-10-04** (America/Los_Angeles).
Outcome: **one validation-only deletion experiment accepted as technically verified,
replacement-dependent evidence**. The primary mean-replacement result is negative;
the median sensitivity is positive. Keep both, without promoting the favorable sensitivity
to primary. This is not replacement-robust faithfulness, final-test evidence or established novelty.

## What we did

- Froze the [protocol](../SPAMBASE_FAITHFULNESS_PROTOCOL.md), config, evaluator, tests and
  independent verifier in `523ba026a4395cae6029c4302e18a6e7318ff657` before real perturbation.
  **259 tests passed in 80.78 seconds**, including 24 new checks; lint/formatting were clean.
- Reused the accepted epoch-64 seed-42 CNN, all **691 validation rows / 629 exact-feature
  groups**, 57 mapped features and accepted Grad-CAM v2 rankings. No new model or attribution.
- Deleted features in top, bottom and **100 group-shared random orders**, at 12 frozen
  cumulative budgets from 0 through 57. Used standardized zero (training mean) as primary,
  saved-training-image median as sensitivity. Never masked the seven padding cells.
- Retained negative effects and every row. Stored every perturbed logit, ranking,
  replacement, curve and per-row AOPC. Independently replayed the entire experiment and
  visually checked the two-panel figure; no retries or post-result policy changes.

## Main result

AOPC is the trapezoidal area of the signed, frozen-predicted-class logit drop versus
deleted-feature count, divided by 57. A positive **top-minus-control** means top-ranked
deletion causes more average target-score damage across the registered budgets.

| Weighting / replacement | Top minus random, mean [95% interval] | Top minus bottom, mean [95% interval] |
|---|---:|---:|
| Row-primary / zero (primary) | **−0.1163 [−0.1624, −0.0714]** | −0.2761 [−0.3567, −0.1936] |
| Row-primary / training median | **+0.0730 [+0.0247, +0.1149]** | +0.2012 [+0.1326, +0.2696] |
| Equal-feature-group / zero | −0.1033 [−0.1488, −0.0596] | −0.2443 [−0.3148, −0.1722] |
| Equal-feature-group / training median | +0.0766 [+0.0267, +0.1197] | +0.2029 [+0.1285, +0.2742] |

Row-weighted logit AOPCs (top / random / bottom) are
**1.3427 / 1.4590 / 1.6188** for zero and **0.9379 / 0.8649 / 0.7368** for median.
Secondary probability-drop AOPC also reverses sign: row-primary top-minus-random is
**−0.02374 [−0.02992, −0.01739]** versus **+0.03576 [+0.02437, +0.04698]**;
top-minus-bottom is −0.06853 versus +0.07107. Equal-group probability summaries retain
the reversal. This is an integrated-curve result, not dominance at every deletion budget.

Intervals use **1,000 paired whole-feature-group bootstrap draws**, retaining group members,
after averaging the 100 fixed random controls. They are conditional on one model/split,
fixed references and controls; they do not include training-seed or control Monte Carlo
uncertainty. No p-values or multiplicity-adjusted subgroup claims were made.

## Predeclared subgroup check

Row-primary logit AOPC **top minus random**, with the same conditional 95% interval:

| Subgroup | Rows / groups | Zero | Training median |
|---|---:|---:|---:|
| True non-spam | 419 / 377 | −0.0655 [−0.1255, −0.0079] | +0.1044 [+0.0509, +0.1486] |
| True spam | 272 / 253 | −0.1945 [−0.2769, −0.1171] | +0.0247 [−0.0625, +0.1093] |
| Predicted non-spam | 438 / 393 | −0.0392 [−0.0893, +0.0127] | +0.1413 [+0.1016, +0.1834] |
| Predicted spam | 253 / 236 | −0.2497 [−0.3413, −0.1611] | −0.0451 [−0.1457, +0.0416] |
| Originally correct | 628 / 572 | −0.1014 [−0.1469, −0.0528] | +0.0765 [+0.0304, +0.1221] |
| Originally incorrect | 63 / 58 | −0.2649 [−0.4882, −0.0917] | +0.0384 [−0.1798, +0.2069] |

The aggregate median advantage is not uniform across subgroups. These overlapping summaries
are not independent replications; a conflicting-label feature group can appear in both
true-class groups. All subgroup metrics, both weightings and both contrasts are retained.

## Verification and retained evidence

Run: `artifacts/20261004-234846-380780__spambase__grid__seed-42__validation-faithfulness-v1/`.
Started **16:48:46 PDT** from clean frozen code; inference took 15.20 seconds and the
whole run 16.36 seconds (UTC span 16.33 seconds). These are execution diagnostics, not a
cross-method efficiency comparison. There were **49,390 batched forward calls**, representing
1,551,295 row evaluations including baseline replay, and zero optimizer steps.

The [independent verifier](../../scripts/verify_spambase_faithfulness.py), without project
model/deletion/statistics imports, checked **63 source hashes**, eight manifest-listed output
hashes, identities, training-derived references, complete rankings and exact mapped-only masks.
It replayed **all perturbation logits bit-for-bit** and recomputed all **280 AOPC/interval
records and 1,008 curve records** at `rtol=atol=1e-12`. Original logits and zero-deletion
endpoints are exact; full deletion is order-independent; inputs and model weights are unchanged.

All 691 rows contain at least one zero feature score, not an all-zero explanation. Complete
rankings break such ties by original feature index; this ordering is not evidence of importance.
There were no positive top-5/top-10 boundary ties. Replacement images differ by at most
0.5668 standardized units. No rows or negative effects were removed.
All nine run files and the separate review decision are pinned in
`configs/spambase_faithfulness_v1_review.yaml`. The original training, v2 tracing and held-v1
reviews remain unchanged. To verify the retained run without a new experiment:

```bash
.venv/bin/python scripts/verify_spambase_faithfulness.py \
  artifacts/20261004-234846-380780__spambase__grid__seed-42__validation-faithfulness-v1
```

## Proposal alignment and next step

This advances proposal sections III.B and V.F: evaluating whether feature rankings reflect
the model's behavior. The observed sign reversal is a useful **development finding** about
replacement sensitivity. It does not establish why it happens, which features are causal,
or that Grad-CAM generally succeeds/fails. Marginal replacement can create unrealistic
feature combinations; neither reference is conditional imputation. Validation selected this
model, the pilot informed the protocol, and zero-score tie ordering can affect later budgets.

Next freeze the **Spambase input-level IG/Occlusion validation comparison**: numerical
identity checks, training-only references, completeness, signed/absolute ranking semantics,
compute limits and comparisons must be fixed before new explanation output. The Grad-CAM
activation-boundary repair does not provide input gradients. Retain both deletion references
and the primary negative result; no baseline or model tuning to reverse it.

The expected matrix remains **3/12 completed CNN conditions**, plus Spambase seed-42
training/validation, tracing and this conditional deletion study. All **690 test rows remain
untransformed, unscored and unexplained**. Remaining explanations, seeds and final-test design
are pending; authoritative SOM2IMG collision/settings guidance and committee scope remain
separate full-scope gates. No defense-calendar assurance or committee approval is implied.
Source/config/tests/reports are Git delivery; ignored raw data/models/run bundles remain local
and are not backed up by a source push. Breast Cancer results remain unchanged.
