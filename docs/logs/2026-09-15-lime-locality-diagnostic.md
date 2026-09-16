# LIME locality diagnostic and thesis progress check

Report date: **2026-09-15**, Los Angeles time. Run timestamp: 2026-09-16 01:14 UTC
(September 15, 18:14 PDT). Protocol: `docs/LIME_LOCALITY_DIAGNOSTIC_PROTOCOL.md`.

## Brief summary

Completed the four-condition training-only diagnostic: **192 surrogate fits, 144 paired
contrasts, all 48 original reference fits reproduced**, and no new model queries or test
explanations. All **76 automated tests** pass.

The main result is a tradeoff, not a fix: narrower weighting improves agreement at the
explained case but does not reliably improve prediction of the same generated neighbors.
Continuous features improve neighborhood approximation but not consistently the explained
case. No alternative is promoted. This closes the bounded diagnostic; the next implementation
milestone is the full, frozen LIME v1 test/comparison runner, with fit limitations reported.

## What we compared

We reused the same eight training anchors, three explainer seeds (11, 29, 47), two saved
seed-42 tree-model probability outputs, and 24 saved neighborhood pairs. The 5,000 fitting
rows and 2,000 independent generated evaluation rows were unchanged.

- **Reference:** original binary quartile indicators and kernel width `0.75 * sqrt(30)`.
- **Narrow raw:** same indicators, kernel width `0.25 * sqrt(30)` and raw kernel weights.
- **Narrow matched mass:** same narrower weights, rescaled to match the reference's total
  fitting weight, controlling global weight scale relative to the fixed Ridge penalty.
- **Continuous, same neighbors:** anchor-centered continuous values in training-standard-
  deviation units, with original inputs and binary-distance weights held fixed. This is a
  hybrid diagnostic surrogate, **not native continuous LIME** or a replacement protocol.

All fits retain Ridge alpha 1, a free intercept and Cholesky solver. Narrowing changes total
weight as well as relative locality; the matched-mass condition separates the global-scale
effect. Representation also changes coefficient units and penalty geometry, so its contrast
is not a clean causal estimate of information lost through binning. The prospective protocol
contains the equations, source references and interpretation boundaries.

## Results

Values below are medians over **24 repeated fits per model/condition** (eight anchors x
three seeds), not 24 independent examples. Evaluation RMSE uses the **same original v1
evaluation weights in every condition**. Lower error is better; no significance tests,
population confidence intervals or parameter-selection claims are made.

| Model | Condition | Anchor absolute error | Common-evaluation RMSE | Common-evaluation R-squared |
|---|---|---:|---:|---:|
| RF | Reference | 0.0811 | 0.1263 | 0.2189 |
| RF | Narrow raw | 0.0322 | 0.1268 | 0.2144 |
| RF | Narrow matched mass | 0.0151 | 0.1266 | 0.2186 |
| RF | Continuous, same neighbors | 0.1912 | 0.0655 | 0.7927 |
| XGBoost | Reference | 0.3017 | 0.2989 | 0.2517 |
| XGBoost | Narrow raw | 0.0420 | 0.3019 | 0.2328 |
| XGBoost | Narrow matched mass | 0.0597 | 0.3000 | 0.2432 |
| XGBoost | Continuous, same neighbors | 0.2678 | 0.2190 | 0.5955 |

The paired differences tell the same cautionary story. For XGBoost, narrow-raw anchor error
fell in all 24 pairs, but common-evaluation RMSE increased in all 24 (median paired change
`+0.00513`). Matched narrowing improved both errors in only 1/24 XGBoost pairs; raw narrowing
did so in 0/24. For RF, the corresponding counts were 3/24 and 1/24. Counts describe this
small repeated-fit sample, not probabilities of success. Median paired differences need not
equal differences between the marginal medians in the table.

Continuous representation reduced common-evaluation RMSE in every pair for both models,
but improved both errors in only 4/24 RF and 14/24 XGBoost pairs. Its largest anchor errors
were 0.412 for RF and 0.757 for XGBoost. Better average neighborhood fit therefore did not
establish a reliable explanation of every anchor. All poor-fit and out-of-range surrogate
predictions remain in the saved evidence.

## Why anchor error alone is misleading here

The median total fit weight dropped from about 2,574 to 16.24 under raw narrowing. Matching
the total restored it to 2,574 but preserved the narrower relative weighting. Median anchor
self-influence rose from **0.0335** at reference to **0.7810** for narrow raw and **0.8239**
for matched narrowing. This measures how strongly the fitted anchor prediction responds to
its own fitting response; it is not causal influence or evidence of useful nearby support.

Narrowing reduced the weighted mean number of changed feature bins only from about 22.33
to 20.91. Across all 24 neighborhoods there were **no non-anchor fitting rows changing five
or fewer bins**. Reweighting did not create closer samples. This is bin-space support evidence,
not proof about clinical similarity or a diagnosis of every source of approximation error.

Own-kernel evaluation can obscure the tradeoff: XGBoost narrow-raw median RMSE was 0.2926
under its changed evaluation weights, versus 0.3019 on the common reference weights. We
therefore retain both and base the paired neighborhood comparison on the common weights.

## Verification and provenance

- Run: `artifacts/20260916-011440-771347__breast-cancer__lime__training-locality-diagnostic`.
- Clean protocol/implementation commit: `137904135ca30f7a9421ebe22479c5423ca3e19f`.
- Config: `configs/lime_locality_diagnostic_v1.yaml`, SHA-256
  `60609ad7844450dd398047e507045bbbe2786e18ab8e12deaf9e29caabd15de1`.
- All 148 original preflight files still match their frozen inventory digest. All **175
  tracked input hashes** were unchanged; models, original LIME protocol and source arrays
  were not modified. All 24 neighborhoods regenerated exactly from the training reference.
- Saved 192 coefficient/prediction bundles, 96 shared design/weight bundles, 144 paired rows,
  32 condition summaries and 24 paired-contrast summaries (overall descriptive plus separate
  seeds), source metadata, exact training scales, config and explicit quality status.
- A read-only audit independently solved weighted normal equations and recomputed common-
  evaluation metrics and anchor self-influence for all 192 fits. All saved paired rows and
  grouped summaries reproduced exactly from the diagnostic records.
- Full suite: **76 passed**; lint, formatting and whitespace checks clean. Runtime was 3.67 s
  with one native thread and no PyTorch import. This is not a cross-method efficiency benchmark.

Raw artifacts remain local/ignored; Git source synchronization is not their backup.

## Are we on track?

**Yes for the pilot implementation sequence; the full study still has important dependencies.**

| Planned area | Current evidence / remaining work |
|---|---|
| Grid and three tracing methods | Complete on the pilot dataset, including three training seeds and deletion checks |
| Tree predictive baselines and TreeSHAP | Complete on the pilot dataset across three seeds |
| LIME | Protocol, preflight and bounded diagnostic complete; full test runner/comparisons next |
| Expected CNN matrix | 3 of 12 planned conditions complete: one dataset, Grid, three seeds; this is not a thesis-completion percentage |
| Remaining full-scope decisions | Second dataset, authoritative SOM2IMG code/collision policy and committee review |

This diagnostic addressed a concrete validity risk and is now closed. No further LIME tuning
is scheduled as the default next step. Build the test runner and named-feature exports for
the original registered v1 settings, retaining approximation diagnostics beside all rank
comparisons. Do not label v1 high-fidelity merely because the code passes checks, and do not
silently substitute an alternative from this diagnostic. Any later replacement needs its
own prospective justification and validation beyond these eight training anchors.

The second dataset and SOM2IMG decisions must progress alongside that implementation;
additional pilot analysis cannot answer the encoding-comparison question. Requests for the
exact author implementation/collision guidance and second-dataset approval were raised with
Yu during this session; neither is treated as resolved without a response. The committee's
exact defense calendar remains unconfirmed, so this is not a guarantee of on-time defense.
