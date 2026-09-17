# Frozen LIME test pilot: agreement is not approximation quality

Report date: **2026-09-17**. Experiment dated **2026-09-16**; source commit
`413f329d81cca0ea682f31cbb663462c250a10a8`. Scientific protocol:
`docs/LIME_COMPARISON_PROTOCOL.md`; implementation contract: `docs/LIME_TEST_EXECUTION.md`.

## Brief summary

Completed the original v1 LIME test pilot: **516 fits on 86 test cases**, 258 shared
neighborhood pairs, 15,480 named-feature rows and 1,290 paired-comparison rows. All technical
checks pass, all **92 automated tests** pass, and a read-only saved-output audit reproduced
every fit and comparison. No diagnostic variant was substituted and no poor-fit case removed.

The useful result is a distinction: LIME rankings agree substantially with same-model
TreeSHAP and repeat reasonably across sampling seeds, yet the surrogate often approximates
the model poorly. Rank agreement and stability therefore do not establish explanation fidelity
in this pilot. This is evidence for the thesis, not an established novel contribution or a
general judgment about LIME. The encoding-comparison question remains unanswered.

## Frozen scope

Use the original seed-42 RF and XGBoost models, existing 397/86/86 split, 30 named features,
training-fitted quartile indicators, width `0.75 * sqrt(30)`, Ridge alpha 1, 5,000 fitting
rows and 2,000 independent generated evaluation rows. Explainer seeds 11 (primary), 29 and
47 change neighborhood sampling, not model training. Each neighborhood is shared across the
two families; each fit targets that model's frozen predicted class. All 86 cases, including
misclassifications, remain present. This is the previously analyzed pilot test set, not a new
external validation cohort. The implementation was committed before any test LIME generation.

## Approximation quality comes first

Each value is a median over **86 distinct cases within one family/seed**, except the final
column, which counts anchor surrogate predictions outside [0, 1]. Probability errors are
absolute probability units, not classification-error rates. Evaluation measures prediction of
independently generated neighbors, not real-world generalization or causal validity.

| Model | LIME seed | Anchor absolute error | Evaluation RMSE | Evaluation R-squared | Out-of-range anchors |
|---|---:|---:|---:|---:|---:|
| RF | 11 | 0.1073 | 0.1259 | 0.2088 | 19/86 |
| RF | 29 | 0.1018 | 0.1266 | 0.2057 | 18/86 |
| RF | 47 | 0.1141 | 0.1261 | 0.2226 | 17/86 |
| XGBoost | 11 | 0.2170 | 0.2998 | 0.2393 | 59/86 |
| XGBoost | 29 | 0.2391 | 0.3010 | 0.2363 | 61/86 |
| XGBoost | 47 | 0.2127 | 0.3027 | 0.2335 | 62/86 |

For the primary seed, largest anchor errors were 0.272 (RF) and 0.682 (XGBoost). Unconstrained
surrogate predictions were retained without clipping. Passing the software checks cannot turn
these approximation limitations into high-fidelity explanations. The difference between model
families is descriptive, not a cross-model causal comparison or an explainer-performance ranking.

## Same-model LIME–TreeSHAP agreement

Compare absolute LIME coefficients with absolute predicted-class TreeSHAP values from the
accepted full-397-background run. All 86 rows are eligible in every comparison. Intervals
are 95% percentile intervals for the **mean** from 1,000 paired sample resamples (seed 2026),
conditional on the fixed model, split, explainer seed and background; they are not intervals
for the median or population-of-seeds estimates.

| Model | LIME seed | Mean Spearman [95% interval] | Median Spearman |
|---|---:|---:|---:|
| RF | 11, primary | 0.7601 [0.7284, 0.7913] | 0.7918 |
| RF | 29 | 0.7532 [0.7196, 0.7865] | 0.7882 |
| RF | 47 | 0.7664 [0.7311, 0.7993] | 0.7989 |
| XGBoost | 11, primary | 0.7877 [0.7652, 0.8123] | 0.8049 |
| XGBoost | 29 | 0.7837 [0.7603, 0.8082] | 0.8049 |
| XGBoost | 47 | 0.7850 [0.7608, 0.8083] | 0.7996 |

Median top-5 and top-10 overlap fractions were 0.8 for all six comparisons (four of five and
eight of ten features). These quantities describe rankings, not a shared contribution scale:
same-quartile membership coefficients and interventional SHAP values have different meanings.
SHAP is a comparator, not ground truth; no formal difference or superiority test was added.

## Sampling stability, cross-model agreement and groups

- Across the three separate within-RF seed pairs, median Spearman ranged 0.8378–0.8610;
  the corresponding XGBoost range was 0.8914–0.8950. Each pair contains 86 cases. Median
  top-5 overlap was 1.0 and top-10 overlap 0.9 for every pair. These ranges summarize separate
  estimates, not a pooled independent sample or a guarantee of stability for every case.
- RF–XGBoost LIME mean Spearman ranged 0.5997–0.6053 across the three matching explainer
  seeds. Each comparison had 84 eligible cases and two different-target exclusions; excluded
  rows remain in the exports. Median top-5/top-10 overlaps were 0.6/0.7.
- No constant-vector exclusions or exact top-5/top-10 cutoff ties occurred. Predictions are
  unchanged: RF has 81 correct / five incorrect cases; XGBoost has 83 / three. Predicted-class
  0/1 counts are 33/53 for RF and 31/55 for XGBoost. Small error-group intervals are omitted
  under the registered minimum of ten eligible cases, not promoted as subgroup evidence.
- Saved 90 agreement summaries and 30 fit-quality summaries, including zero-count groups.
  Every comparison row carries each LIME side's fit diagnostics. No sample/seed pooling or
  fit-quality filtering was applied.

## Verification, artifacts and timing limitation

Run: `artifacts/20260916-205920-546708__breast-cancer__lime__model-seed-42__test-pilot`.
The original config hash remains
`bf6c361aadafeddf57326bfc72cb1ba9a520f728269fef0e4d84e3066c92d3a6`.

- All **213 input-file hashes** are unchanged, including the complete accepted preflight
  and SHAP inventories. Frozen target predictions reproduced; maximum probability differences
  were 0 for RF and `2.90e-8` for XGBoost, within the unchanged reload tolerances.
- A read-only audit regenerated all 258 neighborhoods exactly and independently solved
  augmented weighted normal equations for all 516 fits. Largest coefficient difference was
  `5.05e-14`. Named exports, all paired rows, all grouped summaries and fit metrics reproduced.
  The audit made no new model queries. The run used one native thread and no PyTorch import.
- Tests: **92 passed** (16 added for this milestone); lint, formatting and whitespace clean.
- Timing is **not accepted as a wall-clock benchmark**: the recorded performance-counter
  duration is 56.22 seconds, but UTC manifest timestamps span 1,490.10 seconds. The reason is
  unresolved. Preserve both values; do not select one as an efficiency result. Artifact writing
  and verification are also included. A dedicated timing protocol is still needed.
- The run occupies about 471 MiB locally. Generated artifacts remain ignored by Git: local
  source commits or a later push do not back up this evidence. No artifacts were overwritten.

## On-track check and next step

**On track for the pilot sequence, not finished with the expected thesis scope.** The seed-42
LIME pilot is now complete; Grid tracing/deletion and three-seed tree/TreeSHAP evidence were
already available. This run adds no CNN condition: still **3 of the expected 12** (one dataset,
Grid, three seeds), not a thesis-completion percentage. LIME at additional model-training seeds
and tree–CNN LIME comparisons remain deferred, not silently counted as complete.

The next priority is to resolve the **second dataset and authoritative SOM2IMG implementation**
with Yu/the committee. Obtain the exact author version, collision/rendering/allocation policy
and matched scaling/orientation/training settings before implementing an accepted encoder
comparison. Freeze the second-dataset choice before fitting or inspecting its test results.
Then validate mapping and train one matched SOM2IMG/Grid condition before matrix expansion.
No further test-driven LIME tuning is planned. Committee approval, defense dates and a separate
artifact-backup destination remain unconfirmed; no message or upload was sent on Yu's behalf.
