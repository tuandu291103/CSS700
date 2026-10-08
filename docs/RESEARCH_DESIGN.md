# Research design

**2026-10-07 planning update:** the professor's meeting notes establish a minimum three-dataset
study. The [revised plan](PROJECT_PLAN_2026_2027_REVISED.md) proposes Dry Bean as dataset three,
IGTD as a third encoder, random-layout controls, mapping identifiability checks and explicit
prior-framework comparisons. The design below records the earlier pilot contract; a new
experiment protocol must be frozen before executing the expanded matrix. Inverse feature
lookup alone is not a new attribution algorithm, nor does it establish faithfulness.

## Research questions

1. Can image-space explanations for tabular-to-image CNNs be traced back to original named
   features in a faithful and stable way?
2. How do Grad-CAM, Integrated Gradients, and Occlusion Sensitivity differ in feature rankings,
   faithfulness, stability, and computational cost under controlled conditions?
3. How does the encoding layout - fixed Grid versus data-driven SOM2IMG - change predictive
   behavior and feature-level explanations?
4. When and why do traced CNN explanations agree or disagree with SHAP, LIME, Random Forest,
   and XGBoost comparisons?

## Primary unit of analysis

The primary object is a **local, class-targeted named-feature attribution vector** for one frozen
test sample, one trained model, one encoding, and one explanation method. Aggregate rankings are
derived from local attributions and must retain sample, class, correctness, seed, and dataset
identifiers.

Global tree importance and local explanation values are different objects. They will be reported
separately; local SHAP values for tree models may be used for like-for-like local comparisons.

For the first tabular predictive baseline, reuse the Breast Cancer split fingerprint from Grid
split seed 42 and train Random Forest and XGBoost with model seeds 7, 42, and 2026. Candidate
models are fit on the training partition only, selected by validation macro F1 with configuration
order breaking ties, and evaluated once on the untouched test partition without refitting on
validation data. Use the existing training-only standardized features to preserve exact feature
order and preprocessing parity; tree models do not otherwise require scaling.

Retain Random Forest mean-decrease-in-impurity importance and XGBoost gain importance as separate
global, model-native diagnostics. Report their within-family seed stability, but do not compare
their raw magnitudes across families or treat them as local, class-targeted explanations. Local
TreeSHAP/LIME comparisons require a separately frozen target and background protocol.

The provisional local TreeSHAP design is in `docs/SHAP_COMPARISON_PROTOCOL.md` (v1 frozen
2026-09-10, not committee-approved). It specifies probability-output, interventional explanations with a shared
training-only background, independent reconstruction checks and same-target rank comparisons.
It explicitly separates cross-model agreement from explainer-only effects and does not equate
tree probability attributions with existing CNN logit attributions. The implementation and
seed-42 attempt are documented in `docs/logs/2026-09-12-tree-shap-pilot.md`. A Random Forest
numerical-routing discrepancy prevented initial acceptance, even though the full-background aggregate
reconstruction residual initially fell within tolerance. A training-only routing guard now
stops that native condition before explanation. The subsequent v1.1 routing-equivalent copy
passed all four seed-42 conditions, including exhaustive synthetic contribution checks and
unchanged-model probability checks. See `docs/TREE_SHAP_NUMERICAL_REPAIR.md` and
`docs/logs/2026-09-12-walkthrough-and-tree-shap-repair.md`. These accepted settings were then
extended to saved seeds 7 and 2026 without reselection. All 12 conditions passed; the separate
seed-pair analysis and report are in `docs/TREE_SHAP_REPLICATION_PROTOCOL.md` and
`docs/logs/2026-09-14-tree-shap-multiseed-stability.md`.

The separate LIME v1 design is frozen in `docs/LIME_COMPARISON_PROTOCOL.md`: predicted-class
probability on the unchanged seed-42 tree models, training-quartile membership, all features,
fixed neighborhood budgets and three explainer seeds. Coefficients describe the local
surrogate's bin-membership effects, not SHAP contributions. Independent generated-neighborhood
fit diagnostics and anchor error accompany all technically valid outputs; no poor-fit filter
is applied. A 48-fit training-only preflight passed technical checks but showed weak local
approximation, documented in `docs/logs/2026-09-15-lime-protocol-preflight.md`. The subsequent
four-condition training-only diagnostic is specified in `docs/LIME_LOCALITY_DIAGNOSTIC_PROTOCOL.md`
and reported in `docs/logs/2026-09-15-lime-locality-diagnostic.md`. It controls weight mass and
uses common evaluation weights so improved anchor fit is not mistaken for general local fit.
No alternative was promoted. The original v1 test runner is now implemented and the full
516-fit pilot is complete, with named exports and 15 separate comparisons. See
`docs/LIME_TEST_EXECUTION.md` and `docs/logs/2026-09-17-lime-test-pilot.md`. Primary-seed
same-model median Spearman is 0.792 (RF) / 0.805 (XGBoost), alongside median anchor errors
0.107 / 0.217 and independent generated-evaluation R-squared 0.209 / 0.239. The empirical
coexistence of ranking agreement and weak approximation is not method superiority. No
post-test tuning is planned; additional LIME training seeds and tree–CNN comparisons remain
deferred. Second-dataset and authoritative SOM2IMG decisions take priority for full-scope work.

## Controlled experiment matrix

Expected scope contains:

- 2 datasets;
- 2 encodings (Grid, SOM2IMG);
- 3 training seeds;
- 3 CNN explanation methods applied to the same eligible checkpoints and samples;
- tabular comparison models fit on the same training splits.

This produces 12 primary CNN training conditions before explanation settings. The test split is
not used to choose architecture, preprocessing, layout, stopping point, attribution baseline, or
metric definition.

## Controls and leakage barriers

- Create stratified train/validation/test indices once per dataset and split seed.
- Fit scalers, imputers, feature selection, and SOM layouts on training data only.
- Reuse split indices across encodings and tabular baselines.
- Keep CNN architecture and training policy constant across encodings whenever dimensions allow.
- Explain the same frozen test sample IDs for every applicable method.
- Preserve mapping, padding mask, feature order, preprocessing object, checkpoint, config, seed,
  package versions, and commit SHA in each run.
- Do not inspect test explanation results when choosing hyperparameters.

## Feature-to-pixel mapping contract

Every encoder must export one record per original feature with at least:

```text
feature_index, feature_name, row, column, channel, mapping_version
```

The record must also describe collisions, empty pixels, repeated pixels, normalization, and any
aggregation from pixels to features. Grid Encoding begins as one feature per pixel in original
column order with an explicit padding mask. SOM2IMG must follow the same external contract even
if its internal layout algorithm differs.

## Explanation contract

- **Target:** use the same target class for all local methods. The initial default is the model's
  predicted class; true-class explanations for errors are a separately labeled analysis.
- **Grad-CAM:** pre-register the convolutional layer, resizing rule, sign handling, and mapping
  aggregation.
- **Integrated Gradients:** record baseline image, integration steps, internal batch size, and
  convergence delta; test zero and training-mean baselines on a pre-declared subset.
- For the standardized Grid pilot, zero and the transformed training mean are numerically
  equivalent. Record that alias rather than presenting it as a sensitivity comparison, and use
  the transformed training median as the distinct training-only sensitivity baseline.
- **Occlusion:** mask complete feature pixels/regions, not arbitrary sliding windows. For the
  standardized Grid pilot, replace one mapped feature pixel at a time with zero (the transformed
  training mean), measure the signed predicted-class logit drop as the primary score, retain the
  signed probability drop as a secondary score, and rank by absolute logit drop. Record both
  conceptual model evaluations and actual batched forward calls.
- Retain signed raw values when meaningful. Rank by a pre-registered transformation, such as
  absolute magnitude, rather than silently discarding direction.

## Outcome measures

### Predictive quality

Report accuracy, macro F1, per-class precision and recall, confusion matrices, and the
majority-class baseline. Do not interpret explanations from a CNN that fails the proposal's
minimum predictive-quality gate without clearly labeling the limitation.

### Faithfulness

For each ranking, mask top-ranked features cumulatively at pre-declared values of k and measure
target-confidence and performance decline. Compare against matched random and bottom-ranked
features. Report deletion curves and area-over-perturbation-curve style summaries with
uncertainty, not only a single k.

For the seed-42 Grid pilot, use `k = 0, 1, 2, 3, 5, 10, 15, 20, 25, 30`; share 100 fixed random
permutations across methods; and report standardized-zero replacement as primary with transformed
training median as sensitivity. The primary summary is normalized target-logit-drop AOPC. Use
1,000 hierarchical bootstrap resamples over samples and random-control repeats for 95% intervals.
Treat the observed top/random/bottom ordering as a result rather than a software quality gate,
and disclose that zero replacement is method-aligned with zero-based Occlusion.

### Agreement

- Spearman rank correlation on a consistent feature universe;
- top-k overlap for k = 5 and 10, using both intersection count and Jaccard similarity;
- pairwise method matrices by dataset, encoding, seed, and prediction group.

### Stability

Measure pairwise rank correlation and top-k overlap across training seeds, similar samples,
Integrated Gradients baselines, and any stochastic explainer settings. Separate model-training
instability from explainer instability where possible.

For the first Grid replication, hold the split, preprocessing, layout, hyperparameters, and test
samples fixed with data-split seed 42 while training models with pre-registered seeds 7, 42, and
2026. Compare predicted-class explanations primarily where the model pair predicts the same class;
label different-target comparisons as exploratory. Summarize predictive metrics, pairwise
prediction agreement, per-sample explanation rank/top-k stability, and deletion-faithfulness
contrasts across seeds. Three seeds support descriptive stability estimates, not a population-level
variance claim.

The completed TreeSHAP replication likewise fixes the split and background while comparing
each model-seed pair separately within family. Full-training and fixed-100-row backgrounds
remain separate analyses. Report conditional paired sample intervals and descriptive ranges
across seed pairs, not a pooled interval or an inflated independent sample count. Identical
class predictions do not imply identical probabilities or explanation rankings.

Use the original 128 integration steps and completeness rule for Integrated Gradients, but allow
a 1,024-step retry for any replicate sample that fails initially. This multi-seed retry ceiling is
stored separately from the historical seed-42 pilot config so the earlier config hash remains
reproducible. Integration effort may vary by sample, but every retained attribution must pass the
same absolute-or-relative completeness threshold.

### Computational cost

Record wall-clock time per sample, number of model forward/backward passes, batch size, device,
and peak memory when feasible. Include warm-up and repeated timings.

### Group analysis

Pre-register comparisons for correct versus incorrect predictions, predicted/true class, and
dataset. Treat small error groups cautiously and report counts.

## Statistical reporting

Report distributions and effect sizes, not only means. Use paired comparisons when methods
explain the same samples. Use bootstrap confidence intervals over samples, with seed-level
results visible rather than pooled away. Any multiplicity correction and exclusion rule must be
chosen before final hypothesis testing.

## Interpretation boundaries

No method is ground truth. Agreement is not automatically correctness, disagreement is not
failure, and predictive superiority of CNNs over tabular models is not required. Conclusions
must be limited to the tested datasets, encodings, models, targets, and perturbation choices.
