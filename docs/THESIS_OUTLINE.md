# Living thesis outline

Every completed experiment should update the relevant section below with evidence links,
interpretation, uncertainty, and limitations. Prose marked `TODO` is not a result.

## 1. Introduction

- Tabular-to-image motivation and interpretability gap.
- Why image-space saliency is insufficient for named-feature decisions.
- Research questions and contributions.

## 2. Background and related work

- Tabular explanations: LIME, SHAP, tree importance, and local/global distinctions.
- CNN explanations: Grad-CAM, Integrated Gradients, and Occlusion Sensitivity.
- Encodings: Grid baseline, SOM2IMG, DeepInsight, IGTD, and LM-IGTD.
- Prior evidence on explanation faithfulness and stability.

## 3. Methodology

- Datasets and data cards.
- Leakage-safe splits and preprocessing.
- Feature-to-pixel mapping contract.
- Grid Encoding and SOM2IMG.
- CNN architecture and training protocol.
- Tracing rules for each explanation method.
- Tabular comparison methods.
- Metrics, statistical analysis, compute, and reproducibility.

Mapping-readiness work is specified in `docs/SOM2IMG_READINESS.md`. Tracing validates complete
feature identities, coordinate bounds and one-to-one assignments, and returns columns in
original feature-index order. Shared-pixel layouts require an explicit rendering and
attribution-allocation policy before independent feature explanations can be reported.

The local tabular comparison is specified in `docs/SHAP_COMPARISON_PROTOCOL.md` (provisional
v1, frozen 2026-09-10): probability-output TreeSHAP, a common training-only background, explicit
binary-class handling, independent reconstruction checks and same-target rank comparisons.
The original v1 attempt failed because of RF numerical routing disagreements. A training-only
split-semantics guard now supplements reconstruction for the native implementation.
Cross-model/output-scale differences prevent an explainer-only interpretation of tree SHAP
versus the existing CNN explanations. Committee approval is not claimed.

The subsequent v1.1 implementation uses copied RF thresholds with a finite-float32 routing
equivalence certificate and the predictor's input conversion. Original models and source
background rows/weights are unchanged. The repair and accepted seed-42 pilot are documented
in `docs/TREE_SHAP_NUMERICAL_REPAIR.md` and the walkthrough/repair experiment log.

The provisional LIME protocol is in `docs/LIME_COMPARISON_PROTOCOL.md`. It explains the
same frozen tree probabilities with training-quartile membership coefficients, a fixed
weighted Ridge model and independently seeded fit/evaluation neighborhoods. Keep surrogate
approximation diagnostics separate from technical validity and explanation agreement.

## 4. Experiments

- Pilot validation and tracing unit tests.
- Frozen controlled experiment matrix.
- Ablations and sensitivity analyses.

Recorded evidence: the seed-42 Breast Cancer Grid CNN and predicted-class Grad-CAM pilot is
documented in `docs/logs/2026-08-31-grid-cnn-grad-cam-pilot.md`. The accepted run began from a
clean commit and retained split indices, preprocessing, mapping, checkpoint, predictions, raw
maps, traced feature scores, metrics, and figures.

Integrated Gradients evidence on the same checkpoint and samples is documented in
`docs/logs/2026-08-31-integrated-gradients-pilot.md`. Zero and transformed training mean were
recorded as equivalent, while transformed training median supplied the distinct baseline
sensitivity condition. Signed raw values, completeness deltas, retry counts, baseline-paired
scores, and Grad-CAM agreement are retained.

Feature-wise Occlusion Sensitivity and the first complete three-method comparison are documented
in `docs/logs/2026-09-03-occlusion-pilot.md`. The run masks one named Grid feature at a time with
standardized zero and retains signed predicted-class logit and probability drops. Exact masks,
raw scores, aggregate summaries, runtime accounting, pairwise sample-level metrics, and reviewed
figures are retained in the same frozen run bundle.

Cumulative deletion-faithfulness evidence is documented in
`docs/logs/2026-09-03-faithfulness-pilot.md`. Pre-registered top-, bottom-, and repeated-random
deletion uses both standardized-zero and training-median replacement. Per-sample curves, model-
performance metrics, normalized AOPC summaries, hierarchical bootstrap intervals, paired control
contrasts, and reviewed figures are retained.

Fixed-split model-training-seed evidence is documented in
`docs/logs/2026-09-06-grid-multiseed-stability.md`. The split and 86 test samples remain fixed at
seed 42 while model seeds 7, 42, and 2026 vary. Predictive metrics, pairwise prediction agreement,
same-target explanation rank/top-k stability, and deletion-faithfulness contrasts are retained in
a standalone analysis artifact with source-run provenance and cross-run quality checks.

Random Forest and XGBoost predictive baseline evidence is documented in
`docs/logs/2026-09-06-tabular-baselines.md`. Both model families reuse the exact Grid split and
training-only preprocessing across seeds 7, 42, and 2026. Complete validation searches, selected
models, untouched-test predictions, predictive metrics, seed agreement, and model-native global
importance are retained. Global tree importance remains separate from local explanation results.

The train-only SOM mapping-readiness diagnostic is documented in
`docs/logs/2026-09-06-som-mapping-readiness.md`. With one fixed, explicitly provisional MiniSom
configuration, 30 features occupied 22 pixels and 13 features shared five pixels. This is
implementation-risk evidence, not an authoritative SOM2IMG reproduction or a CNN result.
No collision resolution, encoder comparison or new predictive claim was made.

The first local TreeSHAP attempt and numerical-routing diagnosis are documented in
`docs/logs/2026-09-12-tree-shap-pilot.md`. Both clean-commit attempts retain failure evidence.
The initial RF 100-row background failed training-smoke reconstruction; the strengthened
guard rejects even its full background before explanation. XGBoost passed reconstruction
for both backgrounds in the initial attempt. This is validation evidence, not a completed
cross-model agreement experiment; no RF ranking or seed-stability claim is promoted.

The repaired seed-42 v1.1 pilot passed all four model/background conditions; see
`docs/logs/2026-09-12-walkthrough-and-tree-shap-repair.md`. Original failed bundles remain
diagnostic. New signed attributions, background values, copied thresholds, routing certificates,
source hashes and same-target comparison tables belong to a separate accepted bundle.

The accepted v1.1 extension to seeds 7 and 2026 and separate seed-stability analysis are
documented in `docs/logs/2026-09-14-tree-shap-multiseed-stability.md` (runs dated September 12).
All 12 model/seed/background conditions passed. The analysis retains 1,032 sample-pair rows
and 72 grouped summaries without treating repeated samples or seed pairs as independent.

LIME training-preflight evidence is in `docs/logs/2026-09-15-lime-protocol-preflight.md`:
eight training anchors x two frozen tree models x three explainer seeds produced 48 fits,
with six same-seed repeats. All technical checks passed; no test LIME explanations or
paired LIME–SHAP comparisons were generated. Weak surrogate approximation motivates a
separate training-only locality diagnostic before expanding to the full test runner.

That bounded diagnostic is complete; see `docs/logs/2026-09-15-lime-locality-diagnostic.md`.
Four fixed surrogate conditions on the saved neighborhoods produced 192 fits and 144 paired
contrasts, with common evaluation weights and a weight-mass control. Original preflight
references reproduced and no new model inference was needed. No alternative was selected.

The original-v1 test pilot is now complete; see `docs/logs/2026-09-17-lime-test-pilot.md`
(run September 16). Two seed-42 tree models x 86 cases x three explainer seeds produced
516 verified fits, 15,480 named-feature rows, 1,290 paired rows, 90 agreement summaries
and 30 fit-quality summaries. Accepted sources and all saved calculations were audited;
no model retraining, fit-quality filter or test-driven setting changes were made.

Second-dataset readiness is documented in `docs/SECOND_DATASET_DECISION.md` and
`docs/logs/2026-09-17-scope-readiness.md`. Six synthetic larger-grid/CNN tests extend
compatibility coverage without adding a trained result. The subsequent authorized Spambase
integrity audit is complete: see `docs/SPAMBASE_DATA_CARD.md` and
`docs/logs/2026-09-19-spambase-readiness.md`. All 4,601 rows/57 features pass structural
checks, but 183 repeated-feature groups (577 rows), including three conflicting groups
(six rows), motivate prospective group-aware splitting. No rows/labels were changed and
no partition, preprocessing fit or model was created. Explicitly disclose full-release
pre-split integrity inspection, numerical-equality definitions and lack of message/source
group IDs. Inclusion, split/conflict/evaluation policy and authoritative SOM2IMG collision
and matched-comparison choices remain open; this audit is not a predictive or novelty result.

## 5. Results

- Predictive performance.
- Feature-level faithfulness.
- Method agreement and top-k overlap.
- Stability across seeds, samples, and baselines.
- Encoding effects.
- Group analysis and computational cost.

Pilot result to develop in the draft: the validation-selected Grid CNN achieved 0.9070 test
accuracy and 0.9017 macro F1, compared with 0.6279 accuracy and 0.3857 macro F1 for the
majority-class baseline. Predicted-class Grad-CAM produced finite 5 x 6 maps for all 86 test
samples and traced all 30 named features. The subsequent results below evaluate cross-method
agreement, conditional deletion faithfulness, and model-seed stability.

Integrated Gradients passed the declared completeness rule for all samples and both distinct
baselines after one zero-baseline retry. Median agreement with Grad-CAM was 0.396 Spearman for
zero/train mean and 0.332 for training median; median top-5 overlap was 0.40 for both. The two IG
baselines agreed more strongly with one another (median Spearman 0.790 and top-5 overlap 0.80),
but baseline dependence remained visible.

Occlusion produced finite named-feature scores for all 86 samples with masks that covered exactly
the 30 feature pixels. Median rank agreement was 0.969 between zero-baseline Integrated Gradients
and zero-replacement Occlusion, compared with 0.367 between Grad-CAM and Occlusion. Their median
top-5 overlaps were 0.80 and 0.40, respectively. The shared zero reference may contribute to the
strong input-level-method agreement. That comparison alone does not establish correctness;
cumulative deletion below supplies conditional faithfulness evidence.

Under standardized-zero deletion, mean target-logit-drop AOPC was 1.049 for Grad-CAM, 1.122 for
zero-baseline Integrated Gradients, and 1.115 for zero-replacement Occlusion, compared with 0.759
for the shared random control. Each top-minus-random 95% bootstrap interval remained above zero;
the ordering persisted under training-median replacement. At k = 10, top deletion reduced
accuracy from 0.907 to 0.686, 0.628, and 0.651 respectively, while bottom-ten deletion caused no
prediction flips. This was conditional single-model evidence; the fixed-split replication below
tests training-seed sensitivity, while deletion realism remains a limitation.

Across fixed-split model seeds 7, 42, and 2026, mean test accuracy was 0.9186 with sample standard
deviation 0.0116, and mean macro F1 was 0.9141 with sample standard deviation 0.0123. All three
models agreed on 79 of 86 test predictions. For model pairs predicting the same target, median
per-sample rank stability was 0.718 Spearman for Grad-CAM, 0.746 for zero-baseline Integrated
Gradients, and 0.712 for zero-replacement Occlusion; every method had median top-5 overlap 0.60
and top-10 overlap 0.70. All three methods retained positive top-minus-random deletion intervals
at every seed. Predictive behavior was therefore comparatively stable, but explanation rankings
remained meaningfully sensitive to model training. These three-seed estimates are descriptive,
not population-level variance estimates.

On the same fixed splits and model seeds, Random Forest achieved mean test accuracy 0.9496 and
macro F1 0.9462, while XGBoost achieved 0.9651 and 0.9624. Both exceeded the Grid CNN means of
0.9186 accuracy and 0.9141 macro F1 in this pilot. Random Forest prediction agreement across
seeds was 0.988-1.000; XGBoost produced identical class predictions across all three seeds.
Model-native global rankings were also stable within family: pairwise Spearman was 0.974-0.978
for Random Forest and 0.715-0.871 for XGBoost, with identical top-five feature sets within each
family. These global rankings are not like-for-like local explanation comparators.

First accepted TreeSHAP result: full-training versus fixed-100-row-background median rank
agreement was 0.9913 for RF and 0.9915 for XGBoost (86 samples each). RF–XGBoost agreement
was moderate: mean Spearman 0.620, sample-bootstrap 95% interval [0.597, 0.639], median 0.648
and median top-five overlap 0.60 among 84 same-target samples. Tree–CNN mean agreement
ranged from 0.138 to 0.380 across the six comparisons, each with 79 eligible samples.
This is one-seed, descriptive evidence on differing eligible sets, not a formal contrast or
an isolated model/method effect. One background subset does not establish general robustness.

The subsequent three-seed TreeSHAP replication had full-background seed-pair median
Spearman 0.950–0.960 for RF and 0.956–0.969 for XGBoost. Median top-five overlap was 1.00
for RF and 0.80–1.00 for XGBoost; all pairs had median top-ten overlap 0.90. Eligible counts
were 85–86 for RF and 86 for XGBoost. These are descriptive ranges across separate pairs;
pair-conditional sample intervals and the 100-row sensitivity are in the replication report.
There were no constant-vector exclusions or exact cutoff ties. RF had three-seed class
consensus for 85/86 samples, XGBoost for 86/86, but probabilities and local rankings varied.

Within each seed, full-background RF–XGBoost mean Spearman ranged from 0.598 to 0.620;
tree–CNN means ranged from 0.138 to 0.380 across 18 comparisons with 79–82 eligible samples.
Lower cross-model/method agreement persisted descriptively, but the seed-42 ordering of
CNN explainers was not universal (XGB seed-2026 mean agreement was 0.1416 for IG-zero and
0.1444 for Grad-CAM). No method-superiority or formal cross-comparison claim follows.

LIME compatibility observation (training only, not a test result): median anchor probability
error across 24 repeated fits per model was 0.081 for RF and 0.302 for XGBoost. Median weighted
R-squared on independently generated neighbors was 0.219 and 0.252. Anchor surrogate outputs
fell outside [0, 1] in 3/24 RF and 18/24 XGB fits; these unconstrained predictions were
retained, not clipped. Eight anchors do not establish general LIME performance. Passing
implementation checks alone does not justify treating the coefficients as faithful rankings.

In the training-only locality diagnostic, narrower raw weighting reduced XGBoost median
anchor error from 0.3017 to 0.0420, but increased common-evaluation RMSE in every one of the
24 paired fits (median paired change +0.00513). Median anchor self-influence increased from
0.0335 to 0.7810. The continuous same-neighbor diagnostic lowered neighborhood RMSE for both
families, but improved both neighborhood and anchor errors in only 4/24 RF and 14/24 XGBoost
pairs. It is not native continuous LIME; representation and penalty geometry differ. These
eight-anchor descriptions do not select a replacement protocol or establish a general method
ranking. Approximation diagnostics accompany the completed v1 test comparison.

LIME test-pilot result: primary seed 11 has same-model LIME–TreeSHAP mean Spearman 0.7601
[95% paired-sample interval 0.7284, 0.7913] for RF and 0.7877 [0.7652, 0.8123] for XGBoost;
medians are 0.7918 / 0.8049. All six family/seed comparisons have median top-5/top-10 overlap
0.8. Yet primary median anchor errors are 0.1073 / 0.2170 and generated-evaluation R-squared
0.2088 / 0.2393; 19/86 RF and 59/86 XGBoost anchor predictions are outside [0, 1]. Preserve
these poor-fit cases. Sampling-seed-pair median Spearman ranges 0.8378–0.8610 (RF) and
0.8914–0.8950 (XGBoost), with the model held fixed. Substantial agreement and repeatability
can coexist with limited surrogate approximation; this is pilot evidence, not novelty or
ground-truth validation. Full subgroup/seed summaries and interpretation limits are in the log.
The recorded runtime and UTC timestamp span disagree, so this run is not efficiency evidence.

## 6. Discussion

- When traced explanations appear reliable.
- Meaningful disagreement and method-specific failure modes.
- Implications for artificial-image explanations.
- Practical guidance for users and researchers.

## 7. Ethics, limitations, and threats to validity

- Artificial visual persuasiveness and over-trust.
- Dataset/model/generalization limits.
- Baseline, perturbation, rank-normalization, and mapping dependence.
- No clinical or real-world decision claims.

Implementation fidelity is an additional threat: the seed-42 RF case showed that small
aggregate additivity residuals can coexist with inconsistent near-threshold routing.
Numerical checks must be tied to the actual predictor and background semantics, not just
plausible-looking attribution sums. This local engineering observation is not a claim of
novelty or a general failure of TreeSHAP. The v1.1 routing-equivalent repair passed the
technical gates for all three registered seeds and both backgrounds. Generalization beyond
this dataset, fixed split and validated numerical setting remains untested.

## 8. Conclusion and future work

- Direct answers to each research question.
- Expected contribution and next studies.

## Appendices

- Full configs and hyperparameters.
- Dataset cards and split identifiers.
- Mapping specifications.
- Additional result tables and reproducibility instructions.
