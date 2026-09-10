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

The next local tabular comparison is proposed in `docs/SHAP_COMPARISON_PROTOCOL.md` (draft
v0.1, 2026-09-09): probability-output TreeSHAP, a common training-only background, explicit
binary-class handling, independent reconstruction checks and same-target rank comparisons.
This is planned methodology, not evidence. Cross-model/output-scale differences prevent an
explainer-only interpretation of tree SHAP versus the existing CNN explanations.

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

## 8. Conclusion and future work

- Direct answers to each research question.
- Expected contribution and next studies.

## Appendices

- Full configs and hyperparameters.
- Dataset cards and split identifiers.
- Mapping specifications.
- Additional result tables and reproducibility instructions.
