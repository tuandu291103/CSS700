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
samples and traced all 30 named features. These are functional results only; faithfulness,
multi-seed stability, and cross-method agreement remain TODO.

Integrated Gradients passed the declared completeness rule for all samples and both distinct
baselines after one zero-baseline retry. Median agreement with Grad-CAM was 0.396 Spearman for
zero/train mean and 0.332 for training median; median top-5 overlap was 0.40 for both. The two IG
baselines agreed more strongly with one another (median Spearman 0.790 and top-5 overlap 0.80),
but baseline dependence remained visible. Faithfulness and multi-seed replication remain TODO.

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
