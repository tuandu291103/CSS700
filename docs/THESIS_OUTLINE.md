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

## 5. Results

- Predictive performance.
- Feature-level faithfulness.
- Method agreement and top-k overlap.
- Stability across seeds, samples, and baselines.
- Encoding effects.
- Group analysis and computational cost.

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
