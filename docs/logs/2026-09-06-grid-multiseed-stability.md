# Experiment log: fixed-split Grid model-seed stability

## Question and pre-registered expectation

- **Research question:** When the dataset split, preprocessing, Grid layout, model architecture,
  hyperparameters, and test samples are fixed, how much do predictive behavior, feature rankings,
  and deletion faithfulness change across model-training seeds?
- **Registered seeds:** Data-split seed 42 and model-training seeds 7, 42, and 2026.
- **Primary explanation comparison:** Predicted-class explanations from a model pair are compared
  where both models predict the same target class. Different-target comparisons are retained as
  exploratory because changing the explained target changes the explanation question.
- **Expected pattern:** Predictive metrics and predictions should remain similar; explanations
  should have positive rank correlation and non-trivial top-k overlap; top-ranked deletion should
  remain more damaging than random deletion at every seed.
- **Technical failure criterion:** Any registered seed missing or duplicated; non-identical split,
  samples, labels, preprocessing, feature map, or controlled model configuration; failed source
  quality gate; non-finite feature scores; undefined primary Spearman values; or incomplete
  faithfulness rows. Scientific effect sizes were not software quality gates.

## Reproducibility

- **Analysis artifact:**
  `artifacts/20260906-203847__breast-cancer__grid__fixed-split__model-seeds-7-42-2026__stability`.
- **Analysis and replica commit:** `1b3714de38ae0d4aad7f0402ac7ba3688dc6319e`; the worktree
  was clean when both new models and the stability analysis began.
- **Stability config:** `configs/grid_stability.yaml`; SHA-256
  `9a1d72a29a94e625ef1b0f9c5474156dca65c13395a5e562755596da3426431d`.
- **Seed-7 training config:** SHA-256
  `74ca9a7622399cab6ef88bb603faebdf4bc00e75fd681f818d2ccb77d6b22cd9`.
- **Seed-42 historical training config:** SHA-256
  `5b781c9fe43a77d22f9ae44fa02fb557e2830e0b08c14649bfe4be49ceb9e26f`.
- **Seed-2026 training config:** SHA-256
  `0f0dc712df94420df09bf23cd6a9f193ed7102a4ba7b4aaa81c37ab00bda633a`.
- **Multi-seed Integrated Gradients config:** SHA-256
  `828606d39846037ab42ffe02c17fc8f0234ac36084ac333a9e633658340a3b14`.
- **Faithfulness config:** SHA-256
  `4a0ab11dfa739cc1b132993f4992c7779665383275723e81ee709264aeb5c678`.

Input model runs were:

| Model seed | Run | Source commit |
|---:|---|---|
| 7 | `20260906-203807__breast-cancer__grid__small-cnn__split-seed-42__model-seed-7__replicate` | `1b3714d` |
| 42 | `20260831-083111__breast-cancer__grid__small-cnn__seed-42__baseline` | `f100658` |
| 2026 | `20260906-203807__breast-cancer__grid__small-cnn__split-seed-42__model-seed-2026__replicate` | `1b3714d` |

The historical seed-42 IG run used the same 128 initial integration steps and the same
absolute-or-relative completeness threshold, but a 256-step retry ceiling. The two new replicas
pre-registered a 1,024-step retry ceiling after development testing exposed one seed-2026
training-median sample that did not converge at 256 steps. All accepted samples and baselines
passed the shared completeness threshold. This numerical-effort difference is retained as a
minor comparability limitation rather than hidden by modifying the historical artifact.

Primary evidence is retained in `metrics/predictive_stability.csv`,
`metrics/prediction_pair_agreement.csv`, `metrics/explanation_stability.csv`,
`metrics/explanation_stability_summary.csv`, `metrics/faithfulness_stability.csv`, and
`metrics/stability_summary.json`. The reviewed overview is
`figures/grid_seed_stability.png`.

## Results

### Quality checks

All source-run and cross-run gates passed. Each registered model seed was present exactly once;
split indices, sample IDs, labels, feature maps, training-only preprocessing, and controlled
configuration sections were identical. All source training and explanation quality checks passed,
all feature scores were finite, every primary Spearman value was defined, and every expected
faithfulness row was present.

Both new checkpoints reloaded with zero maximum absolute logit difference. Grad-CAM, Integrated
Gradients, and Occlusion traced all 30 named features for all 86 test samples. IG passed its
completeness rule for both baselines on all samples; the training-median baseline required one
retry for seed 7 and one for seed 2026.

### Predictive stability

| Model seed | Best epoch | Validation macro F1 | Test accuracy | Test macro F1 |
|---:|---:|---:|---:|---:|
| 7 | 19 | 0.9624 | 0.9302 | 0.9262 |
| 42 | 19 | 0.9624 | 0.9070 | 0.9017 |
| 2026 | 38 | 1.0000 | 0.9186 | 0.9144 |

Mean test accuracy was 0.9186 with sample standard deviation 0.0116 and range 0.9070-0.9302.
Mean test macro F1 was 0.9141 with sample standard deviation 0.0123 and range 0.9017-0.9262.
Pairwise prediction agreement was 0.9535 for seeds 7 versus 42 and 0.9419 for both other seed
pairs. All three models agreed on 79 of 86 samples (0.9186).

### Explanation stability

The primary same-predicted-target subset contained 244 pair-sample comparisons across the three
model pairs.

| Method | Median per-sample Spearman | Median top-5 overlap | Median top-10 overlap |
|---|---:|---:|---:|
| Grad-CAM | 0.7177 | 0.60 | 0.70 |
| Integrated Gradients, zero baseline | 0.7459 | 0.60 | 0.70 |
| Occlusion, zero replacement | 0.7119 | 0.60 | 0.70 |

All methods showed positive, moderate-to-strong median rank stability, but a typical pair shared
only three of its top five and seven of its top ten features. Integrated Gradients had the highest
median rank stability in this three-seed comparison, but the differences were not subjected to a
paired inferential test and should not be read as a proven method ranking.

There were only 14 different-target pair-sample comparisons. Grad-CAM's median Spearman was
-0.6386 and its median top-5 overlap was 0.00 in this exploratory subset, whereas IG and Occlusion
remained positive. This contrast reinforces the decision not to mix different explained targets
into the primary stability estimate; the small subset does not support a general method claim.

### Deletion-faithfulness stability

The table summarizes zero-replacement target-logit-drop AOPC across model seeds. Positive
top-minus-random values mean that deleting a method's highest-ranked features damaged its frozen
target more than matched random deletion.

| Method | Mean top AOPC [seed range] | Mean top - random (sample SD) | Seeds with positive 95% interval |
|---|---:|---:|---:|
| Grad-CAM | 1.005 [0.799, 1.167] | 0.260 (0.027) | 3/3 |
| Integrated Gradients, zero baseline | 1.116 [0.798, 1.427] | 0.371 (0.137) | 3/3 |
| Occlusion, zero replacement | 1.114 [0.818, 1.410] | 0.370 (0.119) | 3/3 |

All three methods retained positive top-minus-random bootstrap intervals at every model seed.
Thus the seed-42 deletion-faithfulness result replicated directionally. Absolute AOPC and the
contrast magnitude varied more for IG and Occlusion than for Grad-CAM, so faithfulness strength
should be reported by seed and summarized across seeds rather than represented by one model.

## Interpretation and limitations

The Grid CNN's predictive behavior was comparatively stable under the three registered training
seeds: metric standard deviations were about 0.012, pairwise predictions agreed on at least 94.2%
of test samples, and all three models agreed on 91.9%. The explanations were less stable than the
predictions. Median rank correlations near 0.71-0.75 are encouraging, and deletion evidence
replicated at every seed, but top-feature membership still changed enough to matter when an
explanation is presented as a precise feature ranking.

This experiment isolates model-training variability only for one fixed split, one small dataset,
one Grid layout, and one CNN architecture. Three seeds provide descriptive sensitivity evidence,
not a population-level variance estimate. The same test set is reused across seeds, the dataset
is too small for clinical inference, and predicted-class comparisons condition on model outputs.
Deletion can create unrealistic feature combinations, standardized zero favors methods using the
same reference, absolute-magnitude ranking removes direction, and bootstrap intervals within a
seed do not quantify training-seed uncertainty. No result supports clinical or real-world use.

## Follow-up

1. Add Random Forest and XGBoost predictive baselines on the same fixed splits, keeping global
   model importance separate from local sample explanations.
2. Add SHAP and LIME only after their target, background/reference data, and computational budget
   are frozen in the protocol.
3. Resolve the second-dataset criteria and authoritative SOM2IMG implementation with the
   committee before expanding the controlled matrix.
