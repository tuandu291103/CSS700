# Experiment log: cumulative feature-deletion faithfulness on run 20260831-083111

## Question and pre-registered expectation

- **Research question:** Do the highest-ranked features from Grad-CAM, zero-baseline Integrated
  Gradients, and zero-replacement Occlusion affect the frozen Grid CNN more than matched random
  or bottom-ranked features?
- **Expected pattern:** Cumulative top-ranked deletion should produce larger predicted-class score
  declines than random deletion, while bottom-ranked deletion should produce the smallest decline.
- **Technical failure criterion:** Incorrect cumulative masks, incomplete rankings, unreproducible
  random controls, mismatched targets or samples, non-finite metrics, inconsistent full-deletion
  endpoints, or failure to reproduce the frozen evaluation at zero deletion. The scientific
  ordering itself was not a software quality gate.

## Reproducibility

- **Run:** `artifacts/20260831-083111__breast-cancer__grid__small-cnn__seed-42__baseline`.
- **Model commit:** `f100658e504662aa0054408a1d60ee56c7321b07`.
- **Evaluation commit:** `1329fc4bc5cce1782935ed386587bd32b1e92ea5`; worktree clean at
  evaluation start.
- **Config:** `configs/faithfulness.yaml`; SHA-256
  `4a0ab11dfa739cc1b132993f4992c7779665383275723e81ee709264aeb5c678`.
- **Samples and target:** The same 86 frozen test samples and their original predicted classes.
- **Deletion steps:** `k = 0, 1, 2, 3, 5, 10, 15, 20, 25, 30`.
- **Controls:** Bottom-ranked deletion and 100 fixed random permutations shared across methods.
- **Replacement:** Standardized zero as primary and transformed training median as sensitivity.
- **Primary summary:** Target-logit-drop curve integrated over k and normalized by 30 features.
- **Uncertainty:** 1,000 fixed-seed hierarchical bootstrap resamples over samples and, for the
  random control, permutation repeats. Intervals are estimation intervals conditional on this
  frozen model, not model-training uncertainty.

Primary evidence is retained in `metrics/faithfulness_curves.csv`,
`metrics/faithfulness_aopc.csv`, `metrics/faithfulness_contrasts.csv`, and
`metrics/faithfulness_summary.json`. Per-sample summaries, raw arrays, exact rankings, and
replacement images are also retained under `metrics/`. Reviewed figures are
`figures/faithfulness_deletion_curves.png` and `figures/faithfulness_aopc.png`.

## Results

### Quality checks

All technical gates passed. Every ranking was a complete 30-feature permutation; random controls
were reproducible from their seed; cumulative masks contained exactly k named features; k = 0
reproduced the frozen evaluation; and deleting all 30 features produced the same endpoint
regardless of ranking order. All raw and reported values were finite.

The complete evaluation represented 165,292 conceptual model evaluations and 428 batched forward
calls. Warm-cache execution took approximately 0.428 seconds in this environment. This is an
implementation-specific pilot timing rather than a stable benchmark.

### Primary zero-replacement result

| Ranking | Top AOPC, mean [95% interval] | Random AOPC | Bottom AOPC, mean [95% interval] | Top - random [95% interval] |
|---|---:|---:|---:|---:|
| Grad-CAM | 1.049 [0.939, 1.154] | 0.759 | 0.469 [0.365, 0.576] | 0.290 [0.253, 0.324] |
| Integrated Gradients zero | 1.122 [0.983, 1.283] | 0.759 | 0.403 [0.335, 0.480] | 0.363 [0.301, 0.421] |
| Occlusion zero | 1.115 [0.971, 1.265] | 0.759 | 0.396 [0.332, 0.463] | 0.356 [0.300, 0.415] |

All three top-minus-random target-logit AOPC intervals were above zero. The corresponding
top-minus-bottom differences were 0.580 for Grad-CAM, 0.719 for Integrated Gradients, and 0.719
for Occlusion, again with intervals wholly above zero. Predicted-class probability-drop AOPC
contrasts showed the same ordering.

At k = 10, zero-replacement top deletion reduced accuracy from the frozen 0.907 to 0.686 for
Grad-CAM, 0.628 for Integrated Gradients, and 0.651 for Occlusion. Mean random-deletion accuracy
was 0.861. Deleting each method's bottom ten features caused no prediction flips and left accuracy
and macro F1 at their frozen values of 0.907 and 0.902.

### Training-median sensitivity

The top/random/bottom ordering persisted under training-median replacement. Top AOPC values were
0.931 for Grad-CAM, 1.023 for Integrated Gradients, and 1.009 for Occlusion, compared with random
AOPC of 0.670. Their top-minus-random differences were respectively 0.261 [0.223, 0.302], 0.353
[0.297, 0.413], and 0.338 [0.283, 0.400].

The similar ordering across both replacements makes the pilot conclusion less dependent on the
exact replacement value. The current output does not provide a paired inferential comparison
between the three methods, so their close point estimates should not be interpreted as a proven
performance ranking.

## Interpretation and limitations

Conditional on this one frozen seed-42 Grid CNN, all three explanation rankings show useful
deletion faithfulness: removing their highest-ranked features damages the selected target more
than removing random or bottom-ranked features. The bottom-ranked results are especially
consistent with the rankings because the bottom ten features caused no prediction changes under
the primary replacement.

This evidence remains limited. Feature deletion can create unrealistic combinations outside the
training distribution, and neither zero nor marginal training median is a conditional imputation.
Zero replacement is also method-aligned with zero-based Occlusion and Integrated Gradients;
training-median sensitivity reduces but does not eliminate that concern. Absolute-magnitude
ranking discards attribution direction. Bootstrap intervals quantify test-sample and random-
control variability conditional on one fitted model, not training-seed variability. These results
do not support clinical interpretation.

## Follow-up

1. Repeat the full Grid training, explanations, agreement, and faithfulness pipeline across at
   least two additional pre-registered seeds.
2. Add Random Forest and XGBoost predictive baselines and clearly scoped importance outputs.
3. Resolve the second-dataset and authoritative SOM2IMG implementation decisions before the
   controlled two-encoding matrix.
