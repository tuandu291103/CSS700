# Experiment log: feature-wise Occlusion Sensitivity on run 20260831-083111

## Question and pre-registered expectation

- **Research question:** Can feature-wise Occlusion Sensitivity produce complete, signed scores
  for all 30 named features on the frozen seed-42 Grid CNN, and how do its rankings compare with
  Grad-CAM and Integrated Gradients on the same 86 test samples?
- **Expected result:** Each perturbation should replace exactly one mapped feature pixel with
  standardized zero, preserve the frozen predicted-class target, and produce finite target-logit
  and probability drops. All methods must use the same sample and feature universe.
- **Failure criterion:** Incorrect or overlapping masks, nonzero replacement values, non-finite
  scores, target or prediction mismatch, incomplete feature coverage, or incomparable method
  rows.

## Reproducibility

- **Run:** `artifacts/20260831-083111__breast-cancer__grid__small-cnn__seed-42__baseline`.
- **Model commit:** `f100658e504662aa0054408a1d60ee56c7321b07`.
- **Explainer commit:** `09e5d620bf99c51a903df759ad1810528848bb22`; worktree clean at
  explanation start.
- **Config:** `configs/occlusion.yaml`; SHA-256
  `4e38ac302be56df88be5bd2ac2cd951ea8b6e1d5b01419a90867bdea2327c0cc`.
- **Target:** Frozen predicted class for the same samples used by Grad-CAM and Integrated
  Gradients.
- **Perturbation:** Replace one complete mapped Grid feature pixel at a time with zero in
  standardized space, which is also the transformed training mean.
- **Primary score:** Original target logit minus occluded target logit. Signed values were
  retained; absolute logit drop supplied the registered ranking.
- **Secondary score:** Original target probability minus occluded target probability.

Primary evidence within the run bundle is retained in
`attributions/occlusion_features.csv`, `metrics/occlusion_aggregate_features.csv`,
`metrics/three_method_agreement.csv`, and `metrics/occlusion_summary.json`. The reviewed
diagnostic figures are `figures/occlusion_examples.png` and
`figures/three_method_agreement.png`; raw arrays and exact masks remain under `attributions/`.

## Results

### Perturbation and tracing checks

- All 30 masks contained exactly one pixel, did not overlap, and collectively covered exactly
  the 30 named feature locations in the 5 x 6 Grid image.
- All 86 frozen test samples produced finite scores with shape `[86, 30]`.
- Targets and original predictions matched the frozen evaluation for every sample.
- The runner performed 2,580 feature perturbations plus 86 conceptual original evaluations.
  Batching reduced these to seven measured forward calls after warm-up.
- Warm-cache batched runtime was approximately 0.00818 seconds in this pilot environment. This
  single measurement is implementation-specific and is not a robust benchmark.

Across all 2,580 sample-feature perturbations, the median absolute target-logit drop was 0.0555.
About 74.2% of signed drops were positive and 25.8% were negative. A negative value means that
zero replacement increased the selected target logit; it must not be silently relabeled as
positive support merely because absolute magnitude is used for ranking.

### Descriptive feature rankings

The five largest all-sample mean absolute target-logit drops were worst texture, worst radius,
worst perimeter, area error, and compactness error. The class-specific tables differ, so the
mixed predicted-class average is descriptive and is not global feature importance.

### Three-method agreement

The primary zero-reference comparisons were:

| Method pair | Median Spearman | Median top-5 overlap | Median top-10 overlap |
|---|---:|---:|---:|
| Grad-CAM vs Integrated Gradients zero | 0.396 | 0.40 | 0.50 |
| Grad-CAM vs Occlusion zero | 0.367 | 0.40 | 0.50 |
| Integrated Gradients zero vs Occlusion zero | 0.969 | 0.80 | 0.90 |

Occlusion also agreed strongly with training-median Integrated Gradients: median Spearman was
0.768, with median top-5 and top-10 overlaps of 0.80. The corresponding Integrated Gradients
baseline-to-baseline Spearman value was 0.790.

For Grad-CAM versus Occlusion, median Spearman agreement was 0.384 among 78 correct predictions
and 0.214 among eight errors. Integrated Gradients zero versus Occlusion remained high in both
groups (0.970 and 0.960). Error-group comparisons remain exploratory because only eight samples
were misclassified.

## Interpretation and limitations

This run completes reproducible named-feature tracing for all three proposed CNN explanation
methods on one frozen Grid model. Integrated Gradients and Occlusion produced highly similar
rankings when both used the standardized-zero reference, while Grad-CAM remained more distinct.
One plausible explanation is that the two input-level methods share the same reference and are
probing related local behavior, whereas Grad-CAM summarizes positive evidence at a convolutional
layer. That explanation is an inference to test, not an established causal conclusion.

Agreement does not establish correctness. Single-feature Occlusion Sensitivity is itself a
perturbation score, but it does not substitute for the registered cumulative deletion test with
random and bottom-ranked controls. No clinical conclusion is supported. The evidence remains
limited to one dataset split, one model seed, one encoding, and one replacement value.

## Follow-up

1. Implement cumulative top-, random-, and bottom-ranked feature deletion curves for all three
   rankings, with uncertainty summaries and area-over-perturbation-curve metrics.
2. Repeat the Grid model and explanation pipeline across at least three training seeds.
3. Add tabular Random Forest and XGBoost baselines before beginning the broader SOM2IMG matrix.
