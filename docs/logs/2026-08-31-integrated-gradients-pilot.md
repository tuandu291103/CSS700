# Experiment log: Integrated Gradients pilot on run 20260831-083111

## Question and pre-registered expectation

- **Research question:** Can predicted-class Integrated Gradients be traced to all 30 named
  features on the frozen seed-42 Grid CNN, and how sensitive are the rankings to the baseline?
- **Expected result:** Signed attributions should be finite and complete within the declared
  tolerance, preserve frozen targets and predictions, and support paired comparison with the
  existing Grad-CAM feature scores.
- **Failure criterion:** Non-finite values, target or prediction mismatch, incomplete feature
  mapping, unresolved completeness failures, or presentation of equivalent baselines as a real
  sensitivity comparison.

## Reproducibility

- **Run:** `artifacts/20260831-083111__breast-cancer__grid__small-cnn__seed-42__baseline`
- **Model commit:** `f100658e504662aa0054408a1d60ee56c7321b07`.
- **Explainer commit:** `b364ac65b688898806ec4d4609104190ecbb3628`; worktree clean at
  explanation start.
- **Config:** `configs/integrated_gradients.yaml`; SHA-256
  `dc09f7d69aed61b9b6c7b5ef8762e7b777036da3428ed6b70253bfa428772b95`.
- **Target:** Frozen predicted class for each of the same 86 test samples used by Grad-CAM.
- **Integration:** Gauss-Legendre, 128 steps initially and 256 steps for failed convergence.
- **Ranking:** Absolute magnitude, while retaining the signed raw attribution values.

## Baseline decision

The maximum absolute transformed training mean was `4.21e-15`, making it numerically equivalent
to zero under the declared `1e-7` tolerance. Zero and training mean were therefore recorded as
aliases, not separate experiments. The transformed training median was retained as the distinct
training-only sensitivity baseline; its maximum absolute difference from zero was 0.3860.

## Results

### Completeness and tracing

- Both canonical baselines produced signed maps with shape `[86, 1, 5, 6]` and 30 traced feature
  scores per sample.
- All maps were finite, targets matched the frozen predicted classes, and predictions matched the
  stored checkpoint evaluation.
- Zero/train-mean passed convergence for 85 of 86 samples at 128 steps. The remaining sample
  passed after the automatic 256-step retry.
- Training median passed convergence for all 86 samples at 128 steps.
- Median relative convergence delta was 0.000272 for zero/train mean and 0.000434 for training
  median. Maximum relative deltas after retry were 0.0322 and 0.0101, respectively.
- Single-run batched pilot timing was approximately 2.79 ms per sample for zero/train mean and
  2.60 ms for training median. These are not robust benchmark estimates.

### Descriptive feature rankings

For both baselines, the five largest all-sample mean absolute attributions were worst texture,
worst radius, worst perimeter, compactness error, and smoothness error. Signed values and
predicted-class-specific summaries remain available in the run bundle; the mixed-class average
is descriptive and is not global model importance.

### Agreement

For zero/train mean versus Grad-CAM, median per-sample Spearman correlation was **0.396**,
median top-5 overlap was **0.40**, and median top-10 overlap was **0.50**. For training median
versus Grad-CAM, the corresponding values were **0.332**, **0.40**, and **0.50**.

For zero/train mean, median Grad-CAM rank agreement was 0.419 among 78 correct predictions and
0.224 among eight errors. This subgroup difference is exploratory because the error group is
small. Agreement also varied by predicted class and baseline, reinforcing the need to preserve
sample and target identifiers rather than reporting one pooled number.

The two distinct Integrated Gradients baselines had median per-sample Spearman correlation
**0.790**, median top-5 overlap **0.80**, and median top-10 overlap **0.80**. Baseline selection
therefore changed some rankings even though overall sensitivity agreement was stronger than
IG-to-Grad-CAM agreement.

## Interpretation and limitations

This run establishes complete, reproducible feature tracing for Integrated Gradients and shows
measurable method and baseline dependence. Moderate agreement with Grad-CAM is not evidence
that either method is correct or incorrect. Neither ranking has yet passed the proposal's
feature-deletion faithfulness test, and no clinical interpretation is supported.

The result remains limited to one frozen model, one dataset split, and one training seed. The
timings are single-run pilot measurements, and error-subgroup comparisons contain only eight
samples.

## Follow-up

1. **Completed 2026-09-03:** Feature-wise Occlusion Sensitivity using zero replacement in
   standardized space; see `2026-09-03-occlusion-pilot.md`.
2. **Completed 2026-09-03:** Compare all three methods on the same samples with paired rank and
   top-k metrics; see `2026-09-03-occlusion-pilot.md`.
3. **Completed 2026-09-03:** Run top-, random-, and bottom-ranked feature deletion curves for
   faithfulness; see `2026-09-03-faithfulness-pilot.md`.
4. Repeat the complete Grid condition across at least three training seeds.
