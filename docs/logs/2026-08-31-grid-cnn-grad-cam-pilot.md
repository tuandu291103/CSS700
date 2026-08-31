# Experiment log: 20260831-083111 Grid CNN and Grad-CAM pilot

## Question and pre-registered expectation

- **Research question:** Can the leakage-safe Breast Cancer Wisconsin Grid representation train
  a CNN that beats the majority baseline, and can predicted-class Grad-CAM maps from its frozen
  checkpoint be traced to all 30 named input features?
- **Expected result:** The CNN should beat the majority-class baseline; every 5 x 6 Grad-CAM map
  should preserve the frozen prediction target and trace one-to-one to the 30 feature names.
- **Failure criterion:** Baseline-level prediction quality, test-informed checkpoint selection,
  checkpoint reload drift, non-finite maps, target mismatch, or incomplete feature tracing.

## Reproducibility

- **Run:** `artifacts/20260831-083111__breast-cancer__grid__small-cnn__seed-42__baseline`
- **Commit:** `f100658e504662aa0054408a1d60ee56c7321b07`; worktree clean at training start.
- **Config:** `configs/grid_cnn.yaml`; SHA-256
  `5b781c9fe43a77d22f9ae44fa02fb557e2830e0b08c14649bfe4be49ceb9e26f`.
- **Data:** scikit-learn Breast Cancer Wisconsin, 569 observations and 30 numerical features.
- **Split/training seed:** 42; 397 training, 86 validation, and 86 testing observations.
- **Environment:** Python 3.12.13, PyTorch 2.13.0, scikit-learn 1.9.0, CPU on macOS arm64.
- **Selection:** Highest validation macro F1; the first epoch wins a tie. The test set was
  evaluated only after checkpoint selection.

## Results

### Predictive checkpoint

- Training stopped after 31 epochs and selected epoch 19.
- Validation accuracy was 0.9651 and validation macro F1 was 0.9624.
- Test accuracy was **0.9070** and test macro F1 was **0.9017**.
- The training-derived majority-class baseline had 0.6279 accuracy and 0.3857 macro F1.
- Test confusion matrix, with rows/columns ordered malignant then benign, was
  `[[29, 3], [5, 49]]`.
- Reloading the saved checkpoint reproduced every test logit exactly; maximum absolute
  difference was 0.0.

### Grad-CAM tracing

- Generated finite predicted-class maps for all 86 test samples with shape `[86, 1, 5, 6]`.
- Every explanation target matched the frozen prediction, all predictions matched the saved
  evaluation output, and all 30 named features were traced for every sample.
- The target layer was the second convolution before ReLU. Standard Grad-CAM positive-evidence
  ReLU was retained; max normalization was used for display only and raw values were ranked.
- There were no padding pixels because 30 features exactly fill the 5 x 6 grid.
- One warm-up sample was excluded from a single batched pilot timing. The measured generation
  time was approximately 0.020 ms per sample; this is not yet a robust benchmark.

For predicted malignant class 0, the five largest class-stratified mean raw attributions were
mean symmetry, mean concave points, worst perimeter, worst texture, and smoothness error. For
predicted benign class 1, they were smoothness error, mean symmetry, compactness error, mean
concave points, and mean fractal dimension.

## Interpretation and limitations

This run establishes functional feature tracing, not explanation correctness. The predictive
quality gate, frozen-prediction checks, and mapping checks passed, so the checkpoint is eligible
for explanation evaluation. The class-stratified feature summaries are descriptive pilot results;
mixed-class averages are not global model importance. No feature should be treated as clinically
important from this experiment.

The evidence is limited to one dataset, one split/training seed, one CNN architecture, and one
explanation method. Grad-CAM faithfulness, stability across seeds, and agreement with other
methods remain untested. The validation-to-test performance difference also needs replication.

## Follow-up

1. **Completed 2026-08-31:** Integrated Gradients with signed values, convergence diagnostics,
   and distinct training-median sensitivity; see `2026-08-31-integrated-gradients-pilot.md`.
2. Implement feature-wise Occlusion Sensitivity with zero replacement in standardized space.
3. Run top-feature deletion against random and bottom-ranked controls before interpreting the
   Grad-CAM rankings as faithful.
4. Repeat the complete Grid condition for at least three training seeds.
