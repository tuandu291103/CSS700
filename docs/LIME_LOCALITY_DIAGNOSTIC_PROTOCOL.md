# Bounded LIME locality and representation diagnostic

Status: **prospective diagnostic v1, 2026-09-15**. Motivated by the already-observed
training preflight; not a hypothesis registered before that earlier result. Freeze this
protocol/config/code in a clean commit before calculating the alternative fits. This is
user-authorized pilot work, not committee approval or a change to the original LIME v1.

## Purpose and stopping rule

Determine whether narrower weighting improves only the explained anchor or also prediction
of the same generated evaluation neighbors; check the influence of weight scale and the
loss of within-bin values. This is one bounded diagnostic, **not model/explainer selection**.
Report all four conditions, stop after the registered comparisons, and retain the original
LIME v1 as the registered pilot specification. Do not add widths, seeds, samples, feature
selection, SHAP-rank objectives or new quality cutoffs after seeing the results. Any future
replacement protocol needs a separate justification and prospective validation, not automatic
promotion from eight training anchors. The next implementation milestone remains the full
LIME v1 test runner, with fit limitations visible; do not turn this into an indefinite sweep.

## Fixed evidence and leakage boundaries

Use only the accepted preflight
`artifacts/20260915-065803-601734__breast-cancer__lime__model-seed-42__training-preflight`.
Its 148-file inventory is frozen by SHA-256 in `configs/lime_locality_diagnostic_v1.yaml`;
the digest hashes the compact, key-sorted JSON mapping of relative paths to file hashes.
Verify every file, accepted status, source commit/config hash and all 26 original input
hashes. Regenerate the saved neighborhoods from training data only as an integrity check.
Match their exact arrays, metadata and saved probabilities/feature axes before fitting.

Hold fixed the eight training IDs, model seed 42, RF/XGB probabilities, 30 ordered features,
explainer seeds 11/29/47, 5,000 fit rows including the anchor and 2,000 independent generated
evaluation rows excluding the forced anchor. Reuse stored probabilities: **zero new model
calls and zero test explanations**. No real validation/test row or SHAP attribution enters
the diagnostic. Reading their existing file hashes for source integrity is not new analysis.

## Four conditions, three predeclared contrasts

All conditions fit Ridge with alpha 1, free intercept and Cholesky solver. No clipping.

| Condition | Surrogate features | Fit kernel width | Fit weight scaling | Question |
|---|---|---|---|---|
| Reference | Saved binary same-quartile indicators | `0.75 * sqrt(30)` | Original | Reproduce accepted v1 coefficients/predictions |
| Narrow raw | Same binary indicators | `0.25 * sqrt(30)` | Native raw weights | Combined effect of narrower weighting and changed weight mass |
| Narrow matched mass | Same binary indicators | `0.25 * sqrt(30)` | Multiply all fit weights by reference sum / narrow sum | Control total weight mass relative to fixed Ridge penalty |
| Continuous, same neighbors | `(inverse_value - anchor_value) / training_std` | Original binary-distance width | Original | Diagnostic for alternative representation with inputs/weighting held fixed |

The narrower width is a single investigator-chosen contrast (three times narrower), not an
estimated optimum. Distances for **all** conditions remain the saved binary-space Euclidean
distances. Weights are `exp(-distance^2 / (2 * width^2))`. LIME uses distance-based weights
when fitting its local regressor. [LIME API](https://lime-ml.readthedocs.io/en/latest/lime.html).

The weight-mass control is necessary because the weighted Ridge objective is
`sum_i w_i * (y_i - a - x_i @ beta)^2 + alpha * ||beta||^2`.
Multiplying all weights by c is equivalent to dividing alpha by c, so raw narrowing changes
more than relative locality. Matching the reference total weight controls this global scale;
it does not make the fit geometry or information content equal. This equation and scaling
argument specify our diagnostic control, grounded in the regularized least-squares objective.
[Ridge API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).

The continuous condition uses the same already-generated inverse rows, including their
quartile-based sampling, and the original binary distances. Its population standard deviations
come only from the 397 standardized training rows. The anchor becomes a zero vector. This
is a **hybrid diagnostic surrogate, not native continuous LIME** and not a proposed replacement
explainer. Changing the representation also changes coefficient units and penalty geometry;
do not interpret the contrast as an isolated causal effect of binning, or compare coefficient
magnitudes/ranks between representations. Independence/correlation limitations of the original
sampler remain in every condition.

Each alternative is paired with reference for identical model, sample ID and explainer seed.
Additionally interpret raw versus matched narrowing descriptively to expose the weight-scale
confound. No choice is selected by agreement with SHAP or by a favorable single case.

## Outcomes: do not change the evaluation target

For each fitted surrogate, retain anchor absolute probability error; weighted fit metrics;
own-kernel evaluation metrics; and **common-reference evaluation** R-squared/RMSE/MAE on the
unchanged 2,000 rows with original v1 weights. The latter is the primary neighborhood-fit
comparison. An own-kernel improvement alone may just reflect a different weighting of errors.
No evaluation response is used for fitting or for selecting settings in this diagnostic.

Retain negative R-squared, null with a flag for target variance <= `1e-15`, and predictions
outside [0,1]. Use the existing v1 diagnostic definitions. Technical acceptance is separate
from good approximation; poor fits remain visible and no quality-based sample filter is used.

Also record total fit weight, alpha / total weight, anchor weight fraction, effective sample
size both including and excluding the anchor, weighted mean changed-bin count and standardized
input distance excluding the anchor, and the count of non-anchor fit rows with <=5 bin changes.
The <=5 count describes support only; it is not a selection or acceptance threshold.

Record the anchor's Ridge self-influence (weighted smoother diagonal): with weighted-centered
design Xc, anchor row xc0 and `A = Xc.T @ W @ Xc + alpha*I`,
`h00 = w0 * (1 / sum(w) + xc0.T @ solve(A, xc0))`.
It measures the fitted anchor's sensitivity to its own response, not causal influence.
Narrow weighting can make anchor error small through self-influence without useful nearby
support. Check this formula against finite-difference refitting on a synthetic fixture.

## Validation, summaries and deliverables

- Refuse failed, incomplete, mutated or unregistered source bundles. Verify class targets,
  arrays, original source hashes, package versions, reference coefficients/predictions and
  diagnostics. Use unchanged numerical equivalence tolerances `atol=1e-10`, `rtol=1e-8`.
- Test exact reference reproduction, independent weighted normal equations, weight/alpha
  scaling equivalence, common-evaluation weights, held-out-response noninterference, target
  identity, source corruption rejection and retention of failure metadata.
- Save 192 fits (8 anchors x 3 explainer seeds x 2 models x 4 conditions), 144 alternative-
  reference paired rows, coefficient/prediction arrays, weights/designs, sample/seed-level
  diagnostics, descriptive summaries, source hashes/config/commit and explicit status.
- Report median/min/max and paired error differences, separately by family/condition and
  explainer seed; overall medians across 24 repeated fits are descriptive only. No pooled
  independence, confidence intervals, significance tests, optimal-setting or novelty claims.
- Keep all artifacts append-only and original v1 sources untouched. Source commits are not
  backups of raw artifacts. A session report must include the on-track check and next task.

## Schedule check

The Grid three-method milestone and fixed-split seed replication are complete; tree predictive
baselines and TreeSHAP replication are complete on the pilot dataset. The expected CNN matrix
still has only its three Grid/pilot-dataset conditions, out of 12 planned (two datasets x two
encodings x three seeds). This count is **not a percentage of thesis completion**. LIME test
analysis, the second dataset, authoritative SOM2IMG collision rules and committee approval
remain outstanding. This bounded check supports trustworthy comparisons; repeated tuning must
not displace those full-scope dependencies. The exact defense calendar is not confirmed.
