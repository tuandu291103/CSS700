# Local LIME pilot protocol

Status: **provisional v1, 2026-09-14**, to be frozen in a clean source commit before the
training preflight and any test explanations. User-authorized research continuation, not
committee approval. Executable settings: `configs/lime_seed_42_v1.yaml`. Choices below are
investigator decisions, not claims that library defaults are optimal. Results belong in a
separate log; do not revise settings in response to observed test agreement or fit quality.

## Question and scope

For the existing seed-42 RF and XGBoost models, how well does LIME's surrogate approximate
the chosen class probability, how sensitive are feature rankings to neighborhood sampling,
and how do its rankings agree with TreeSHAP on the same model and target?

LIME fits a locally weighted surrogate to a classifier's behavior. That is different from
an exact additive accounting of the original prediction. This motivates separate fit and
sampling-stability diagnostics. [Ribeiro, Singh and Guestrin (2016)](https://arxiv.org/abs/1602.04938).

Use the frozen models and split registered by `configs/tree_shap_seed_42_v1_1.yaml` (its
exact hash is locked in the LIME config). Reuse the 397 training / 86 validation / 86 test
partition and ordered 30-feature standardized representation. All preprocessing and LIME
reference statistics use training rows only. No model retraining, selection or dependency
upgrade. This is a numerical binary pilot; mixed types and multiclass are out of scope.

The current milestone implements **training preflight only**, on the first eight training
IDs in saved split order. It makes no new inference on real validation/test rows and produces
no test LIME explanations. Existing source artifacts are read only for integrity checks.
The planned test pilot below still requires its runner, exports and comparison validation.
Previous test predictions and non-LIME results are already known; the protocol is prospective
for LIME, not a claim that the whole study uses a newly untouched test set.

## Frozen settings and coefficient meaning

| Component | v1 choice |
|---|---|
| Library | Installed LIME 0.2.0.1; preserve its source-file hashes and package snapshot |
| Model function | Original estimator `predict_proba`, ordered classes [0, 1]; no probability renormalization |
| Target | Frozen predicted class, one target per fit |
| Reference data | All 397 standardized training rows; no training labels supplied to LIME |
| Representation | Training-fitted quartile bins; one binary same-bin indicator per original feature |
| Feature selection | `none`, all 30 coefficients retained in original feature-index order |
| Distance and kernel | Euclidean on binary indicators; width `0.75 * sqrt(30)`; weight `exp(-d^2 / (2 * width^2))` |
| Local regressor | Weighted Ridge, alpha 1, free intercept, explicit Cholesky solver |
| Fit neighborhood | 5,000 rows including the forced original instance |
| Independent fit-evaluation neighborhood | 2,000 generated rows, excluding its forced anchor; never used for fitting |
| Explainer seeds | 11 primary; 29 and 47 registered sampling repeats; model seed stays 42 |
| Feature ranking | Absolute LIME coefficient, with signed coefficient and bin identity retained |
| Fit-quality exclusion | None: report all technically valid fits and their errors |

LIME supports numeric feature IDs and all-feature fitting. Its returned fit score measures
the surrogate on the fitting neighborhood; it is not independent validation. Asking for
one target at a time also avoids ambiguity in the scalar score/local-prediction fields when
multiple labels are explained. [LIME API](https://lime-ml.readthedocs.io/en/latest/lime.html).

Here `z_j = 1` means the generated feature lies in the same training-derived bin as the
explained case. The anchor is all ones and the surrogate is `g(z) = intercept + sum(beta_j*z_j)`.
Coefficients describe bin-membership effects in this fitted surrogate, not continuous-feature
derivatives or TreeSHAP contributions. Do not parse rounded display strings as feature IDs,
multiply the coefficients by the original standardized values, or treat the surrogate
intercept as SHAP's expected prediction. Preserve exact quartile boundaries and anchor-bin
IDs alongside the original names. Reject nonfinite inputs, duplicate names and constant
training columns in this bounded adapter; do not silently repair their discretization.

The installed sampler draws each feature's bin using its training frequency, then generates
within-bin values using its native undiscretization. It does not preserve feature correlations
or condition on the other feature values. Binary-distance weighting does not guarantee
nearness in the original measurement space; many features can change together. Record the
actual inverse/model-input rows and weights, not only a seed. The upstream
[tabular source](https://github.com/marcotcr/lime/blob/master/lime/lime_tabular.py) and
[discretizer source](https://github.com/marcotcr/lime/blob/master/lime/discretize.py) provide
context; the installed 0.2.0.1 files, not moving `master`, define this run's implementation.

Ridge is regularized least squares; the explicit solver fixes an implementation choice and
does not constrain predictions to [0, 1]. Retain out-of-range surrogate predictions rather
than clipping them. [Ridge API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).

## Randomness and matched neighborhoods

Derive each sampler's integer state from
`numpy.random.SeedSequence([explainer_seed, original_sample_id, role]).generate_state(1, dtype=uint32)[0]`,
where role 0 is fitting and role 1 is independent neighborhood evaluation. Construct a fresh
explainer for each sample/seed/role. Do not use Python's process-dependent `hash`, traversal
order, a persistent advancing RNG or model family in seed derivation. Share each generated
design and inverse neighborhood between RF and XGBoost; keep their targets separate.

For evaluation, generate 2,001 native rows, discard the forced anchor, and evaluate the
already-fitted surrogate on the remaining 2,000. Fit/evaluation use separate pseudorandom
streams from the same declared distribution. This evaluates generalization to generated
neighbors, not held-out real-world accuracy or causal validity. Different generated rows
may legitimately share a binary representation. No sampling-budget/kernel/bin sweep is
authorized by this v1 protocol; any extension gets a separate prospective version.

## Technical acceptance versus scientific fit

Required technical checks:

1. Revalidate source hashes, saved feature/class order, split IDs and training-only scaler
   before deserializing frozen models. The later test runner must also compare original
   probabilities/predictions with saved references, using existing reload tolerances.
2. Confirm all reference IDs belong exclusively to training, every generated input is finite,
   the anchor is preserved exactly, and binary columns equal independently reconstructed
   same-bin membership of the actual model inputs. Reject an unreviewed LIME release.
3. Keep all feature IDs once, and verify `intercept + design @ coefficients` agrees with
   the returned surrogate predictions (`atol=1e-10`, `rtol=1e-8`). This checks the surrogate's
   own arithmetic, **not** equality to the original model's prediction.
4. Independently recompute weighted fit metrics. Tests compare the adapter with the public
   LIME API and with weighted Ridge normal equations; cover class 0, complementary binary
   targets, a constant predictor, invalid classes, leakage and a deliberately poor fit.
5. At the first training anchor for each of the three explainer seeds, regenerate both
   neighborhoods exactly and repeat both model fits. Compare coefficients, original
   probabilities and surrogate predictions at the unchanged repeat tolerance above.
6. Use a clean commit, isolated no-PyTorch process, one native thread, append-only outputs,
   retained failures, final source-hash recheck and explicit completion count.

Report for **every** fit: original target probability, surrogate intercept and anchor
prediction, absolute anchor error, and separate fit/evaluation weighted RMSE, MAE and R-squared.
Calculate weighted variance around each neighborhood's weighted mean probability. If that
variance is at most `1e-15`, store R-squared as null with an explicit near-constant-target
flag; retain the native library score separately. Keep negative R-squared values. Also report
effective sample size `(sum w)^2 / sum(w^2)` and the unweighted fraction of surrogate predictions
outside [0, 1]. These are continuous scientific diagnostics, not post hoc acceptance thresholds.

Poor fit or unstable rankings must not trigger sample removal, additional samples or a new
kernel chosen after looking at test outcomes. A successful technical preflight establishes
compatibility only. Low model–surrogate error, if observed, still does not establish that the
generated neighborhood is realistic. No exact SHAP reconstruction gate applies to LIME.

## Planned test analysis (not yet executed)

Retain all 86 test sample IDs, including errors: two models x three explainer seeds = 516
targeted fits. Generate six within-model seed-pair comparisons (11–29, 11–47, 29–47, separately
for each family). These measure stochastic-explainer sensitivity with the model fixed,
unlike the earlier TreeSHAP model-training-seed experiment.

Primary same-model comparison: LIME seed 11 versus full-397-background accepted TreeSHAP
from `artifacts/20260912-171418-618975__breast-cancer__tree-shap__seed-42__pilot`, for each
family and identical predicted target. Revalidate that accepted bundle before using it.
Repeat the same comparisons at explainer seeds 29 and 47, separately, not selecting whichever
agrees best. SHAP's replacement distribution and LIME's indicator coefficients differ even
with the same model and output. No shared raw-magnitude scale or superiority claim follows.

Secondary comparisons: RF versus XGB LIME at matching explainer seeds and matching predicted
targets. Tree–CNN LIME comparisons and additional training seeds are deferred. Use existing
Spearman, top-5/top-10 overlap/Jaccard and cutoff-tie handling; report constant-vector and
different-target exclusions, not arbitrary ranks for all-zero explanations. All technically
valid fits remain eligible regardless of fit quality; show their diagnostic distributions
prominently beside agreement. No accuracy or fidelity ranking of explainers is licensed.

Use separate comparison/seed summaries, predicted-target and correctness-group counts, and
1,000 paired sample bootstrap resamples (seed 2026) for conditional mean intervals. Omit
subgroup intervals below ten eligible rows. Retain zero counts and all sample-level values.
Do not pool repeated sample/seed pairs as independent observations, calculate a seed-population
variance from three runs, or add formal significance tests after inspecting the results.

## Outputs and decision gate

The training preflight retains 24 shared neighborhood pairs and 48 primary fitted surrogates,
six same-seed repeat checks, per-fit diagnostics, signed coefficient arrays, exact training
IDs/boundaries and generated inputs, probabilities, weights, package/source hashes, CPU/thread
settings, runtimes and any errors. Its command is `make lime-preflight PYTHON=.venv/bin/python`.
Repeat work is included in recorded preflight runtime; it is not an efficiency benchmark.

After reporting the preflight, implement and test the full test runner and named-feature/
comparison exports against this protocol. Commit that implementation before generating any
test LIME bundle. Failures require an explicit repair and new append-only run, not overwritten
evidence. Generated artifacts are ignored by Git and need separate backup arrangements.
Second-dataset and authoritative SOM2IMG decisions remain independent full-thesis dependencies.
