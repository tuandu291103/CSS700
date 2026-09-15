# LIME protocol and training-only preflight

Report date: **2026-09-15**. Protocol frozen September 14, Los Angeles time. The run started
at 2026-09-15 06:58 UTC (September 14, 23:58 PDT). No LIME test explanations were generated.

## Brief summary

We froze the provisional LIME protocol, implemented auditable neighborhood generation and
local fitting, and completed **48 training-anchor fits plus six same-seed repeat checks**.
All technical checks passed; the complete test suite now has **66 passing tests**.

The important caution is scientific, not a software failure: several surrogates approximated
the original model poorly. Median absolute error at the explained case was **0.081 for RF**
and **0.302 for XGBoost**. A reproducible LIME output is not automatically a trustworthy
explanation. This finding changes the immediate priority to a training-only locality/fit
diagnostic before implementing the full test comparison, while preserving v1 unchanged.

## What we implemented

The separately versioned design is in `docs/LIME_COMPARISON_PROTOCOL.md` and
`configs/lime_seed_42_v1.yaml`. It fixes the original seed-42 models, predicted-class
probability, 397 training reference rows, all 30 named features, training-quartile membership,
weighted Ridge settings, 5,000 fit rows and 2,000 independently generated evaluation rows.
Explainer seeds 11, 29 and 47 vary neighborhood sampling, not model training.

The adapter retains exact generated model inputs, binary feature representations, weights,
coefficients, intercepts and both neighborhoods' predictions. Feature identity, native LIME
API equivalence, independent weighted-regression equations, binary class handling, same-seed
repeatability and separation of fitting from evaluation are tested. Constant-target R-squared
is explicitly undefined, and poor fits or out-of-range surrogate values are not removed.

LIME is a locally fitted surrogate, so we assess approximation error separately from the
surrogate's own arithmetic. Its coefficients are not TreeSHAP contributions and do not share
the same additive guarantee. [Original LIME paper](https://arxiv.org/abs/1602.04938).

## Training-only results

The eight training IDs were 460, 165, 122, 455, 412, 397, 182 and 429, selected as the first
eight rows in the frozen training order, without selecting on explanation quality.
Each model has 24 fits: eight cases x three explainer seeds. These are repeated measurements,
not 24 independent real-world examples. No confidence intervals or generalization claims
are made from this compatibility sample.

| Diagnostic across the 24 fits per model | Random Forest | XGBoost |
|---|---:|---:|
| Anchor absolute probability error, median [min, max] | 0.081 [0.003, 0.267] | 0.302 [0.082, 0.472] |
| Generated-evaluation weighted R-squared, median [min, max] | 0.219 [0.068, 0.705] | 0.252 [0.198, 0.414] |
| Generated-evaluation weighted RMSE, median [min, max] | 0.126 [0.077, 0.138] | 0.299 [0.267, 0.309] |
| Anchor surrogate predictions outside [0, 1] | 3 / 24 | 18 / 24 |

These errors are in class-probability units, not changes in classification accuracy. The
evaluation set consists of independent synthetic neighbors, not the real validation/test
partition. Unconstrained Ridge can predict outside [0, 1]; those values were retained, not
clipped. [Ridge documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).

Exploratory inspection of the 24 saved neighborhoods found that an average of 22.42–22.59
of 30 feature-bin indicators changed per non-anchor fitting row. Mean non-anchor weights
were 0.513–0.516, and fit effective sample sizes were 4,973–4,976 out of 5,000. Thus the
registered weighting distinguished generated rows only weakly in this sample. This suggests
examining locality and representation, but does not isolate the cause of poor fit: surrogate
linearity, model behavior and independently sampled correlated features may also matter.
This inspection was post-preflight diagnosis on training anchors, not a pre-registered result
or a parameter-selection sweep. No settings were changed in response.

## Reproducibility and verification

- Run: `artifacts/20260915-065803-601734__breast-cancer__lime__model-seed-42__training-preflight`.
- Clean implementation/protocol commit: `dfa0327d3f14fef34f75f5ca24ceff15e10ee757`.
- Config SHA-256: `bf6c361aadafeddf57326bfc72cb1ba9a520f728269fef0e4d84e3066c92d3a6`.
- LIME 0.2.0.1, NumPy 2.5.2, SciPy 1.18.1, scikit-learn 1.9.0, XGBoost 3.4.1,
  Python 3.12.14; package versions and installed LIME source hashes are in the manifest.
- 24 shared fit/evaluation neighborhood pairs, 48 primary coefficient/probability bundles
  and six repeated fits at training ID 460. RF and XGB used identical generated inputs.
- 336,000 primary model-input rows in 48 calls, plus 42,000 rows in six repeat calls.
  Recorded total runtime: 4.39 s with one native thread and no PyTorch import; not a benchmark.
- All 26 tracked input-file hashes remained unchanged. No model was retrained or altered.
- A read-only artifact audit independently reconstructed bin membership and surrogate
  predictions, recomputed R-squared/RMSE and solved weighted Ridge normal equations for all
  48 fits. All checks matched the saved outputs within declared numerical tolerances.
- All 66 automated tests passed; lint, formatting and whitespace checks were clean.

Generated bundles remain local/ignored. Source commits do not back up their raw arrays.

## Next step and boundaries

Before expanding to the planned 516 test fits, specify a **bounded, training-only LIME
locality/representation diagnostic**. Preserve this v1 result, declare any alternative
settings and selection rules in a separate prospective version, and keep the real test
partition out of the investigation. The purpose is to understand approximation failure,
not to optimize agreement with SHAP or to exclude inconvenient cases. The registered v1
poor-fit cases remain technically valid and fully reported; no new quality cutoff is imposed.

The eventual comparison still needs a validated test runner, named-feature exports,
accepted-SHAP source revalidation and paired analysis. No LIME–SHAP rank result, full LIME
pilot completion, general LIME failure, clinical claim or established novelty is asserted
here. Second-dataset, authoritative SOM2IMG and committee-review decisions remain open.
