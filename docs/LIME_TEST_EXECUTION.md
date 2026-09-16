# Frozen LIME v1 test execution contract

This implements the already registered test analysis in `docs/LIME_COMPARISON_PROTOCOL.md`.
It does not amend its scientific settings or replace v1 with a locality-diagnostic variant.
Commit this implementation and its synthetic tests before generating any test explanations.
Earlier non-LIME test results and the training-only diagnostic are already known.

## Inputs and gates

- Keep `configs/lime_seed_42_v1.yaml` byte-for-byte unchanged (SHA-256
  `bf6c361aadafeddf57326bfc72cb1ba9a520f728269fef0e4d84e3066c92d3a6`).
- Revalidate the accepted 48-fit training preflight, its 148-file inventory, original source
  inputs, installed package versions and active LIME source hashes before loading models.
  The frozen inventory digest is
  `49b11bf5fef8b238f768e71951c0f23657c18c29853367bbdc3d70d11de82505`.
- Revalidate the registered seed-42 accepted TreeSHAP run, all four conditions and its 39-file
  inventory digest `a53b5b759d74435d9e62a5441fb1e1edb67c9ab4deb6fef5d0316dbbf5349964`.
  Inventory digests hash compact sorted-key JSON mappings of relative paths to file SHA-256s.
  Comparisons use only the full-397-row background; sensitivity files are integrity inputs.
- Verify ordered test IDs, labels, feature/class names, training-only scaler and unchanged
  model hashes before deserialization. Reproduce frozen test predictions (same targets,
  probability `atol=1e-7`, `rtol=1e-6`) and check each generated fit's anchor against them.
- Run from a clean commit, in a no-PyTorch process with one native thread. Preserve failures
  in new append-only directories. Recheck every tracked source hash at completion.

## Execution and outputs

`make lime-test PYTHON=.venv/bin/python` generates 258 shared neighborhood pairs and 516
targeted fits: 86 test IDs x three explainer seeds x two frozen models. Each pair retains the
5,000 fitting / 2,000 independent generated evaluation inputs, indicators, distances, weights,
derived seeds, training IDs, exact quartiles and anchor-bin IDs. Each model output retains all
probabilities, coefficients, predictions and diagnostics, including negative/undefined R-squared
and out-of-range predictions. No clipping, quality filtering, new tuning or model training.

For all fits, independently solve the weighted centered Ridge normal equations and verify
coefficients, intercept and generated-neighborhood predictions at the unchanged surrogate
tolerance (`atol=1e-10`, `rtol=1e-8`). Recompute both sets of diagnostics and anchor errors.
Save raw evidence before this gate, and never interpret a passing numerical gate as good fit.
Training same-seed repeat checks come from the accepted preflight; test sampling repeats are
the registered seeds 11, 29 and 47, not new model-training seeds.

Export 15,480 named coefficient rows with target/label identity, signed coefficient, bin ID,
stable-index absolute rank and fit diagnostics. Exact quartile boundaries remain in each
case's metadata. Re-read the CSV to verify named-feature matrix identity. Tied export ranks
use original feature order; rank correlations use average ties and comparisons retain
cutoff-tie flags under the existing common metric implementation.

Create 15 separate comparisons, each retaining all 86 rows (1,290 total):

- Six same-model LIME–TreeSHAP comparisons: seed 11 primary, 29 and 47 separate repeats.
- Six within-model LIME sampling-seed pairs: three per family with the model fixed.
- Three RF–XGBoost LIME comparisons: matching explainer seed, same-target eligibility only.

Retain different-target and constant-vector exclusions explicitly. Attach each LIME side's
anchor error and independent generated-evaluation RMSE/R-squared to every comparison row.
Keep 90 agreement summaries (15 comparisons x six all/target/correctness groups), with 1,000
paired-sample bootstrap resamples, seed 2026; omit intervals below ten eligible cases. Never
pool repetitions into an independent-sample total. Retain zero-size groups.

Save 30 fit-quality summaries (two families x three seeds x five all/target/correctness groups),
including all technically valid cases, not just rank-eligible ones. Show these descriptive
distributions alongside rank agreement in the session report; no fit-quality intervals or
post hoc significance tests are introduced. Runtime includes artifact writing and checks,
so it is not a cross-method efficiency benchmark.

## Interpretation and stop rule

Agreement is not faithfulness or accuracy. LIME coefficients describe same-quartile membership
in a generated-neighborhood surrogate; SHAP describes a different replacement game. Preserve
this distinction even for the same model, target and original feature names. A poor surrogate
can produce stable rankings. The 86 cases are the existing repeatedly analyzed pilot test set,
not a new external validation cohort. No clinical or causal conclusions follow.

After the single frozen test pilot, report the evidence and update the project plan. Do not
launch a test-driven repair/tuning loop. The second dataset and authoritative SOM2IMG mapping
remain full-study dependencies, not things additional pilot comparisons can resolve.
