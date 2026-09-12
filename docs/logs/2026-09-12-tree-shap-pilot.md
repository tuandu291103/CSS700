# Session report: TreeSHAP pilot and numerical-routing safeguard

## Outcome

Implemented and ran the seed-42 TreeSHAP pilot, but **the full pilot is not accepted**.
The Random Forest (RF) 100-row-background condition failed training-only reconstruction.
Investigation exposed inconsistent near-threshold routing, including risks in the full
background. A new safeguard now rejects RF before generating explanations. XGBoost passed
the original numerical checks for both backgrounds. No RF agreement finding is promoted.

## Work completed

- Committed the frozen provisional v1 configuration, probability-output adapters, explicit
  binary-class normalization, source-hash checks, training-only backgrounds, paired ranking
  metrics, raw artifact exports, command-line entry point and isolated tree-runtime CI job.
- Verified the unchanged numerical XGBoost booster adapter without changing its categorical
  capability flag, fitted trees, predictions, or dependencies. Kept PyTorch out of tree runs.
- Preserved both clean-commit attempts, diagnosed the RF failure using training values only,
  and added synthetic regression tests plus an auditable pre-explanation routing guard.
- Updated the README, research design, project plan, open questions and thesis outline.

The September 10 commit attempt had been interrupted by a usage-limit approval denial;
implementation resumed successfully on September 12. No committee approval is inferred.

## Reproducibility

Both attempts used `configs/tree_shap_seed_42.yaml`, SHA-256
`b7d376554af48c0d7b8d308243b7d736e631afece0af00c81dc7dc2b085844f0`.
The frozen split remains 397 training / 86 validation / 86 test rows, with 30 features.
Primary background: all training rows; sensitivity: the registered 100-row subset, seed 2026.
No model was retrained or reselected; targets, output scale and tolerances were unchanged.

| Attempt | Clean code commit | Local artifact directory under `artifacts/` | Status |
|---|---|---|---|
| Initial v1 | `87a79621008cfaf423cb46894e19fd00047fe70f` | `20260912-093331-466766__breast-cancer__tree-shap__seed-42__pilot` | `quality_gate_failed` |
| Strengthened routing guard | `847481867367a6c1b62bb5a60ff186b3bf00bf5d` | `20260912-093956-834341__breast-cancer__tree-shap__seed-42__pilot` | `quality_gate_failed` |

Each bundle contains its manifest, exact config, source hashes, split IDs, condition checks
and available raw explanations. The guarded RF condition retains `routing_audit.json` and
has no smoke/test explanation arrays. All 21 original source-file hashes were rechecked and
remain unchanged. Generated bundles stay local under the repository's existing ignore policy;
committed code and this report do not constitute a remote backup of those bundles.

## Validation results

Initial-run maximum absolute probability-reconstruction residuals:

| Condition | Residual | Interpretation |
|---|---:|---|
| RF, 397 rows | `6.30994e-6` on test rows | Original numerical gate passed; later routing audit prevents promotion |
| RF, 100 rows | `2.50115e-5` on training smoke rows | Failed; no test explanations generated |
| XGBoost, 397 rows | `2.02971e-7` on test rows | Original condition checks passed |
| XGBoost, 100 rows | `2.02840e-7` on test rows | Original condition checks passed |

Tolerance remains `abs(residual) <= 1e-5 + 1e-5 * abs(probability)` for each sample/class.
The guarded rerun stopped RF earlier and skipped the sensitivity stage because both primary
conditions were not accepted. Its XGBoost primary result again passed. Run times were 5.50 s
and 1.22 s respectively; the shorter incomplete rerun is not a performance comparison.

## Diagnosis and research significance

RF prediction converts inputs to float32. SHAP 0.52.0 retains background precision while its
interventional native routine narrows thresholds to float32. Near a threshold, these rules
can disagree with the original predictor. See the [RF prediction contract](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
and [versioned SHAP implementation](https://github.com/shap/shap/blob/v0.52.0/shap/cext/tree_shap.h).

The training-only audit checked all 6,350 internal nodes: 2,520,950 node/value comparisons per
role for the full background. It found 198 background-role and 55 query-role disagreements;
the 100-row subset had 46 and 17. These include potentially reachable feature mixtures, not
198 or 55 incorrect predictions. A separate complete-path emulation found five full-background
rows with probability differences of magnitude 0.0025; only sample ID 514 was in the subset,
giving its mean discrepancy of `0.0025 / 100 = 2.5e-5`, consistent with the failed residual.

Casting the background alone removed the subset smoke failure but left a full-background
discrepancy, so it was not adopted. Neither library code nor model thresholds were patched.
The new guard records mismatching comparisons and rejects the condition rather than silently
changing its explanatory function. It is a conservative training-only check, not a proof for
all future inputs.

The useful finding is methodological: a small aggregate reconstruction residual can conceal
numerical-routing disagreement. This is local implementation evidence, not a demonstrated
novel contribution, a general failure of TreeSHAP, or a clinical claim. Saved partial ranking
tables are diagnostic only; the guarded figure is visibly labeled incomplete and was reviewed.
Cross-model probability/logit and background differences remain separate interpretation limits.

## Verification and next step

All **47 tests passed** in 21.69 s, including isolated real-model and near-threshold fixtures.
Lint, formatting and whitespace checks passed. Raw arrays were checked for finite values and
expected `[86, 30, 2]` shape wherever test explanations existed. GitHub CI was not used as
evidence for these local checks.

Next: implement and independently validate a numerically faithful RF adapter in a versioned
repair, preserving the original model's behavior for background/query feature mixtures.
Keep the current failed bundles and unchanged tolerances, then rerun the seed-42 pilot from
a clean commit. Only after acceptance should we replicate saved seeds 7 and 2026.
The broader thesis remains incomplete: authoritative SOM2IMG collision handling, a second
dataset, LIME design and committee review are still outstanding.
