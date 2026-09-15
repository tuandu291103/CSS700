# TreeSHAP three-seed replication: results and research checkpoint

Report and verification date: **2026-09-14**. The accepted experiments and stability analysis
ran on **2026-09-12**. This session audits and documents those completed runs; it does not
retrain models, generate new test explanations, or change the frozen analysis settings.

## Brief summary

The pilot TreeSHAP milestone is complete: all **12 conditions** (three model seeds, two tree
families, two training-background sets) passed their technical gates. Local feature rankings
were highly, but not perfectly, stable across model-training seeds. Lower tree–CNN agreement
also appeared in the replicas. These are descriptive results on one dataset and one fixed
split, not evidence of explainer superiority, clinical relevance, or established novelty.

We are progressing through the proposal's implementation sequence. The full two-dataset,
Grid/SOM2IMG study is not complete; LIME, the second dataset and authoritative SOM2IMG
collision handling remain open. This checkpoint does not imply committee approval or a
guaranteed defense date.

## What was completed

- Reused the saved, validation-selected RF and XGBoost models for seeds 7 and 2026, with
  accepted seed 42 as the reference. No model, dependency, tolerance or scientific setting
  was changed. The protocol was frozen before these replica explanations in
  `docs/TREE_SHAP_REPLICATION_PROTOCOL.md`.
- Retained the same 397/86/86 split, 30 ordered features, training-only scaler, probability
  targets, absolute-score rankings, full 397-row background and fixed 100-row sensitivity.
- Revalidated original source files, raw class attributions, named-feature exports,
  reconstruction and RF routing-equivalence checks before comparing seeds.
- Kept each seed pair and background separate: **12 comparisons, 1,032 sample-pair rows
  and 72 grouped summaries**. These repeated observations are not 1,032 independent samples.

## Primary local training-seed stability

Background: all 397 training rows. Eligible samples have the same predicted class in both
models. RF means Random Forest; XGB means XGBoost. Top-k overlap is intersection size / k.

| Family | Seed pair | Eligible n / 86 | Mean Spearman [95% interval] | Median Spearman | Median top-5 | Median top-10 |
|---|---|---:|---|---:|---:|---:|
| RF | 7–42 | 85 | 0.936 [0.924, 0.947] | 0.953 | 1.00 | 0.90 |
| RF | 7–2026 | 86 | 0.936 [0.923, 0.947] | 0.960 | 1.00 | 0.90 |
| RF | 42–2026 | 85 | 0.936 [0.923, 0.946] | 0.950 | 1.00 | 0.90 |
| XGB | 7–42 | 86 | 0.962 [0.957, 0.966] | 0.969 | 0.80 | 0.90 |
| XGB | 7–2026 | 86 | 0.958 [0.953, 0.962] | 0.965 | 1.00 | 0.90 |
| XGB | 42–2026 | 86 | 0.950 [0.945, 0.954] | 0.956 | 1.00 | 0.90 |

Intervals use 1,000 paired sample bootstrap resamples, seed 2026, conditional on each fixed
model pair and its eligible samples. They do not estimate population-level training-seed
uncertainty; no pooled interval or formal cross-family contrast is reported. Subgroups below
ten eligible rows have no interval. No constant-vector exclusions or exact top-k cutoff ties
occurred in these comparisons. Different-target rows remain in the raw diagnostics.

All three RF models predicted the same class for 85/86 samples; XGB did so for 86/86. Class
agreement did not imply identical probabilities: maximum pairwise probability differences
were 0.0425–0.0900 for RF and 0.0745–0.1239 for XGB. Similarly, median top-five overlap of
1.00 does not mean every sample's top-five set was unchanged.

## Registered 100-row sensitivity

The same fixed subset was reused across seeds; it was not selected for favorable agreement.

| Family | Seed pair | Eligible n | Mean Spearman [95% interval] | Median Spearman |
|---|---|---:|---|---:|
| RF | 7–42 | 85 | 0.940 [0.931, 0.948] | 0.952 |
| RF | 7–2026 | 86 | 0.942 [0.933, 0.950] | 0.956 |
| RF | 42–2026 | 85 | 0.938 [0.928, 0.946] | 0.947 |
| XGB | 7–42 | 86 | 0.960 [0.955, 0.964] | 0.968 |
| XGB | 7–2026 | 86 | 0.957 [0.952, 0.962] | 0.964 |
| XGB | 42–2026 | 86 | 0.951 [0.946, 0.955] | 0.957 |

All six sensitivity comparisons had median top-five overlap 1.00 and top-ten overlap 0.90,
with no constant-vector exclusions or exact cutoff ties. Within each frozen model, full-versus-
subset median Spearman ranged from 0.9907 to 0.9929 across the six model/seed combinations.
This is one size/subset check, not evidence of robustness to arbitrary reference distributions.

## What replicated, and what did not

Across seeds 7, 42 and 2026, full-background RF–XGB mean Spearman was 0.609, 0.620 and 0.598
(85, 84 and 85 eligible samples). The six tree–CNN comparisons at each seed had means in
the overall range 0.138–0.380, with 79–82 eligible samples. Thus, lower tree–CNN agreement
persisted descriptively across the registered seeds.

Do not promote a universal ordering of CNN explainers: for XGB seed 2026, mean agreement
with IG-zero was 0.1416 versus 0.1444 with Grad-CAM. The seed-42 ordering is not universal,
and this small difference is not a tested method contrast. More importantly, tree probability
SHAP, CNN logit-based explanations and Grad-CAM differ in fitted function, reference and
method. Rank comparisons do not remove these differences. Stability and agreement do not
establish faithfulness; tree probability-deletion would require its own frozen protocol.

## Provenance and verification

Accepted bundles (repository-relative paths):

- Seed 7: `artifacts/20260912-173614-010318__breast-cancer__tree-shap__seed-7__replicate`.
- Seed 42: `artifacts/20260912-171418-618975__breast-cancer__tree-shap__seed-42__pilot`.
- Seed 2026: `artifacts/20260912-173621-741350__breast-cancer__tree-shap__seed-2026__replicate`.
- Stability: `artifacts/20260912-173656-395831__breast-cancer__tree-shap__seeds-7-42-2026__stability`.

Replica and stability code commit: `953b3cdfa484f2c9d116f8ef26e15c5ff6a47809`.
The accepted seed-42 reference used `3df01fdb09f3c4756bad8ae9ae8975e50cfb2844`.
All runs started clean. Exact config hashes, package snapshots and source hashes remain in
the manifests; the original failed v1 bundles are unchanged and remain diagnostic only.

Maximum test reconstruction residuals across the two backgrounds were RF `1.314e-8` and
XGB `1.181e-7` for seed 7, and RF `1.160e-8` and XGB `1.299e-7` for seed 2026 (rounded
upward). Both replicas passed the unchanged per-class tolerance
`1e-5 + 1e-5 * abs(probability)`, as well as the separate RF equivalence gates. Their recorded
runtimes were 6.74 s and 6.86 s; the stability run took 0.66 s. These are environment-specific
execution records, not cross-method efficiency claims.

On 2026-09-14, all **134 analysis input hashes** were rechecked unchanged. All 12 accepted
conditions were revalidated and all 72 grouped summaries recomputed exactly from their saved
attributions, without writing to the original bundles. Both replica agreement figures and the
stability figure were visually reviewed. The full suite passed **58 tests**; lint, formatting
and whitespace checks passed. Generated artifacts remain local and ignored by Git: a source
commit or push is not a backup of the raw research bundles.

## Next milestone

Prepare a separately versioned **LIME pilot protocol** on the already-frozen seed-42 tree
models. Specify the explained probability and target, training-only neighborhood construction,
feature representation, sampling budget/seeds, local-fit diagnostics, repeatability analysis,
same-target comparison rules and failure reporting before generating test explanations.
Do not silently reuse SHAP's exact-reconstruction gate for an approximate local surrogate.
This is the next design task, not an implemented or accepted LIME experiment.

For the full matrix, obtain the committee's second-dataset choice and exact SOM2IMG author
code/version plus collision/allocation policy. Do not invent author settings or count the
existing collision diagnostic as an evaluated encoding. Those decisions remain the main
scope dependencies while the bounded LIME design can proceed.
