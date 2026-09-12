# TreeSHAP fixed-split replication protocol

Status: provisional research extension, frozen before seed-7/2026 test explanations on
2026-09-12. User-authorized continuation; committee approval is not claimed. This implements
the replication planned in `SHAP_COMPARISON_PROTOCOL.md` and the accepted v1.1 repair.

## Fixed inputs and scope

- Reuse the existing validation-selected RF and XGBoost models for seeds 7 and 2026.
  Do not retrain, reselect, change dependencies or tune the explanation configuration.
- Reuse accepted seed 42 from
  `artifacts/20260912-171418-618975__breast-cancer__tree-shap__seed-42__pilot` rather than
  regenerating it or using either failed v1 attempt.
- New configs are `configs/tree_shap_seed_7_v1_1.yaml` and
  `configs/tree_shap_seed_2026_v1_1.yaml`. Only model seed, matching Grid source path and
  saved-model hashes may differ from the seed-42 config. A regression test enforces this.
- Keep the 397/86/86 split, ordered sample IDs, training-only scaler, 30-feature universe,
  probability-output interventional TreeSHAP, numerical adapters, tolerances and batch size.
  Retain the same source backgrounds: all 397 training rows and the fixed 100-row subset.
  Effective RF background values retain the predictor-equivalent float32 conversion.
- Every replica uses its seed-matched existing Grid explanations for the same descriptive
  within-seed comparisons as the pilot. Original tree/CNN output and reference differences
  remain explicit; no explainer-superiority claim is licensed by rank agreement.

## Acceptance and analysis

Use clean commits and append-only output directories. Keep the original source/hash,
prediction reload, background, RF routing certificate, smoke/repeat, base-value and raw
reconstruction checks. Retain a failed condition rather than discarding samples, switching
methods or changing tolerance. Each seed needs all four model/background conditions accepted.

The stability command takes exactly one accepted run for each seed 7, 42 and 2026. It
revalidates all 12 conditions against their frozen configs and original source files;
checks raw reconstruction, named-feature/class identity and declared backgrounds; and
requires identical scientific settings, package versions, split, feature order and reference
rows across seeds. Duplicate seeds, changed/corrupted inputs and failed bundles are rejected.
The original seed-42 reference is registered explicitly in `configs/tree_shap_stability.yaml`.

For each family and each background, compare seed pairs 7–42, 7–2026 and 42–2026 separately:

1. Retain prediction agreement, different-target cases and three-seed prediction consensus.
2. Compare absolute feature scores only on identical sample IDs with matching predicted
   target classes. Keep constant-vector exclusions, tied cutoffs and all sample-level rows.
3. Report mean/median/quartiles of Spearman and top-5/top-10 overlap/Jaccard, with the same
   predicted-class and correctness subgroups as v1.1. Show excluded and zero counts.
4. Use the unchanged 1,000 paired sample bootstrap resamples, seed 2026, for each pair's
   conditional mean interval. Omit subgroup intervals below ten eligible rows.
5. Do not pool repeated sample/seed-pair rows into an independent sample count or pooled
   confidence interval. Summarize across seed pairs with descriptive ranges, not a claim
   about population variance from three training seeds. No formal cross-family contrast.

Primary stability uses all 397 background rows. Repeat the same analysis separately for
the fixed 100-row background as a bounded sensitivity check, not a replacement chosen for
favorable results. This is training-seed sensitivity of the whole fitted-model/explanation
pipeline, not randomness in a deterministic explainer. It remains one dataset and one split.

## Outputs and next decision

Retain each replica's original full bundle and a separate stability bundle with source
commits/hashes/configs, per-pair named comparisons, prediction agreement/consensus, grouped
summaries, reviewed figure and explicit acceptance status. The session report will summarize
new results alongside limitations and update the living thesis outline.

After this bounded replication, identify the next scope-critical work: second-dataset and
authoritative SOM2IMG decisions, or a separately frozen LIME protocol. Tree-probability
deletion needs its own design; do not reuse CNN logit-AOPC as comparable tree faithfulness.
