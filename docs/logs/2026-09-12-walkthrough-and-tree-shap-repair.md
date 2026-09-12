# Results walkthrough and accepted TreeSHAP repair

## Are we on track?

**Yes for pilot implementation; the full thesis scope is not yet complete.** The planned
Autumn Grid/tracing sequence is implemented, and the pilot already includes three training
seeds, deletion experiments and tree baselines from later milestones. This is progress
against the implementation sequence, not a guarantee of the defense schedule: the exact
committee calendar and remaining design approvals are not confirmed.

| Milestone and recorded evidence | What it establishes | What it does not establish |
|---|---|---|
| Leakage-safe 397/86/86 split; 30 features in a 5 x 6 Grid | Reproducible named-feature trace-back | Benefit of another encoding |
| Grid CNN across seeds 7, 42, 2026: mean accuracy 91.86%, macro F1 0.9141 | A working, comparatively stable pilot classifier | Predictive superiority: RF accuracy was 94.96%, XGBoost 96.51% |
| Grad-CAM, IG and Occlusion on frozen checkpoints | Three local explanation methods on matched features | Ground-truth feature importance |
| Seed-42 IG–Occlusion median Spearman 0.969; Grad-CAM–IG 0.396 | Methods can disagree substantially | That the agreeing methods are correct |
| Positive top-minus-random deletion intervals at all three seeds | Conditional faithfulness under tested replacement rules | Causal/clinical relevance or realistic missing-feature interventions |
| Cross-seed explanation medians about 0.712–0.746; top-five overlap 0.60 | Rankings remain training-sensitive despite similar predictions | Population-level stability from only three seeds |
| SOM readiness: 30 features occupy 22 pixels under one provisional setting | Collision handling needs an explicit policy | An authoritative SOM2IMG implementation or evaluated CNN |

The earlier evidence is in the Grid, faithfulness, stability, tabular-baseline and SOM logs.
The full matrix still needs a second dataset, authoritative SOM2IMG rendering/collision
rules, LIME design, and committee review. These are the largest scope risks; adding more
Grid experiments alone cannot complete the encoding-comparison research question.

## What we continued in this session

Repaired RF numerical routing in an explanation-only copy, without retraining or modifying
the fitted model or relaxing tolerances. The versioned derivation and validation requirements
are in `docs/TREE_SHAP_NUMERICAL_REPAIR.md`. Tests cover adjacent float32 boundaries,
unchanged model hashes, and individual signed contributions against exhaustive Shapley
coalition enumeration on a synthetic three-feature forest. The original guarded failure
fixture still fails as intended; both earlier failed bundles remain unchanged.

The new pilot is **technically accepted: all four model/background conditions passed**.

- Run: `artifacts/20260912-171418-618975__breast-cancer__tree-shap__seed-42__pilot`.
- Clean code commit: `3df01fdb09f3c4756bad8ae9ae8975e50cfb2844`.
- Config: `configs/tree_shap_seed_42_v1_1.yaml`; SHA-256
  `40a1ea7d1fd8c9594d74256391dfec635ca717798fe41c72a2a28e7376c3e71f`.
- Same frozen model seed 42, split, targets, backgrounds, ranking metrics and bootstrap
  settings as v1. Source background IDs/weights remain unchanged; RF effective values
  explicitly follow the original predictor's float32 conversion and are saved separately.
- All 6,350 RF internal-node threshold certificates passed; adapted background routing had
  zero mismatches. Copied-model background probabilities differed from the original by at
  most `1.033e-14`, within the separate `1e-12` equivalence tolerance.
- Maximum test reconstruction residual: RF full `1.271e-8`, RF subset `1.226e-8`, XGBoost
  full `2.030e-7`, XGBoost subset `2.028e-7`. All meet the unchanged per-class criterion
  `1e-5 + 1e-5 * abs(probability)`; no post hoc residual redistribution was used.
- Retained 4 arrays of shape `[86, 30, 2]`, 10,320 target-feature rows, 860 paired comparison
  rows, 60 grouped summaries, exact backgrounds, copied thresholds and routing certificates.
  All 21 source-file hashes remained unchanged, and PyTorch was not imported.

## First accepted local TreeSHAP results

RF = Random Forest; XGB = XGBoost. Unless marked 100, tree backgrounds contain all 397
training rows. Rankings use absolute local values and matched predicted targets.

| Comparison | Eligible n / 86 | Mean Spearman [95% interval] | Median Spearman | Median top-5 overlap |
|---|---:|---|---:|---:|
| RF full vs RF 100 | 86 | 0.988 [0.986, 0.990] | 0.991 | 1.00 |
| XGB full vs XGB 100 | 86 | 0.990 [0.988, 0.991] | 0.992 | 1.00 |
| RF vs XGB | 84 | 0.620 [0.597, 0.639] | 0.648 | 0.60 |
| RF 100 vs XGB 100 | 84 | 0.618 [0.597, 0.636] | 0.643 | 0.60 |
| RF vs Grid Grad-CAM | 79 | 0.138 [0.082, 0.191] | 0.168 | 0.20 |
| RF vs Grid IG-zero | 79 | 0.374 [0.319, 0.427] | 0.442 | 0.20 |
| RF vs Grid Occlusion | 79 | 0.380 [0.326, 0.431] | 0.443 | 0.20 |
| XGB vs Grid Grad-CAM | 79 | 0.147 [0.104, 0.189] | 0.143 | 0.20 |
| XGB vs Grid IG-zero | 79 | 0.269 [0.231, 0.309] | 0.272 | 0.40 |
| XGB vs Grid Occlusion | 79 | 0.266 [0.231, 0.304] | 0.280 | 0.20 |

Intervals are 1,000 paired sample bootstrap percentile intervals for the mean, conditional
on fixed models and each comparison's eligible rows. They are not cross-seed uncertainty
or formal tests of differences between comparisons. Two different-target rows were excluded
from each RF–XGB primary comparison; seven from each tree–CNN comparison. No constant-vector
exclusions or exact top-5/top-10 cutoff ties occurred. All rows and exclusion reasons remain
in the bundle. RF–XGB's same-target group includes 81 both-correct and three both-wrong rows;
no interval or generalized claim is made for the three errors. Predicted-class counts are
31 malignant and 53 benign among its 84 eligible rows.

Interpretation: local tree rankings changed little under this one registered background
size/subset check. RF and XGB agreed moderately; their agreement with CNN explanations was
lower on the respective eligible sets. That pattern is useful pilot evidence, not an
isolated effect of model family or explainer choice. The models, explained probability/logit
functions and references differ. Median full top-five overlap across backgrounds does not
mean every sample was unchanged, and one subset does not establish general background
robustness. The observed pattern is not yet an established novel thesis contribution.

`agreement.png` was visually reviewed; it displays mean Spearman intervals, with background
sizes in parentheses. This is an agreement figure, not a faithfulness ranking. Runtime was
6.49 s in this environment; condition-level initialization, warm-up, repeat and test timings
are retained. No cross-hardware efficiency claim is made.

## Verification and next milestone

The full suite passed **49 tests**, with lint, formatting and whitespace checks clean.
An independent artifact audit rechecked raw reconstruction, feature-row uniqueness, background
partition membership, all copied threshold certificates and unchanged source hashes.
Generated run bundles remain local/ignored; source commits are not backups of raw artifacts.

Next: reuse this technically accepted protocol on the already-fitted seeds 7 and 2026, with
no retraining or reselection, and report local TreeSHAP training-seed stability. Keep the same
backgrounds and same-target eligibility rules; do not pool repeated sample/seed pairs as
independent observations. In parallel with that research sequence, obtain committee decisions
on the second dataset and authoritative SOM2IMG policy. LIME remains a separately frozen
protocol, and committee approval is not claimed by this successful technical pilot.
