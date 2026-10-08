# Three-quarter project plan

**Historical plan:** the professor's October meeting notes expand the minimum scope to three
datasets and require explicit comparative algorithms and quantitative interpretation metrics.
Use [the revised 2026–2027 plan](PROJECT_PLAN_2026_2027_REVISED.md) for prospective planning.
The original schedule and progress record below are preserved; specific new choices remain
subject to advisor review and do not retroactively change accepted experiment protocols.

The proposal estimates approximately 32 working weeks from Autumn 2026 through Spring 2027.
This plan keeps that sequence while adding explicit decision gates and a writing cadence.

## Pre-quarter setup - current checkpoint

- [x] Convert the proposal into a repository plan and reproducibility contract.
- [x] Select Python, PyTorch, Captum, and scikit-learn as the initial stack.
- [x] Scaffold a tested Grid Encoding pilot and living thesis documentation.
- [ ] Confirm scope decisions and unanswered methodology questions with the committee.
- [x] Create the private GitHub repository, push the initial commit, and enable issue tracking.

## Autumn 2026 - foundation and three tracing methods (weeks 1-10)

| Weeks | Milestone | Exit evidence |
|---|---|---|
| 1-2 | Freeze research questions, comparison rules, and expected scope | Committee-reviewed protocol and selected second-dataset criteria |
| 3-4 | Complete the end-to-end Grid pilot | Leakage-safe split, mapping, trained CNN, metrics, and reproducible run bundle |
| 5-6 | Add Grad-CAM tracing | Class-targeted maps, named-feature scores, and tracing tests |
| 7-8 | Add Integrated Gradients tracing | Completeness check, documented baseline, and named-feature scores |
| 9-10 | Add feature-wise Occlusion Sensitivity | Perturbation contract, confidence deltas, runtime, and named-feature scores |

**Autumn gate:** all three methods run on the same Grid model and test samples; failures and
method-specific assumptions are documented.

Progress through 2026-09-06:

- [x] Complete the seed-42 end-to-end Grid pilot, including a clean-commit CNN checkpoint,
  predictive evaluation, and reproducible artifact bundle.
- [x] Generate predicted-class Grad-CAM maps and named-feature scores for every frozen test
  sample, with target, mapping, and checkpoint consistency checks.
- [x] Add signed Integrated Gradients with convergence checks, distinct training-only baseline
  sensitivity, and paired Grad-CAM agreement on the same checkpoint and samples.
- [x] Add signed feature-wise Occlusion Sensitivity with exact zero-replacement masks, runtime
  accounting, and paired three-method agreement on the same checkpoint and samples.
- [x] Complete the Autumn three-method gate for the seed-42 Grid pilot; track cumulative
  faithfulness and multi-seed stability as separate evaluation milestones.
- [x] Complete the first cumulative top/random/bottom deletion-faithfulness pilot with zero and
  training-median replacements plus hierarchical bootstrap intervals. Multi-seed replication
  was completed next to assess training-seed sensitivity.
- [x] Complete a fixed-split Grid replication across pre-registered model seeds 7, 42, and 2026,
  including predictive agreement, same-target explanation stability, and deletion-faithfulness
  stability. Treat the three-seed estimates as descriptive and retain broader replication for the
  controlled experiment matrix.

## Winter 2027 - baselines, SOM2IMG, and controlled runs (weeks 11-20)

| Weeks | Milestone | Exit evidence |
|---|---|---|
| 11-12 | Add Random Forest and XGBoost | Tabular model metrics and clearly separated global/local importance outputs |
| 13-14 | Add SHAP and LIME | Per-sample and aggregate named-feature explanations on frozen test samples |
| 13-16 | Implement and validate SOM2IMG | Training-only layout fit, mapping contract, images, CNN, and trace-back tests |
| 17-20 | Run the expected controlled experiment matrix | Two datasets, two encodings, three seeds, frozen configs, run manifests, CSV outputs |

**Winter gate:** expected-scope raw experiments are complete and reproducible before analysis
choices are tuned to observed results.

Pilot-dataset progress through 2026-09-17:

- [x] Complete fixed-split, three-seed Random Forest and XGBoost predictive baselines with
  validation-only model selection, untouched-test evaluation, and reproducible run artifacts.
- [x] Retain model-native global importance with within-family seed stability while keeping it
  explicitly separate from local, class-targeted explanations.
- [x] Draft the local TreeSHAP target, background, validation and comparison protocol with
  explicit cross-model/output-scale limitations; see `docs/SHAP_COMPARISON_PROTOCOL.md`.
- [x] Freeze and implement provisional local TreeSHAP v1, with source integrity checks,
  training-only backgrounds, independent reconstruction, and isolated runtime tests.
  Committee review remains outstanding; this is not approval of the final thesis design.
- [x] Run and report the seed-42 pilot attempt; diagnose the RF numerical-routing failure
  and add a pre-explanation guard. See `docs/logs/2026-09-12-tree-shap-pilot.md`.
- [x] Repair numerical routing, validate unchanged-model equivalence, and accept a new
  seed-42 TreeSHAP pilot; all four model/background conditions passed. See
  `docs/logs/2026-09-12-walkthrough-and-tree-shap-repair.md`.
- [x] Extend accepted TreeSHAP v1.1 to saved seeds 7 and 2026 and report local training-seed
  stability without treating repeated sample/seed pairs as independent observations.
  All 12 conditions passed; the September 12 runs were audited and reported in
  `docs/logs/2026-09-14-tree-shap-multiseed-stability.md`.
- [x] Freeze provisional LIME v1 separately, including neighborhood construction, local-fit
  diagnostics and explainer-seed repeats; see `docs/LIME_COMPARISON_PROTOCOL.md`.
- [x] Implement and report the training-only LIME preflight: 48 fits, six repeat checks,
  no new test explanations. Technical checks pass, but local approximation is often weak;
  see `docs/logs/2026-09-15-lime-protocol-preflight.md`.
- [x] Investigate LIME locality/fit under a separate prospective training-only design:
  192 fits, 144 paired contrasts, no new model queries or test explanations. Close this
  diagnostic without selecting a replacement; see `docs/logs/2026-09-15-lime-locality-diagnostic.md`.
- [x] Implement and validate the full frozen LIME v1 test runner, named-feature exports and
  paired LIME–SHAP comparisons: 516 fits, 15 comparisons, 92 tests passing. The September 16
  run is audited in `docs/logs/2026-09-17-lime-test-pilot.md`. Retain poor-fit cases; substantial
  ranking agreement does not establish surrogate fidelity. Additional LIME model seeds and
  tree–CNN LIME comparisons remain deferred.
- [x] Review the local SOM2IMG manuscript and complete a reproducible, training-only native-BMU
  collision diagnostic. Document unresolved settings and reject ambiguous feature tracing;
  do not count this diagnostic as an implemented or evaluated SOM2IMG CNN.
- [ ] Resolve the second dataset and authoritative SOM2IMG implementation before expanding the
  full controlled matrix.
- [x] Prepare a source-backed second-dataset recommendation, synthetic larger-grid checks
  and an unsent committee decision note. See `docs/logs/2026-09-17-scope-readiness.md`.
  The recommendation did not itself authorize selection or training.
- [x] Complete Yu's authorized Spambase readiness audit with pinned source bytes and
  independent count/membership verification. See `docs/logs/2026-09-19-spambase-readiness.md`:
  4,601 rows/57 features pass integrity checks; 183 duplicate groups and three conflicting
  groups require explicit policy. All original rows/labels are retained; 114 tests pass.
- [x] Proceed with Spambase as the working second dataset; freeze and inspect one grouped
  split with original labels, preprocessing principles and grouped evaluation policy.
  V1 is **diagnostic-only**: technical gates passed, but all repeated groups were held out.
  See `docs/logs/2026-09-19-spambase-grouped-split.md`; no fitting occurred.
- [x] Prospectively correct the allocation policy and accept a v2 structural audit:
  3,220 / 691 / 690 rows, about 12.5% repeated-group rows per partition, zero group overlap.
  See `docs/logs/2026-09-19-spambase-grouped-split-v2.md`; 150 tests pass and v1 remains held.
- [x] Implement a reviewed-split-aware Spambase loader and complete training-only scaling,
  Grid mapping/padding/tracing probes and bounded CPU compatibility preflight. All 57
  features are retained; no held-out transformation or optimizer steps. See
  `docs/logs/2026-09-20-spambase-training-preflight.md`; 172 tests pass.
- [x] Freeze the first Spambase Grid CNN training/validation protocol and run seed 42:
  epoch 64 selected, stopping at 76; validation accuracy 90.88%, macro F1 0.9032.
  Independent inputs/metrics/checkpoint verification passed; 190 tests pass. See
  `docs/logs/2026-09-20-spambase-grid-first-training.md`. No test scores or explanations
  were produced in that training milestone.
- [x] Freeze and execute the first validation-only Grad-CAM tracing diagnostic: all 691
  rows and 57 features traced; independent analytic maps and records verified. **Held**:
  gradient-logit equivalence failed on 74 rows despite unchanged classes. No post-run
  threshold changes. See `docs/logs/2026-10-01-spambase-grad-cam-diagnostic.md`; 210 tests pass.
- [x] Resolve numerical equivalence under the prospective v2 activation-boundary protocol:
  all saved logits match bit-for-bit, independent analytic maps and all named scores pass.
  Accept **validation tracing only**, preserving held v1; 235 tests pass. See
  `docs/logs/2026-10-04-spambase-grad-cam-v2.md`.
- [x] Freeze and independently verify validation-only Grad-CAM feature deletion: all 691
  rows/629 feature groups, two training-derived replacements, 100 random controls and
  paired group uncertainty. The primary top-minus-random logit AOPC is negative (−0.1163),
  median sensitivity positive (+0.0730); preserve both. See
  `docs/logs/2026-10-04-spambase-faithfulness.md`; 259 tests pass, test remains closed.
- [x] Freeze and independently verify the Occlusion part of the input-level validation
  comparison: 691 rows, 57 features, mean and median replacements, three rank comparisons.
  Mean Spearman with Grad-CAM is −0.0777 / +0.0071; replacement also changes Occlusion
  rankings (mean–median rho 0.3192). All k=1 deletion identities match; 284 tests pass.
  See `docs/logs/2026-10-05-spambase-occlusion.md`. No IG or test evaluation.
- [ ] Freeze IG input-gradient numerical preflight, then the remaining explanation/seed
  protocol before test evaluation.

Next bounded milestone: the Spambase IG input-gradient numerical preflight using the accepted
seed-42 checkpoint. The retained full-gradient discrepancy is unresolved for IG; do not reuse
the Grad-CAM activation-boundary repair as an input gradient. Freeze predictor identity,
gradient checks, training-only references, completeness and compute limits before explanations.
Do not select the median reference merely because the deletion sensitivity is favorable.
Resolve committee dataset scope and authoritative SOM2IMG implementation/rendering/tracing
settings separately; freeze the matched encoder comparison before SOM
fitting or comparative CNN experiments. Validate mapping and one matched training condition
before expanding the matrix. The LIME test pilot and its earlier training diagnostic
are closed: preserve v1 and all poor-fit cases; no additional tuning sweep is planned.
Additional pilot comparisons do not replace these full-matrix decisions. Raw artifact backup
also remains separate from Git source synchronization because generated run bundles are ignored.
The LIME run's performance-counter/UTC timestamp discrepancy prevents an efficiency claim;
cross-method timing still needs a dedicated verified protocol.

Immediate proposal: retain the faithfulness and `configs/spambase_occlusion_v1_review.yaml`
reviews and their exact checkpoint/map/run identities. Continue planned input-level work
without conflating tracing or replacement-dependent deletion with causal importance. Keep model settings and
planned seeds unchanged; no validation tuning sweep. The split v1 and separate Grad-CAM v1
both remain diagnostic-only. A second-dataset Grid track can progress independently while the SOM2IMG source/collision
request is resolved. Neither audit success nor an unsent committee draft counts as approval.

On-track check (2026-10-05): the LIME pilot, Spambase integrity, v2 split correction and
training-only preprocessing/Grid/CPU preflight and first seed-42 training/validation are
complete. One frozen scaler was reused; no Spambase test data were transformed or scored.
Grad-CAM v2 now has exact inference-logit identity and independently accepted validation
tracing. A frozen deletion study now supplies conditional faithfulness evidence, with a
negative primary result and positive median sensitivity, not replacement-robust reliability.
Occlusion now provides independently verified single-feature effects and low agreement with
Grad-CAM under both replacements; its median top-ten ranks include zero ties on 162 rows.
Only three of the expected 12 CNN conditions are fully complete (one dataset x Grid x three seeds),
plus one Spambase training/validation checkpoint with accepted tracing, conditional deletion
and Occlusion comparison, awaiting IG, remaining explanations/seeds and final evaluation.
This is not a percentage of thesis completion. The new dataset pipeline and authoritative
SOM2IMG decisions remain major full-scope dependencies; the exact committee/defense calendar
is unconfirmed.

## Spring 2027 - analysis, thesis, and defense (weeks 21-32)

| Weeks | Milestone | Exit evidence |
|---|---|---|
| 21-24 | Analyze explanation quality and begin full draft | Faithfulness, agreement, stability, runtime, subgroup results, uncertainty estimates |
| 25-27 | Freeze final figures and tables | Reviewed, script-generated, captioned thesis assets |
| 28-30 | Complete thesis v2 and final revision | Introduction through limitations with evidence links and committee feedback resolved |
| 31-32 | Defense and submission | Final thesis, slides, demo, reproducibility package, optional paper decision |

## Standing weekly cadence

- **Start of week:** select one milestone outcome and write its acceptance check.
- **During work:** use versioned configs and append-only run directories.
- **End of experiment:** complete a log, interpret uncertainty, and update the thesis outline.
- **End of week:** commit the research narrative as well as code; record blockers and decisions.
- **Committee cadence:** prepare a one-page update with completed evidence, next decision, and risk.

## Scope guardrails

Do not begin IGTD or broad robustness sweeps until the expected two-dataset, Grid + SOM2IMG,
three-seed matrix is complete. If SOM2IMG slips, preserve the Grid study, document the reduced
scope early, and prioritize trustworthy analysis over the number of methods.
