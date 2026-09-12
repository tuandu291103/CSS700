# Three-quarter project plan

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

Pilot-dataset progress through 2026-09-12:

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
- [ ] Extend accepted TreeSHAP v1.1 to saved seeds 7 and 2026 and report local training-seed
  stability without treating repeated sample/seed pairs as independent observations.
- [ ] Complete LIME design separately.
- [x] Review the local SOM2IMG manuscript and complete a reproducible, training-only native-BMU
  collision diagnostic. Document unresolved settings and reject ambiguous feature tracing;
  do not count this diagnostic as an implemented or evaluated SOM2IMG CNN.
- [ ] Resolve the second dataset and authoritative SOM2IMG implementation before expanding the
  full controlled matrix.

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
