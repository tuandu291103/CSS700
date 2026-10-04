# CSS700: Feature-Level Interpretability for Tabular-to-Image CNNs

This repository supports Yu Dinh's MS thesis project, **Evaluating Feature-Level
Interpretability Methods for Tabular-to-Image CNN Classification Models**.

The study asks whether Grad-CAM, Integrated Gradients, and Occlusion Sensitivity explanations
can be traced from artificial image representations back to meaningful importance scores for
the original named tabular features. Grid Encoding is the traceable baseline; SOM2IMG is the
primary advanced encoding. SHAP, LIME, Random Forest, and XGBoost provide tabular comparison
points.

## Study at a glance

- **Pilot dataset:** Breast Cancer Wisconsin from scikit-learn.
- **Expected study:** at least two public classification datasets, Grid Encoding and SOM2IMG,
  and at least three random seeds.
- **CNN explanation methods:** Grad-CAM, Integrated Gradients, and feature-wise Occlusion
  Sensitivity.
- **Evaluation:** predictive quality, faithfulness, Spearman rank correlation, top-k overlap,
  stability, computational cost, and prediction-group analysis.
- **Principle:** no explanation method is ground truth; reproducible disagreement is a result.

## Repository map

```text
configs/                 Versioned experiment settings
data/                    Data policy; local data files are ignored
docs/                    Research design, plan, logs, and living thesis outline
notebooks/               Exploration only; production logic belongs in src/
src/css700/              Reusable data, encoding, model, tracing, and evaluation code
tests/                   Unit tests for research-critical invariants
artifacts/               Generated run outputs (ignored)
checkpoints/             Trained weights (ignored)
```

## Quick start

Python 3.12 is the reference environment. Python 3.11 is also tested.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
make doctor
make test
make pilot
make train-grid
```

Install the heavier CNN, explanation, plotting, and notebook stack before model work:

```bash
python -m pip install -e ".[research,dev]"
```

To reproduce the exact Python 3.12 macOS arm64 environment resolved at project creation:

```bash
python -m pip install -r requirements.lock.txt
python -m pip install -e . --no-deps
```

`make pilot` performs the first reproducibility checkpoint: it makes stratified train,
validation, and test splits, fits preprocessing on training data only, creates fixed grid
images, and exports a documented feature-to-pixel mapping under `artifacts/`.

`make train-grid` trains the validation-selected Grid CNN, evaluates its frozen checkpoint,
and writes predictive metrics, predictions, figures, and reproducibility metadata to a new
append-only run directory. Generate predicted-class Grad-CAM outputs for that checkpoint with:

```bash
make grad-cam RUN_DIR=artifacts/<grid-cnn-run-id>
```

Raw maps, named-feature attributions, class-stratified aggregate summaries, method settings,
runtime, and reviewed example figures are added to the same run bundle.

Continue on the same frozen checkpoint with signed Integrated Gradients, convergence checks,
training-only baselines, and paired Grad-CAM agreement metrics:

```bash
make integrated-gradients RUN_DIR=artifacts/<grid-cnn-run-id>
```

Because standardization maps the training mean to zero, the command records zero and training
mean as equivalent baseline names and calculates a distinct training-median sensitivity run.

Complete the three-method Grid checkpoint with feature-wise zero-replacement Occlusion
Sensitivity and paired per-sample agreement tables:

```bash
make occlusion RUN_DIR=artifacts/<grid-cnn-run-id>
```

The command masks exactly one mapped feature pixel at a time, retains signed predicted-class
logit and probability drops, ranks by absolute logit drop, and compares Occlusion with Grad-CAM
and both retained Integrated Gradients baselines on the same frozen samples.

Evaluate whether the three primary rankings affect frozen model behavior in the expected order:

```bash
make faithfulness RUN_DIR=artifacts/<grid-cnn-run-id>
```

This runs cumulative top-, bottom-, and repeated random-feature deletion with standardized-zero
and training-median replacements. It retains sample curves, normalized area-over-perturbation-
curve summaries, paired contrasts, fixed-seed random controls, and hierarchical bootstrap
intervals. Scientific ordering is reported independently of the technical quality gate.

For fixed-split model-training stability, train the registered seed-7 and seed-2026 configs and
run the complete explanation and faithfulness sequence on each. Then compare those runs with the
seed-42 reference:

```bash
make grid-stability RUN_DIRS="artifacts/<seed-7-run> artifacts/<seed-42-run> artifacts/<seed-2026-run>"
```

The stability analysis verifies identical splits, preprocessing, layout, and hyperparameters;
reports predictive and deletion-faithfulness variation; and evaluates predicted-class explanation
stability primarily where each model pair predicts the same target.

Train validation-selected Random Forest and XGBoost baselines on the same fixed split and model
seeds with:

```bash
make train-tabular-baselines
```

The command retains the complete validation search, untouched-test predictions, predictive
metrics, serialized models, seed agreement, and model-native global importance. Random Forest
impurity importance and XGBoost gain importance are explicitly labeled as global diagnostics;
they are not treated as local class-targeted explanations or compared on a shared magnitude scale.

The provisional seed-42 local TreeSHAP protocol is in `docs/SHAP_COMPARISON_PROTOCOL.md`.
Run it from a clean commit with:

```bash
make tree-shap PYTHON=.venv/bin/python
```

It explains the existing tree models' probabilities with full-training and fixed-subset
backgrounds, independently checks reconstruction, and compares same-target feature ranks.
The XGBoost adapter verifies numerical splits before using the unchanged fitted booster.
Raw values and source hashes are retained; failures cannot be promoted to accepted results.
Cross-model/output-scale differences remain explicit, and committee approval is not claimed.

**2026-09-12 checkpoint:** the original v1 pilot was not accepted because the fixed Random
Forest model triggered numerical-routing checks. Its failure evidence is retained in
`docs/logs/2026-09-12-tree-shap-pilot.md`. The command now selects the versioned v1.1 repair
in `docs/TREE_SHAP_NUMERICAL_REPAIR.md`: a copied forest with equivalent float32 branching,
not a changed predictor. Scientific settings and tolerances are unchanged. The repaired
seed-42 run passed all four conditions; see
`docs/logs/2026-09-12-walkthrough-and-tree-shap-repair.md`.

**2026-09-14 checkpoint:** seed-7/2026 replication and the separate three-seed stability
analysis are complete (runs dated 2026-09-12). All 12 model/seed/background conditions passed;
full-background seed-pair median Spearman was 0.950–0.960 for RF and 0.956–0.969 for XGBoost.
These are local ranking-stability results on one fixed split, not explainer-superiority claims.
See the report in `docs/logs/2026-09-14-tree-shap-multiseed-stability.md` for pairwise intervals,
sensitivity results, provenance and limitations. All 58 tests passed at this checkpoint.

The frozen replication sequence is in `docs/TREE_SHAP_REPLICATION_PROTOCOL.md`. Explain the
saved seed-7/2026 models and then compare those new runs with the accepted seed-42 reference:

```bash
make tree-shap-replicas PYTHON=.venv/bin/python
make tree-shap-stability PYTHON=.venv/bin/python RUN_DIRS="artifacts/<seed-7-run> artifacts/20260912-171418-618975__breast-cancer__tree-shap__seed-42__pilot artifacts/<seed-2026-run>"
```

The analyzer revalidates accepted inputs, keeps each seed-pair estimate separate and saves
new append-only outputs. It does not treat repeated sample/seed pairs as independent data.

## LIME protocol, preflight and test pilot

The provisional protocol is frozen in `docs/LIME_COMPARISON_PROTOCOL.md`. Check its
compatibility with the saved seed-42 tree models from a clean commit using:

```bash
make lime-preflight PYTHON=.venv/bin/python
```

**2026-09-15 checkpoint:** all 48 training-anchor fits and six same-seed repeats passed their
technical checks; all 66 tests passed. Local approximation was often weak: median anchor
probability error was 0.081 for RF and 0.302 for XGBoost. See
`docs/logs/2026-09-15-lime-protocol-preflight.md`. No test LIME explanations or LIME–SHAP rank
comparisons were generated at that training-only checkpoint.

The subsequent bounded locality diagnostic is complete: 192 fits, 144 paired contrasts,
all original references reproduced, and all 76 tests passing. Narrower weighting improved
anchor fit without reliably improving common-neighborhood fit; no alternative was selected.
See `docs/logs/2026-09-15-lime-locality-diagnostic.md` for results and the on-track check.
Reproduce it from a clean commit with:

```bash
make lime-locality PYTHON=.venv/bin/python
```

The diagnostic protocol is in `docs/LIME_LOCALITY_DIAGNOSTIC_PROTOCOL.md`. That bounded
investigation is closed; the original v1 settings were retained for the test pilot.

**2026-09-17 checkpoint (run September 16):** the full frozen LIME pilot is complete:
516 verified fits, 15,480 named-feature rows, 15 paired comparisons and **92 passing tests**.
Primary-seed same-model LIME–TreeSHAP median Spearman is 0.792 for RF and 0.805 for XGBoost,
but median anchor probability errors are 0.107 and 0.217 and generated-evaluation R-squared
medians are only 0.209 and 0.239. Ranking agreement is not approximation quality. See
`docs/logs/2026-09-17-lime-test-pilot.md` for separate-seed estimates, fit limitations,
verification and a recorded timing discrepancy. Execute the frozen contract from a clean
commit with:

```bash
make lime-test PYTHON=.venv/bin/python
```

`docs/LIME_TEST_EXECUTION.md` documents acceptance and exports. No further LIME tuning is
planned; next resolve the second dataset and authoritative SOM2IMG choices for matrix expansion.

The second-dataset choice and authoritative SOM2IMG collision/allocation policy remain
necessary for the full study. Generated run bundles are local and ignored by Git;
committing or pushing source code does not back up the raw research artifacts.

**2026-09-19 checkpoint:** the authorized Spambase integrity audit passed: 4,601 rows,
57 features, no detected missing/invalid values, and **114 passing tests**. It found 183
repeated-feature groups involving 577 rows, including three conflicting-label groups
involving six rows. All rows/labels were retained; no split or model was created.
See `docs/SPAMBASE_DATA_CARD.md`, `docs/SPAMBASE_AUDIT_PROTOCOL.md` and the session report
`docs/logs/2026-09-19-spambase-readiness.md`.

**Subsequent split-policy checkpoint:** Spambase is the working second dataset, with all
rows/labels retained. The frozen v1 trial produced 3,221 / 690 / 690 rows and no feature-group
overlap, but scientific review found every repeated-feature group was assigned to validation
or test. **V1 is diagnostic-only, not an accepted training split**, despite passing its
technical gates. See `docs/logs/2026-09-19-spambase-grouped-split.md` and the explicit hold in
`configs/spambase_split_v1_review.yaml`. No preprocessing or model was fitted.
The v1 checkpoint had 131 passing tests; test success did not override its scientific hold.

**V2 correction accepted (September 19 local time):** the separately frozen allocation
produced **3,220 / 691 / 690 rows**, with repeated-group row shares of **12.55% / 12.59% /
12.46%** and zero feature-group overlap. All original rows/labels and split seed 42 are
unchanged. The full suite now has **150 passing tests**. Independent raw-data verification
and exact-arithmetic replay support the affirmative review in
`configs/spambase_split_v2_review.yaml`. V1 remains diagnostic-only. See
`docs/SPAMBASE_SPLIT_V2_PROTOCOL.md` and `docs/logs/2026-09-19-spambase-grouped-split-v2.md`.

**2026-09-20 preflight accepted:** the reviewed-split-aware loader, training-only scaling,
57-feature 8×8 Grid (seven padding pixels), synthetic tracing probes and bounded CPU
forward/backward check passed. The scaler used only the 3,220 training rows; no held-out
rows were transformed and no model weights were updated. Independent NumPy calculations
reproduced the saved standardization exactly. The suite now has **172 passing tests**.
See `docs/SPAMBASE_PREFLIGHT_PROTOCOL.md`, `configs/spambase_preflight_v1_review.yaml`
and `docs/logs/2026-09-20-spambase-training-preflight.md`.

**First Spambase model accepted (September 20):** the fixed seed-42 Grid CNN selected
epoch 64 and stopped at epoch 76. Validation accuracy is **90.88%**, macro F1 **0.9032**;
the training-majority baseline gives 60.64% and 0.3775. Group-equal sensitivity is similar
(90.86%, 0.9038). These are checkpoint-selection validation results, **not final test
performance**. All 690 test rows remain untransformed/unscored. The suite has **190 passing
tests**; an independent model replica reproduces the saved validation logits exactly.
See `docs/SPAMBASE_GRID_TRAINING_PROTOCOL.md`, `configs/spambase_grid_seed_42_v1_review.yaml`
and `docs/logs/2026-09-20-spambase-grid-first-training.md`.

**Grad-CAM diagnostic hold (October 1):** the validation-only implementation traces all
57 features on all 691 saved validation rows, with explicit padding exclusion and independent
analytic map verification. The first real run is **not accepted**: gradient-enabled logits
exceeded the frozen numerical tolerance on 74 rows (maximum difference 4.28e-5), despite
exact no-gradient replay and unchanged predicted classes. No tolerance was relaxed or run
retried. The suite has **210 passing tests**. See `docs/SPAMBASE_GRAD_CAM_PROTOCOL.md`,
`configs/spambase_grad_cam_validation_v1_review.yaml` and
[`the session report`](docs/logs/2026-10-01-spambase-grad-cam-diagnostic.md).

**Grad-CAM v2 accepted (October 4):** starting autograd at the original inference-path
activation resolves the numerical hold for validation tracing. All 1,382 logits now match
saved inference bit-for-bit; 691 maps and 39,387 named-feature scores passed an independent
analytic audit. No retraining or tolerance relaxation; v1 remains held. **235 tests pass**.
See `docs/SPAMBASE_GRAD_CAM_V2_PROTOCOL.md`, `configs/spambase_grad_cam_validation_v2_review.yaml`
and [the v2 session report](docs/logs/2026-10-04-spambase-grad-cam-v2.md). This establishes
numerical correctness/tracing, not explanation faithfulness or causal importance.

Next: freeze validation-only feature-deletion faithfulness for this exact checkpoint and
v2 map bundle, followed by the input-level explanation comparison and remaining seed/test
plan. The matrix remains 3/12 completed conditions plus one Spambase training/validation
condition with accepted validation tracing, not final test results. Committee scope approval and
authoritative SOM2IMG guidance remain separate open decisions. Training replay (new
outputs are not automatically a new accepted source):

```bash
make train-spambase-grid PYTHON=.venv/bin/python
```

To replay the earlier bounded preflight:

```bash
make preflight-spambase PYTHON=.venv/bin/python
```

To reproduce the frozen v2 allocation (a replay is not automatically a new accepted source):

```bash
make split-spambase-v2 PYTHON=.venv/bin/python
```

`docs/SECOND_DATASET_DECISION.md` preserves the original recommendation and audit update;
`docs/COMMITTEE_CHECKIN_2026-09-17.md` remains an unsent scope/SOM2IMG decision request.
To repeat the offline audit from a clean code state using the retained pinned source:

```bash
make audit-spambase PYTHON=.venv/bin/python \
  SOURCE_DIR=data/raw/20260919-090143-663503__spambase__uci-94
```

`make fetch-spambase PYTHON=.venv/bin/python` acquires a new official source bundle, not
an audit or training run. A fresh receipt must be explicitly reviewed/pinned; do not replace
the original source lock or treat a fresh download as the same acquisition.

## SOM2IMG readiness

Before the first SOM2IMG CNN, audit native SOM feature assignments with:

```bash
make audit-som-mapping
```

This is a training-only, fixed-settings collision diagnostic, not an authoritative SOM2IMG
reproduction. It saves all candidate assignments and refuses to mark the encoder ready for a
CNN. See `docs/SOM2IMG_READINESS.md` for the source review, frozen settings, and unresolved
collision/scaling/orientation choices. Feature tracing now rejects shared pixels unless an
explicit allocation policy is implemented and preserves original feature order even when
mapping records are reordered.

## Research workflow

1. Create or select a versioned YAML config.
2. Run from a clean commit and retain the commit SHA, seed, package snapshot, and mapping.
3. Write outputs to a new run directory; never overwrite a completed run.
4. Complete an experiment log using `docs/logs/experiment-log-template.md`.
5. Update the relevant section of `docs/THESIS_OUTLINE.md` while the evidence is fresh.
6. Promote only reviewed, reproducible figures and tables into the thesis.

The three-quarter roadmap is in `docs/PROJECT_PLAN.md`; the controlled comparison and metrics
are specified in `docs/RESEARCH_DESIGN.md` and `docs/EXPERIMENT_PROTOCOL.md`.

## Scope discipline

The minimum, expected, and aspirational scopes from the proposal are tracked explicitly. IGTD,
large robustness sweeps, and publication preparation remain stretch goals until the expected
two-dataset, two-encoding, three-seed study is complete.

The original proposal PDF is intentionally not committed. Before making this repository public
or adding source documents, review privacy, committee, licensing, and unpublished-method details.
