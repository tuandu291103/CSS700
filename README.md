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
not a changed predictor. Scientific settings and tolerances are unchanged; full pilot
acceptance and seed replication require a new verified run.

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
