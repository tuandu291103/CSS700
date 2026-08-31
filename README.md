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
