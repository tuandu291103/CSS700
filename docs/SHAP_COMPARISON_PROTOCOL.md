# Local TreeSHAP comparison protocol

Status: **provisional research protocol v1, frozen for the seed-42 pilot, 2026-09-10**.
Yu authorized continuation from the draft; this does not claim committee approval. Executable
settings are in `configs/tree_shap_seed_42.yaml`. These are investigator choices, not library
requirements. The original draft is retained in commit `80f721d`; accepted results belong in
a separate experiment log, not in this protocol.

**Execution checkpoint, 2026-09-12:** v1 has been implemented, but the pilot is incomplete.
The Random Forest sensitivity condition failed independent reconstruction on training rows.
A subsequent routing audit found numerical split disagreements even for the full background,
so no RF agreement result is promoted. Resolve and revalidate numerical fidelity before
replicating seeds. The settings and tolerances below remain unchanged; no library patch or
background replacement has been accepted. See the execution record in section 4.

## 1. Question and scope

For the existing Breast Cancer models, how stable are local tree explanations, and how do
their feature rankings compare with the traced CNN explanations for the same samples and
target classes? SHAP is a comparator, not ground truth.

Begin with the already-selected Random Forest and XGBoost models at model seed 42. If the
technical checks pass, extend to the existing seeds 7 and 2026 without retraining or reselection.
This is separate from the unresolved SOM2IMG implementation and second-dataset decision.
LIME requires its own sampling/local-fit protocol and is not implemented or frozen here.

Source tree run:
`artifacts/20260906-210148__breast-cancer__tabular__rf-xgboost__split-seed-42__model-seeds-7-42-2026__baseline`.
Its training commit is `99df51c59b93a14025a0c5dae8b8b3ce61b718dd`. Reuse its
`models/{random_forest,xgboost}__seed-{7,42,2026}.joblib`, saved predictions, feature order,
training-only StandardScaler and ordered split IDs. Load only these trusted local artifacts;
record model-file hashes and verify predictions before explaining them.

Use the existing 397 training / 86 validation / 86 test partition with split fingerprint
`79e85d4ceae692441b84f6921174e89879dc7eca4e27e8e6f6744799a67aa7cb`.
All 86 test rows remain eligible, including errors. Never use their labels, attributions or
agreement scores to choose a background, SHAP setting or model. Existing test outcomes are
already known: this is prospective for SHAP, not a newly untouched test-set study.

## 2. Frozen seed-42 settings

| Decision | Primary proposal | Reason / boundary |
|---|---|---|
| Explainer | TreeExplainer on the fitted Random Forest estimator / verified numerical XGBoost booster | Original fitted trees; see compatibility record below |
| Explained function | Class probability, `model_output="probability"` | Explicit output semantics for both tree families |
| Feature removal assumption | `feature_perturbation="interventional"` | Explicit shared background; never rely on `auto` |
| Background | All 397 standardized training rows, equal weight, original split order | Avoid background subsampling randomness in this small pilot |
| Per-sample target | Each frozen model's predicted class | Retain both binary class outputs for verification |
| Ranking | Absolute signed SHAP values; all 30 original features | Preserve signed values and base values separately |
| Pairing | Same sample ID and model seed; primary comparison requires equal predicted classes | Different-target rows retained with exclusion reasons |
| Execution | CPU, isolated tree process, batch size 32, `approximate=False` | No PyTorch import or parameter sweep in the tree process |

The SHAP API supports probability output with interventional perturbation. Raw output has
model-dependent semantics; the installed adapter uses probabilities for Random Forest and
margins for binary XGBoost. Array shape also depends on the model, so a universal axis slice
is unsafe. [TreeExplainer documentation](https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html).

Construct `shap.maskers.Independent(background, max_samples=397)` explicitly and pass it to
TreeExplainer as `data`. Assert that the explainer retains exactly the declared rows, values
and order. Do not assume that passing a 397-row array alone preserves all rows: the masker
can subsample above its default cap of 100. [Independent masker documentation](https://shap.readthedocs.io/en/stable/generated/shap.maskers.Independent.html).

Compatibility reference inspected locally: SHAP 0.52.0, scikit-learn 1.9.0, XGBoost 3.4.1 and
NumPy 2.5.2, all matching `requirements.lock.txt`. Installed `_tree.py` and `_tabular.py`
confirm the relevant adapter, background and check behavior. This source inspection does
not establish runtime compatibility; runtime acceptance requires the checks below. Do not
silently change output space, background, approximation mode or dependencies if it fails.

### Training-only compatibility record (2026-09-10)

The initial smoke test exposed a categorical-capability flag mismatch: the frozen XGBoost
wrapper has `enable_categorical=True`, while its booster has no categorical feature metadata,
every saved `split_type` is zero, and all category arrays are empty. SHAP 0.52.0 rejects that
wrapper flag for interventional explanations even in this numerical case.

The v1 adapter therefore passes the original fitted booster to TreeExplainer after verifying
all trees and category arrays. It does not mutate the predictor, patch SHAP, retrain, change
probability output or change the background. Reconstruction still uses the original wrapper's
`predict_proba`. Actual categorical metadata/splits are rejected. Training-only checks passed
for both models after this explicit compatibility adjustment; tests verify unchanged booster
bytes and that no PyTorch module is imported by the tree process.

## 3. Class and output normalization

Validate the dataset's class labels against each estimator's `classes_`; in this dataset,
class 0 is malignant and class 1 is benign. Never assume that class 1 means malignant.
Use the saved prediction to choose the primary target, after verifying that reloaded model
predictions reproduce it. True labels are metadata for grouping, not the attribution target.

Normalize output to `[sample, feature, class]` with explicit class labels:

- For two-output Random Forest SHAP arrays, validate axes against sample count, 30 feature
  names and two classes before selecting the target slice.
- If binary XGBoost probability SHAP returns only the class-1 output, verify its reconstruction
  against `predict_proba(X)[:, class_1_index]`. Derive the complementary probability explanation
  with `phi_0 = -phi_1` and `base_0 = 1 - base_1`, recording that derivation. If the adapter
  returns both outputs, validate both instead. Fail on an unidentified shape/output contract.
- Require each class's base value plus its 30 signed contributions to reconstruct that class's
  probability. Do not apply a sigmoid separately to individual logit SHAP values.

The complement rule follows mathematically from `p0 = 1 - p1` under the same background
and attribution definition. Therefore the two binary classes have the same absolute SHAP
rankings within a model; signs and base values still differ. This is not evidence of
class-insensitive reasoning and does not generalize to the existing separate CNN class logits.

## 4. Validation before interpretation

Freeze an executable config and its code commit before generating test explanations. First
use synthetic fixtures and the first eight rows of the saved training order for compatibility
checks; do not choose smoke-test examples based on outcomes. Then process all 86 test rows.

For every sample and class, independently calculate
`residual = base_value + sum(signed_phi) - model_probability` and require
`abs(residual) <= 1e-5 + 1e-5 * abs(model_probability)`. Apply the same proposed tolerance to
base-value agreement with the mean model probability on the actual retained background.
These are fixed v1 engineering tolerances, not SHAP guarantees.
The built-in additivity check is not sufficient for this probability-output setting, so the
independent reconstruction check is mandatory. [TreeExplainer validation documentation](https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html).

Other required checks:

- Exact split fingerprint, complete sample coverage, unique feature identities/order and
  matching scaler parameters; no validation/test row enters the background.
- Original and reloaded class predictions match exactly. Raw `predict_proba` values match
  the saved probabilities to `atol=1e-7, rtol=1e-6`; the historical exporter normalized each
  probability row. Record any raw-versus-normalized difference instead of modifying SHAP.
- All values are finite, probabilities are in [0,1] and sum to one within `atol=1e-6`;
  class labels and any binary-complement transformation are verified explicitly.
- Repeating explanations on the declared training smoke-test rows gives equal values within
  `atol=1e-10, rtol=1e-8` with the same model/background. No stochastic-explainer seed claim.
- Tests cover reversed mapping/class axes, class-0 selection, synthetic known contributions,
  a constant predictor, poisoned held-out/background candidates, and failed reconstruction.

Failures stop acceptance of that model/background condition. Retain the diagnostics; do not
delete failing rows, relax tolerances after seeing agreement results, or switch to raw SHAP.
Declare any repair/config change and rerun in a new artifact directory. Poor rank agreement,
negative correlations or constant explanations are scientific observations, not technical failures.

### Numerical-routing failure and strengthened gate (2026-09-12)

The first clean-commit run (`87a7962`) passed the original full-background numerical checks
but failed RF reconstruction for the 100-row background before explaining its test rows.
The maximum training-smoke residual was about `2.5012e-5`, above the unchanged tolerance.

The predictor converts inputs to float32 while retaining stored tree thresholds. This input
conversion is documented in the [Random Forest API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
Installed SHAP 0.52.0 casts query inputs but retains background precision; its interventional
native routine stores thresholds in float32, as shown by the `Node` structure and
`dense_independent` routine in the [versioned SHAP source](https://github.com/shap/shap/blob/v0.52.0/shap/cext/tree_shap.h).
These differences can change branch decisions near thresholds. A training-only audit reproduced
such disagreements. Merely casting the background did not eliminate all discrepancies and
has not been adopted as a repair.

The implementation now rejects an RF condition before explanation if training-background
values yield different node comparisons under the predictor and SHAP routing rules. The audit
checks both background and query roles at every internal node, including nodes a whole row
does not reach, since interventional feature mixtures can change paths. Counts are node/value
comparisons, not counts of wrong predictions. Passing this training-only guard is necessary,
not proof for arbitrary held-out inputs; reconstruction remains independently required.
The guard is explicitly tied to the reviewed SHAP version and is covered by synthetic
near-threshold fixtures. Failed audits are retained as `routing_audit.json`.

Do not promote the first run's RF full-background ranks just because its averaged residual
fell within tolerance. Both runs and their diagnostics remain append-only. Any numerical
adapter repair needs a versioned description, original-model equivalence tests, a clean code
commit and a new pilot before seed replication. No looser tolerance, post hoc SHAP adjustment,
model retraining, or output-space switch is authorized by this failure record.

## 5. Comparisons and interpretation limits

| Comparison | Eligible primary rows | What it measures |
|---|---|---|
| Random Forest SHAP vs XGBoost SHAP | Same sample, seed, predicted class and background | Cross-model explanation agreement, not an explainer-only effect |
| Tree SHAP vs each existing Grid explainer | Same sample, seed and predicted class | Descriptive cross-model/cross-method agreement; output and reference differ |
| Same tree family across model seeds | Same sample and predicted class, fixed background | Model-training sensitivity with explainer settings fixed |
| Same tree model across backgrounds | Same sample and frozen target | Background sensitivity, not model-seed instability |

Use absolute-score Spearman with average ranks for ties; top-5/top-10 intersection count,
overlap fraction and Jaccard. Break exact top-k ties by original feature index, matching the
existing helper, and report the number of cutoff ties. Constant vectors have undefined rank
agreement; mark their Spearman and top-k comparisons as not informative rather than assigning
zero correlation or claiming agreement from arbitrary tied selections. Report valid and
excluded counts, reasons and all source predictions.

Retain per-sample rows and summarize each model/seed pair separately with mean, median and
quartiles. Proposed uncertainty: 1,000 paired sample-ID bootstrap resamples using seed 2026,
percentile 95% intervals for the mean, conditional on the fixed models and eligibility set.
Resample paired rows, not features or seeds; comparisons of contrasts must use common IDs and
the same resamples. Keep three-seed summaries descriptive instead of treating repeated sample
measurements as independent observations. No significance tests or superiority claims.

Show predicted class, both-correct / one-correct / neither-correct groups and their counts.
With shared predicted targets on this binary dataset, the one-correct group is necessarily
empty; it belongs to the different-target diagnostic. Preserve zero counts explicitly.
For groups below ten eligible samples, retain individual values but omit subgroup intervals
and generalized subgroup claims. Different-target cases remain a separately labeled diagnostic,
not part of the primary aggregate.

Existing Grid IG/Occlusion primarily explain class logits, while Grad-CAM is a nonnegative
activation-based map. Tree SHAP probability values, their empirical-background reference and
the fitted tree functions differ. Ranking comparisons do not remove those confounds. Do not
compare raw magnitudes, signed sums or pooled bar heights across these families; do not infer
that SHAP or a CNN explainer is more faithful from agreement alone. A same-model, same-scalar-
output explainer comparison would be a separate design and is not supplied by this pilot.

Aggregate mean absolute SHAP only as a summary of local values, retaining target/group counts.
Keep it distinct from saved impurity/gain importance. Neither SHAP nor the current deletion
experiments establish causal or clinical feature relevance. Interventional replacement can
combine correlated features unrealistically; it is not conditional imputation. The choice
of feature-removal distribution changes the explanatory question. See
[Janzing, Minorics and Bloebaum (2020)](https://proceedings.mlr.press/v108/janzing20a.html).

Faithfulness is deferred to a separately frozen tree probability-deletion protocol. Do not
reuse the Grid logit-AOPC numbers as comparable tree faithfulness scores. The existing
zero/median replacement limitations must remain visible in any later extension.

## 6. Bounded sensitivity, artifacts and next implementation

After the seed-42 primary checks pass, proposed sensitivity uses one common 100-row subset:
take the first 100 positions from `numpy.random.default_rng(2026).permutation(397)`, then sort
those positions to preserve source order. Reuse this identical subset across tree models and
model seeds, with explicit masker `max_samples=100`. Record positions, source IDs and values.
Do not replace the primary 397-row result with a more favorable subset result. This is one
size/subset sensitivity, not an estimate of all possible background variability. If compute
prevents it, disclose noncompletion rather than quietly changing the primary protocol.

Create a new append-only TreeSHAP bundle; never modify prior training/attribution bundles.
Retain source paths, model/config/feature-order hashes, code commit, package versions, exact
background arrays and IDs, split IDs, resolved output semantics, all signed class values,
base values, predictions, residuals, target-selected ranks, comparison/exclusion tables and
class-stratified aggregates. Report initialization and explanation runtime separately with
batch size, CPU/thread settings, one training-only warm-up, and the repeatability-check cost.
Do not equate internal tree-path work with CNN forward/backward-call counts.

Implementation sequence (committee review remains outstanding):

1. Promote the approved choices to a versioned executable config; implement adapters, tests
   and independent probability reconstruction checks without importing the CNN runtime.
2. Commit code/config, run the two seed-42 models, and write a pilot report with all failures
   and undefined comparisons visible. Attempted 2026-09-12; numerical RF routing repair and
   a new accepted pilot remain outstanding. No model retraining is needed.
3. Only after the repaired pilot passes, extend technically accepted settings to seeds 7 and 2026; retain the same
   background, split, feature universe and interpretation boundaries.

Review decisions: confirm local TreeSHAP scope, probability output, full-training background,
same-predicted-target primary comparisons and the bounded sensitivity. LIME, the second
dataset and the authoritative SOM2IMG collision policy remain separate open decisions.
