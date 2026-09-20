# First Spambase Grid CNN: training/validation v1

Prospective freeze: **2026-09-20**, before any Spambase model training or validation
prediction. Commit this protocol, config, code and synthetic tests before execution.
Question: can the unchanged pilot CNN learn a useful signal on the second dataset
under the accepted group-disjoint split? This is not a tuning search or final test result.

## Inputs and fixed settings

- Require `configs/spambase_preflight_v1_review.yaml` and all pinned source/split/preflight
  identities. V1 split remains diagnostic-only. No resplit, row removal or label repair.
- Reuse the accepted scaler and 3,220 training images verbatim; do not fit another scaler.
  Transform only the 691 validation rows using those saved training means/scales. Preserve
  all 57 features, original ordering, 8×8 layout and seven zero-padding pixels. Never create
  test tensors. Whole-cohort parsing for integrity is disclosed, not prediction/test scoring.
- Unchanged `SmallTabularImageCNN`: 1→16→32 convolution channels, 3×3 kernels, padding one,
  ReLU, global average pooling and a two-logit linear head; 4,866 parameters. No dropout,
  augmentation, class weighting, oversampling, feature clipping or grid reordering.
- Training seed **42** for this first condition; planned matched model seeds remain
  **7, 42, 2026**, independent of split seed 42. Do not select a favorable seed. This
  execution authorizes only seed 42; later seeds use the same frozen method.
- Match the existing Breast Cancer pilot: Adam, learning rate **0.001**, weight decay
  **0.0001**, cross-entropy, batch size **32**, at most **100 epochs**, patience **12**.
  Each original training row has weight one, including repeated groups.
- CPU, one Torch thread, deterministic algorithms, seeded shuffled training batches and
  ordered validation batches, zero data-loader workers. Restore CPU RNG/thread/determinism
  settings after training. No accelerator, external service or background job.
- A **600-second** training wall-time cap is checked between batches and after validation;
  an individual operation may finish beyond the deadline. Max 10,100 optimizer steps for
  3,220 rows. Timeout is a retained failed attempt, not authorization for a larger budget.

## Selection, diagnostics and test boundary

Select the highest **row-weighted validation macro F1**, improvement greater than `1e-12`;
retain the earliest checkpoint on ties. Stop after 12 consecutive non-improving epochs
or 100 total epochs. Persist each completed epoch's history and each new best checkpoint.
Do not use validation loss, group-weighted scores or test predictions to choose a checkpoint.

Report validation cross-entropy, accuracy, macro F1, predicted-class counts and confusion
matrix (rows=true, columns=predicted; 0 non-spam, 1 spam). Compare with the majority class
chosen on training rows only. Also report validation accuracy/macro F1/confusion matrix
with weight `1 / exact-feature-group size` per row; each intact group totals one. These
sensitivity scores never affect selection. Conflicting labels remain unchanged. No confidence
intervals are produced here; future uncertainty must resample whole groups with paired
methods and a separately frozen budget.

Validation scores are optimistic because validation selects the checkpoint. They are not
held-out test performance, evidence of CNN superiority or proof of faithful explanations.
Report weak/collapsed results instead of silently tuning. Beating the majority baseline is
a descriptive diagnostic, not a gate that discards unfavorable runs. Seeds and preprocessing
will not be changed because of these scores.

**Test gate stays closed:** no 690-row test transformation, model prediction, performance
metric or explanation. No Grad-CAM/IG/Occlusion/SHAP/LIME is computed in this milestone.
Before test use, freeze model/explainer/comparison/uncertainty budgets and accept the checkpoint
through a separately recorded review. Training completion alone never opens that gate.

## Technical acceptance and outputs

Require clean execution code/config; affirmative, unchanged pins; finite training gradients,
history, checkpoint and predictions; probability sums within `1e-6`; actual parameter updates;
budget-consistent epoch/step counts; exact selection-rule replay; and exact validation logits
after safe checkpoint reload using the same batch/thread settings. Retain all failures and
partial completed-epoch history. No automatic retries or replacements.

The append-only bundle contains exact config, provenance/package/timing manifest, only
training/validation arrays and IDs with frozen scaler/mask, feature map, incremental history,
best checkpoint, ordered validation predictions and summary. Git ignores these outputs;
source commits are not an artifact backup. This separate runner leaves Breast Cancer's
existing end-to-end training/test code and results unchanged.

Synthetic tests cover full review-chain loading, no scaler refit or test tensors, tamper
rejection, deterministic training, tie/patience behavior, runtime-state restoration,
timeout/failure preservation, checkpoint reload and group-weighted metrics. Add them to
the Torch CI job. After the real run, independently verify saved inputs, selection/stop
rule, metrics, majority/group sensitivity and checkpoint inference, then write the session
report and separate acceptance pins. A trained-but-test-closed condition must not be counted
as an additional completed full-matrix CNN condition.

```bash
make train-spambase-grid PYTHON=.venv/bin/python
```
