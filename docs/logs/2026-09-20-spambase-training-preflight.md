# Session summary: Spambase training-only preflight

Date: **2026-09-20** (America/Los_Angeles).
Outcome: **passed and accepted for freezing the first Grid CNN training protocol**.
This is preprocessing/encoding/CPU compatibility evidence, not a trained model result.

## What we did

- Added a loader requiring the exact affirmative v2 split review and source/output pins.
  It rechecks memberships, labels, feature groups and evaluation weights, never reallocates
  rows, and exposes only training features/labels to preprocessing.
- Froze the [preflight protocol](../SPAMBASE_PREFLIGHT_PROTOCOL.md), config, implementation
  and 22 synthetic tests in commit `d5557859098ca4293763034a50c86164dd6b8726` before the
  first actual preflight. The complete suite passed **172 tests in 39.09 seconds**;
  lint, formatting and whitespace checks passed.
- Executed one real preflight from that clean commit. Fit one StandardScaler on exactly
  3,220 training rows; created training Grid arrays and feature-tracing probes; ran one
  untrained CPU forward/backward pass without an optimizer or checkpoint.
- Independently reloaded raw data with NumPy (no project/scikit-learn imports), selected
  the accepted training IDs, and calculated mean, two-pass population variance, scale
  and transformed values. Verified arrays, labels, original feature names, exact mapping,
  occupancy mask, padding, float32 cast and per-feature diagnostics.
- Verified all 22 input hashes and six manifest-listed output hashes, and confirmed all
  six diagnostic-v1 file hashes remain unchanged. Recorded acceptance separately in
  `configs/spambase_preflight_v1_review.yaml`; original run outputs were not edited.

## Results

| Check | Result |
|---|---|
| Accepted train / validation / test rows | 3,220 / 691 / 690, unchanged |
| Training non-spam / spam rows | 1,951 / 1,269 |
| Training feature matrix | 3,220 × 57, all features retained |
| Constant / numerically guarded training columns | 0 / 0 |
| CNN input | 3,220 × 1 × 8 × 8, float32 |
| Grid mapping | 57 one-to-one feature pixels; seven zero-padding pixels; no collisions |
| Largest absolute standardized training mean | 6.46e-15 |
| Largest standardized variance error from one | 2.18e-13 |
| Saved standardized values vs independent calculation | Maximum absolute difference 0.0 |
| Float64-to-float32 cast | Maximum absolute difference 1.23e-6; not lossless float64 storage |
| Tracing probes | Signed and absolute reconstruction pass; padding excluded |
| CPU compatibility | 32 training rows; 4,866 parameters; finite output/gradients; weights unchanged |
| Held-out rows transformed / optimizer steps / models trained | 0 / 0 / 0 |

The smoke input is the first 32 accepted training IDs, not a representative timing sample.
Recorded preflight runtime was about 1.10 seconds and the CPU check about 0.37 seconds;
these are descriptive execution measurements, **not a throughput or efficiency claim**.
The synthetic tracing probes are not CNN-generated attributions. Full-cohort rows/labels
were reread for integrity verification, so do not describe test rows as never inspected;
they were not used to fit scaling, encode images, select models or score predictions here.

## Evidence and proposal alignment

- Run: `artifacts/20260920-104659-474102__spambase__training-preflight-v1/`.
- Execution began at `2026-09-20T10:46:59.474332+00:00` (03:46:59 PDT).
- Config: `configs/spambase_preflight_v1.yaml`.
- Tracked review: `configs/spambase_preflight_v1_review.yaml`, with all seven run-file pins.
- Reproduction: `make preflight-spambase PYTHON=.venv/bin/python` from a clean checkout
  with pinned local source/audit/split inputs. A replay gets a new directory and is not
  automatically the accepted downstream source.

Spambase implements the proposal's open additional-public-classification-dataset slot;
it was not explicitly named in the original proposal. Its 57 named inputs let us test
the same tabular-to-image explanation-tracing question beyond the medical pilot.
Today's result supports compatibility, not predictive superiority, explanation faithfulness,
novelty, modern spam-filter deployment or committee approval. Breast Cancer code/configs
and results were unchanged. The study still has **3/12 expected CNN conditions completed**
(one dataset × Grid × three model seeds); this is not a thesis-completion percentage.

## Next step

Freeze and implement the first Spambase Grid CNN training protocol using this exact split,
feature map and training-fitted scaling. Predeclare architecture, model seeds, epoch/compute
budget, optimizer, validation-only checkpoint selection and the later test-use gate. Begin
with one bounded training/validation condition, then trace Grad-CAM and expand under the
frozen explanation/evaluation budgets. No need for another split search or LIME tuning sweep.

Separately resolve authoritative SOM2IMG collision/allocation settings and committee scope.
The methods work remains aligned with the proposal, but those dependencies and the incomplete
matrix prevent a claim that the full study is finished or a guaranteed schedule claim.
Generated arrays and raw artifacts remain local/Git-ignored; source commits are not their
backup. No GitHub push or external communication was performed in this session.
