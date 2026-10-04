# Session summary: Spambase Grad-CAM numerical repair accepted

Date: **2026-10-04** (America/Los_Angeles); implementation began October 3.
Outcome: **v2 accepted for validation-only feature tracing**, after exact inference replay
and an independent analytic audit. The held v1 result stays unchanged. No retraining,
tolerance relaxation, test-set evaluation or explanation-faithfulness claim.

## What we did

- Froze the [v2 protocol](../SPAMBASE_GRAD_CAM_V2_PROTOCOL.md), config, implementation,
  tests and independent audit script in `b0399a2e12599851c93b170ee929ead96d1ebc3e` before
  the real run. The full suite passed **235 tests in 64.58 seconds**; all 25 new synthetic
  checks also passed, including the separate audit-script smoke test. Lint and formatting
  were clean. The final full-suite rerun passed all 235 tests in 67.04 seconds. No real-data
  candidate was tried before the freeze.
- Kept the same epoch-64 checkpoint, saved 691 validation inputs, predicted-class logit
  targets, pre-ReLU second convolution, positive-evidence formula and 57-feature mapping.
- Evaluated the convolutional stem using the original no-gradient inference path, then
  started autograd at its final convolution activation. Grad-CAM needs the derivative
  **from that activation to the output**, not the derivative through earlier convolutions.
  This change is specific to this sequential model; it is not suitable for input gradients.
- Ran one fixed candidate plus four predeclared CPU diagnostic modes. Preserved every
  output in a new directory and checked the fixed example figure visually.

## Numerical result

| Comparison, all 691 validation rows | Maximum logit difference | Values failing v1 tolerance |
|---|---:|---:|
| NNPACK enabled: full-gradient versus no-gradient | 4.28e-5 | 76, affecting 74 rows |
| NNPACK disabled: full-gradient versus no-gradient | 5.72e-6 | 0, but not bitwise identical |
| **V2 activation-boundary versus original saved inference** | **0.0** | **0** |

The backend toggle reduced the mode-dependent discrepancy but did **not** establish a
single offending internal kernel or make the disabled paths bitwise equal. Disabled-backend
outputs also differ from the enabled full-gradient outputs. We did not adopt the disabled
backend as a replacement prediction model. V2 retains the original inference path and
matches **all 1,382 stored logit values bit-for-bit**, with all 691 classes unchanged.
Both v1's original tolerance and v2's stronger exactness check pass for the candidate.

PyTorch's [autograd documentation](https://docs.pytorch.org/docs/stable/notes/autograd)
supports using ordinary no-grad tensors in subsequent gradient computations. Its
[backend controls](https://docs.pytorch.org/docs/stable/backends.html#torch.backends.nnpack)
provide the diagnostic toggle. These references motivated the design; the table is
measured local evidence from Torch 2.13.0 on macOS arm64, not a general library-bug claim.

## Independent verification and what the maps show

The [read-only verifier](../../scripts/verify_spambase_grad_cam_v2.py) reconstructs the CNN
without importing project model or explanation code. It verified **51 input hashes**,
nine manifest-listed output hashes, exact replay of all four diagnostic modes and v2,
all IDs/targets, display normalization, every feature score/per-row record, both aggregate
weightings and the deterministic example selection. A separate saved-array calculation
verified the numerical comparison counts, padding aggregates and v1–v2 changes.

All **691 maps and 39,387 named-feature scores** passed the independent formula check.
The float64 analytic reference uses inference-path activations and
`W[target,k] * 1[A>0] / 64` for their gradient. Maximum raw-map error was **5.46e-7**, with
zero failing cells at the unchanged `rtol=1e-5, atol=1e-7` (both terms define the bound).
Weights, source files and feature identities were unchanged; no parameter gradients or
hooks were left behind.

Relative to held v1, raw maps changed by at most **1.67e-5** and positive top-five
feature membership changed on **0/691 rows**. This is limited numerical sensitivity
evidence, not proof that those features are causally important or that rankings are stable
under retraining/perturbation.

Padding carried positive spatial signal in **685/691 maps**: mean fraction **10.75%**,
median 8.32%, maximum 49.37%. Its seven cells are excluded from all feature scores/ranks.
No maps were zero; all rows had at least ten positive feature scores, with no positive
top-5/top-10 boundary ties. These are spatial diagnostics, not logit-completeness identities
or explanation-quality results. The unchanged four example IDs are 1824, 6, 2014 and 89.

## Evidence, scope and next step

Run: `artifacts/20261004-080030-502532__spambase__grid__seed-42__validation-grad-cam-v2/`.
Started at approximately 01:00:30 PDT from clean frozen code; reported whole-run time was
0.900 seconds. No comparative speed claim is warranted. The ten output pins and separate
acceptance decision are in `configs/spambase_grad_cam_validation_v2_review.yaml`.
The v1 negative review, outputs and original tolerance remain unchanged.

To independently verify the retained run without generating a new experiment:

```bash
.venv/bin/python scripts/verify_spambase_grad_cam_v2.py \
  artifacts/20261004-080030-502532__spambase__grid__seed-42__validation-grad-cam-v2
```

We are advancing the proposal's reliable named-feature tracing goal. This fixes a
reproducibility obstacle; it is **not yet a new finding about explanation faithfulness**.
The matrix remains **3/12 full CNN conditions**, plus Spambase seed-42 training/validation
and accepted validation tracing. All **690 test rows remain untransformed, unscored and
unexplained**; full-source reads in this milestone were hash-only provenance checks.
The defense calendar and authoritative SOM2IMG choices remain separate open dependencies.

Next: freeze a **validation-only feature-deletion faithfulness protocol** for this exact
accepted model/map bundle, with fixed replacement values, budgets, random/reverse controls,
padding exclusion and group-aware summaries. Establish numerical identity for the chosen
evaluation path; do not assume this Grad-CAM-specific repair solves input-gradient methods.
Then add the planned input-level explanation comparison and seed/test protocol. No test
opening, new seeds or faithfulness experiment was performed in this session.

GitHub delivery scope is source/config/tests and this report; confirm delivery from the
session handoff. Generated raw data, models and run bundles remain local/Git-ignored,
so a source push does not back them up. Original Breast Cancer results are unchanged.
