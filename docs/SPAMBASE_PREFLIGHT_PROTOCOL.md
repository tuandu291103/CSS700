# Spambase training-only preflight v1

Prospective protocol, 2026-09-20. Freeze code, configuration and synthetic tests in a
clean commit before the first real-data execution. This follows the accepted v2
split review; it does not authorize a new split or test-set model selection.

## Purpose and scope

Check whether the second dataset can use the thesis's existing Grid/CNN/feature-tracing
interfaces without losing feature identity or fitting preprocessing on held-out rows.
Spambase fills the proposal's open additional-public-classification-dataset slot; it
was not named in the proposal. This milestone is infrastructure validation, not a
new predictive or interpretability finding, and not a committee approval claim.

1. Verify the pinned affirmative v2 review, six split files, original source archive,
   source lock, receipt and readiness-audit exports. Reproduce integrity checks,
   saved row memberships, labels, feature-group separation and evaluation weights.
   Full-cohort values and labels are read here **only for source/split integrity**;
   no allocator is rerun and no preprocessing is fitted at this stage.
2. Copy exactly the accepted training IDs (3,220 rows); return no validation/test
   feature matrices from the loader. Preserve the accepted 691/690 held-out IDs,
   original labels, all 57 named features and all rows, including duplicates.
3. Fit one `StandardScaler` using training rows only (population variance, ddof=0).
   Transform training rows only. No imputation, clipping, logarithms, feature
   selection, balancing or target-dependent transformations. Retain constant
   features with scale one; record any numerical near-constant scale safeguards.
4. Use a fixed 8×8, one-channel, row-major Grid in the official 57-feature order.
   The last seven pixels are zero padding. The occupancy mask uses `true` for
   feature pixels. Float64 standardized values are cast to float32 for CNN input;
   record the cast error, not a claim of lossless float64 storage.
5. Check synthetic signed and absolute tracing probes, including a reversed
   mapping-record order and large padding values. These are **not model-generated
   explanations**. Padding must never acquire an original feature identity.
6. Instantiate the unchanged small CNN on CPU, seed 42, one thread. Use only the
   first 32 accepted training IDs for one forward and cross-entropy backward pass.
   Require finite logits, loss, input and parameter gradients and bitwise-unchanged
   model state. No optimizer, parameter updates, checkpoint, accuracy estimate or
   attribution computation. Restore Torch RNG/thread/determinism state afterward.

## Fixed gates

- Reject missing, altered, unreviewed or rejected v1/v2 inputs; verify hashes again
  after the run. Require a clean code/config execution commit.
- Scaler mean/variance must match independent training NumPy moments at
  `rtol=1e-10, atol=1e-12`. Standardized means must be within `1e-10` of zero;
  nonconstant, non-numerically-guarded feature variances within `1e-10` of one.
- All 57 feature identities map one-to-one, seven padding pixels remain zero,
  all transformed values remain finite, and Grid/tracing values exactly match
  the declared float32 conversion.
- All CPU compatibility flags pass and no weights change. Runtime is descriptive,
  not an efficiency benchmark; a single smoke batch cannot estimate training cost.
- Keep every failure with diagnostics. Do not change thresholds after observing a
  failure without a separately documented protocol amendment.

## Outputs and next gate

An append-only timestamped directory contains the exact config, hash manifest,
training-only arrays and scaler state, feature map, feature checks, CPU check and
summary. These local artifacts are ignored by Git; commits alone do not back them up.
Synthetic tests must exercise held-out perturbation invariance, constant retention,
source/review corruption, mapping/tracing, unchanged model state and preserved failures.

After the first execution, independently check source/output hashes, training IDs
and labels, scaler moments/arrays and mapping with a separate NumPy calculation.
Record acceptance in a tracked review and a brief session report.

A passed preflight advances only to **freezing the first Spambase Grid CNN training
protocol**: fixed seeds/split/model budget, validation-only checkpoint selection,
explicit test-use gate and explanation budgets. It does not enable test evaluation
or prove CNN accuracy, generalization, attribution faithfulness or novelty. The
Breast Cancer pipeline and unresolved SOM2IMG methodology are unchanged.

Command after the prospective commit:

```bash
PYTHONPATH=src .venv/bin/python -m css700.cli preflight-spambase
```
