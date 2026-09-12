# TreeSHAP numerical-fidelity repair: provisional v1.1

Frozen for a new seed-42 attempt on 2026-09-12, before inspecting repaired test explanations.
This is an implementation repair, not committee approval or a change in scientific scope.
The original v1 config and both failed bundles are retained. Executable settings:
`configs/tree_shap_seed_42_v1_1.yaml`. A test requires all original settings to remain equal
except the protocol version and explicit RF adapter selection.

## Decision and equivalence argument

Use SHAP's documented [custom tree representation](https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Example%20of%20loading%20a%20custom%20tree%20model%20into%20SHAP.html)
for a **copy** of the fitted RF trees. Keep tree structure, normalized class leaf values and
forest averaging unchanged. Do not modify the estimator, saved model, SHAP installation or
XGBoost adapter. The copy exists only to explain the original predictor's probability output.

The original RF's finite inputs are converted coordinate-wise to float32 before prediction,
as specified in the [RF prediction contract](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
Its stored split threshold `t` is float64. For each internal node, define `q` as the largest
finite float32 value no greater than `t`. Compute nearest float32 first and step down one
representable value only if nearest rounding went upward. Require the certificate:

```text
q <= t < nextafter_float32(q, +infinity)
```

For every finite float32 value `z`, `z <= t` if and only if `z <= q`. Thus the copied tree
has identical branching for the predictor's representable inputs; the argument applies to
all internal nodes, not just observed training paths. Since `q` is representable, SHAP's
independent routine cannot change that threshold when narrowing it to float32. The original
failure mechanism is recorded in the [v0.52.0 source](https://github.com/shap/shap/blob/v0.52.0/shap/cext/tree_shap.h)
and the earlier session log.

Both RF query and background values use the original predictor's float32 conversion. Keep
the raw standardized background separately; retain the effective values passed to SHAP.
Coordinate-wise conversion commutes with feature replacement: converting a mixture of query
and background features equals mixing their converted coordinates. This preserves the
original model's empirical-background explanatory question rather than choosing new rows
or a new imputation rule. XGBoost continues to use the unchanged v1 path and original
background representation. The shared source IDs/weights remain identical across families.

This is routing equivalence, not a claim of bit-exact SHAP arithmetic: native leaf/contribution
calculations still have floating-point error. Keep independent probability reconstruction at
the original `1e-5 + 1e-5 * abs(probability)` tolerance; do not adjust or redistribute residuals.
Missing/nonfinite inputs and thresholds outside finite float32 range are outside this repair.

## Required validation and retained evidence

- Preserve the native-routing failure fixture: the old guarded adapter must still reject it.
- Check adjacent representable values around positive, negative, zero and tiny thresholds,
  including midpoint cases where nearest rounding fails; reject unsupported inputs.
- For a synthetic three-feature forest, enumerate every coalition against the original
  predictor. Compare individual signed SHAP values for both classes, not merely their sums,
  and verify copied-model probabilities on all enumerated query/background mixtures.
- Verify the original fitted synthetic model hash before/after explaining it.
- For the real forest, certify every internal threshold, require zero adapted background
  node-comparison mismatches and retain the original failed-routing audit alongside the repair.
- Independently compare copied-model probabilities with the original model on the background
  and every explained training/test row (`atol=1e-12`, `rtol=0`). These checks supplement the
  original raw-model reload, smoke/repeat, base-value and reconstruction checks.
- Commit code/config before generating repaired test explanations. Use a new append-only
  bundle; save effective backgrounds, source IDs, copied thresholds, certificates, signed
  outputs, residuals and all previous provenance. Never overwrite the failed attempts.

All model seeds, background rows/weights, class targets, rank metrics, bootstrap settings,
output units and reconstruction tolerances remain unchanged. No changes are selected using
rank agreement. Run seed 42 only; extend to seeds 7 and 2026 after a technically accepted
pilot and its report. Cross-model agreement remains descriptive, not faithfulness or an
isolated explainer effect. Second-dataset, SOM2IMG and LIME decisions remain separate.
