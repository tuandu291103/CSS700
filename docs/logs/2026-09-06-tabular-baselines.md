# Experiment log: fixed-split Random Forest and XGBoost baselines

## Question and registered protocol

- **Research question:** How do validation-selected Random Forest and XGBoost classifiers perform
  relative to the Grid CNN when trained and evaluated on the same fixed partitions?
- **Split and seeds:** Reuse data-split seed 42 and model seeds 7, 42, and 2026.
- **Selection:** Fit every candidate on training data only, select the first maximum validation
  macro F1, and evaluate the selected fitted model once on the untouched test set. Do not refit on
  combined training and validation data.
- **Preprocessing:** Reuse the training-only StandardScaler and original 30-feature order from the
  Grid pipeline. Trees do not require scaling; it is retained for exact pipeline parity.
- **Global importance scope:** Retain Random Forest mean decrease in impurity and XGBoost gain
  importance as separate model-native diagnostics. Do not compare raw magnitudes across model
  families or treat them as local, class-targeted explanations.
- **Technical failure criterion:** Wrong split fingerprint, overlapping or incomplete partitions,
  non-finite probabilities, probabilities not summing to one, more or fewer than one selected
  candidate, failed serialized-model reload, malformed global importance, missing model/seed
  conditions, or a dirty worktree. Predictive superiority was recorded as a scientific check,
  not a technical quality gate.

## Reproducibility

- **Run:**
  `artifacts/20260906-210148__breast-cancer__tabular__rf-xgboost__split-seed-42__model-seeds-7-42-2026__baseline`.
- **Commit:** `99df51c59b93a14025a0c5dae8b8b3ce61b718dd`; worktree clean at run start.
- **Config:** `configs/tabular_baselines.yaml`; SHA-256
  `4303cb200f3118b35bf61dbe7500dfc26f6f9c98bb9084c11059a267486ff99e`.
- **Split fingerprint:**
  `79e85d4ceae692441b84f6921174e89879dc7eca4e27e8e6f6744799a67aa7cb`, matching the
  accepted Grid split with 397 training, 86 validation, and 86 test samples.
- **Random Forest search:** 400 trees; maximum depth in {unbounded, 6, 12}; minimum leaf size in
  {1, 3}; square-root feature sampling; six candidates per seed.
- **XGBoost search:** 100 or 300 trees; maximum depth 2 or 4; learning rate 0.03 or 0.10; fixed
  0.9 row and feature subsampling; eight candidates per seed.
- **Determinism:** CPU execution with one model thread, fixed model seeds, ordered candidate grids,
  and exact serialized-model probability reload checks.

Primary evidence is retained in `metrics/validation_search.csv`,
`metrics/predictive_metrics.csv`, `metrics/predictive_summary.csv`,
`metrics/test_predictions.csv`, `metrics/prediction_seed_agreement.csv`,
`metrics/global_feature_importance.csv`, `metrics/global_importance_seed_stability.csv`, and
`metrics/summary.json`. The reviewed diagnostic is
`figures/tabular_baseline_stability.png`.

## Results

### Quality checks

All 13 technical checks passed. The split fingerprint matched the accepted Grid reference;
partitions were disjoint and covered all 569 samples once; the StandardScaler was fit on training
only; every model/seed condition selected exactly one validation candidate; all probabilities and
importance values were finite and valid; and all six serialized models reproduced their test
probabilities exactly. The complete run took approximately 5.32 seconds in this environment.

Both scientific checks also held: every selected model exceeded the majority classifier's 0.6279
accuracy and 0.3857 macro F1. These checks describe the observed result and were not used to tune
the technical gate.

### Predictive performance

| Model | Test accuracy, mean (sample SD) | Test macro F1, mean (sample SD) | Accuracy range |
|---|---:|---:|---:|
| Majority classifier | 0.6279 | 0.3857 | fixed |
| Grid CNN | 0.9186 (0.0116) | 0.9141 (0.0123) | 0.9070-0.9302 |
| Random Forest | 0.9496 (0.0067) | 0.9462 (0.0070) | 0.9419-0.9535 |
| XGBoost | 0.9651 (0.0000) | 0.9624 (0.0000) | 0.9651-0.9651 |

The per-seed results were:

| Seed | Grid accuracy / macro F1 | Random Forest accuracy / macro F1 | XGBoost accuracy / macro F1 |
|---:|---:|---:|---:|
| 7 | 0.9302 / 0.9262 | 0.9535 / 0.9502 | 0.9651 / 0.9624 |
| 42 | 0.9070 / 0.9017 | 0.9419 / 0.9382 | 0.9651 / 0.9624 |
| 2026 | 0.9186 / 0.9144 | 0.9535 / 0.9502 | 0.9651 / 0.9624 |

XGBoost selected 300 trees, depth 2, and learning rate 0.10 at every seed, reaching validation
macro F1 1.000. Random Forest selected unbounded depth and leaf size 1 at every seed; validation
macro F1 ranged from 0.9629 to 0.9751. The perfect XGBoost validation score is conditional on only
86 validation samples and should not be generalized.

Relative to the seed-matched Grid CNN, Random Forest had four or five tree-only correct cases and
one or two Grid-only correct cases per seed. XGBoost had four to six tree-only correct cases and
zero or one Grid-only correct case. These are descriptive paired counts; no post hoc significance
test is reported.

### Prediction and global-importance stability

Random Forest pairwise prediction agreement was 0.9884, 1.0000, and 0.9884 across the three seed
pairs. XGBoost class predictions were identical across all seeds, although its probability-based
ROC AUC and log loss varied slightly.

Random Forest global-importance Spearman correlations ranged from 0.9737 to 0.9778, with complete
top-5 and top-10 overlap for every seed pair. Its five highest mean importances were worst
perimeter, worst area, worst concave points, worst radius, and mean concave points.

XGBoost global-importance Spearman correlations ranged from 0.7152 to 0.8705. Top-five overlap was
complete, while top-ten overlap ranged from 0.70 to 0.80. Its five highest mean gain importances
were worst radius, worst perimeter, worst concave points, worst area, and mean concave points.
The shared top-five set is descriptive only: impurity and gain importance have different meanings,
biases, and scales.

## Interpretation and limitations

The direct tabular models outperformed the current Grid CNN on this fixed Breast Cancer pilot,
with XGBoost strongest and most stable by class prediction. This supports the thesis design:
tabular-to-image transformation does not automatically improve predictive performance, and any
value claimed for the CNN pipeline must come from controlled representation or explanation
findings rather than predictive superiority.

The result is limited to one small, numerical, binary dataset and one fixed split. Only three
model seeds and small, pre-declared search grids were tested. The same test samples are reused
across models, and the descriptive mean/SD values are not population-level estimates. No formal
paired hypothesis test was pre-registered. Mean-decrease-in-impurity importance can favor features
with more split opportunities, gain importance depends on fitted tree structure, and correlated
Breast Cancer features can divide importance arbitrarily. Neither global diagnostic answers the
local predicted-class explanation question, and no result supports clinical use.

## Schedule check and follow-up

The project is on track relative to its implementation sequence. The complete Grid pilot,
three-method explanation gate, deletion faithfulness, model-seed stability, and the pilot-dataset
portion of the planned weeks 11-12 tree baselines are complete and reproducible. The main schedule
risks are now external design decisions rather than unfinished baseline infrastructure.

1. Confirm with the committee whether TreeSHAP is required for like-for-like local tree
   explanations and freeze its target/background protocol before implementation.
2. Freeze the LIME sampling, discretization, distance, feature-selection, and random-seed policy.
3. Resolve the authoritative SOM2IMG implementation and second-dataset criteria before beginning
   the full two-dataset, two-encoding controlled matrix.
