# Session summary: first Spambase Grid CNN training

Date: **2026-09-20** (America/Los_Angeles).
Outcome: **seed-42 training/validation checkpoint accepted; test evaluation stays closed**.
The fixed small CNN learned a useful validation signal on the second dataset. This is
an encouraging first model result, not final test performance or an explanation finding.

## What we did

- Froze the [training protocol](../SPAMBASE_GRID_TRAINING_PROTOCOL.md), config and code in
  `565b39f846911627ef1d83af386c8f38d68d3539` before any real Spambase training. Added 18
  synthetic tests and included the new tests in the Torch CI job. All **190 local tests
  passed in 39.77 seconds**, with clean lint, formatting and whitespace checks.
- Reused the accepted 3,220-row training Grid and scaler unchanged. Transformed the 691
  validation rows with those saved training statistics. Kept all 57 features, original
  labels, duplicates and the 8×8 mapping. No scaler refit or test tensors.
- Ran one seed-42 CPU condition with the unchanged 4,866-parameter CNN, Adam, learning rate
  0.001, weight decay 0.0001, batch size 32, maximum 100 epochs and patience 12.
- Selected **epoch 64** by validation macro F1; stopped at **epoch 76** after exactly 12
  non-improving epochs, with **7,676 optimizer steps**. No setting or seed was changed
  after seeing performance. Saved the checkpoint, epoch history and ordered predictions.

## Validation results — not held-out test estimates

| Model / weighting | Accuracy | Macro F1 |
|---|---:|---:|
| CNN, one vote per original row (primary) | 90.88% | 0.9032 |
| Majority class chosen on training rows (primary) | 60.64% | 0.3775 |
| CNN, equal total weight per exact-feature group | 90.86% | 0.9038 |
| Majority baseline, equal group weights | 59.86% | 0.3744 |

Validation has 419 non-spam and 272 spam rows, representing 629 intact feature groups.
The row-primary confusion matrix is `[[397, 22], [41, 231]]` (rows=true, columns=predicted;
class 0 non-spam, class 1 spam). Thus 628/691 predictions were correct: 22 non-spam rows
were flagged as spam and 41 spam rows were missed. Cross-entropy was **0.26036**.

The similar group-weighted scores are a useful sensitivity result, not proof that messages
are independent or that duplication cannot bias the study. Validation selected the checkpoint,
so these numbers are optimistic selection-set diagnostics. Do not compare them as if they
were the Breast Cancer test results, claim CNN superiority over unrun Spambase tree baselines,
or infer generalization to contemporary email. No explanation quality/novelty claim follows.

## Verification and retained evidence

A separate calculation used NumPy and an independently reconstructed PyTorch architecture,
with **no project model/training imports**. It verified:

- all 31 source pins, seven manifest-listed output hashes, unchanged seven preflight files
  and six diagnostic-v1 files;
- training/validation row IDs, labels, source-feature names, saved-scaler reuse, image values,
  zero padding and the 629 validation groups' `1/group-size` weights;
- all 76 history records, selection flags, earliest-best rule, stopping patience and step count;
- primary/group-equal confusion matrices, accuracy, macro F1 and training-derived majority;
- exact checkpoint inference: **maximum validation-logit difference 0.0**; a separate
  float64 cross-entropy calculation differed by only **6.54e-9**.

Run: `artifacts/20260920-172934-706220__spambase__grid__model-seed-42__train-validation-v1/`.
Started at `2026-09-20T17:29:34.706455+00:00` (10:29:34 PDT), clean code/config.
Separate acceptance and all eight run-file pins are in
`configs/spambase_grid_seed_42_v1_review.yaml`; original artifacts were not edited.
Training took about 10.12 seconds; whole-run timing was 10.57 seconds by performance counter
and 10.54 seconds between manifest timestamps (different measurement boundaries). These
are descriptive measurements, not a comparative efficiency result.

All **690 test rows remain untransformed and unscored**. Full-source values/labels were
reread only for integrity/group checks, as disclosed in earlier protocols. No Spambase
attributions were computed and no test gate was opened.

## Are we on track, and what is next?

Yes, this advances the proposal's second-dataset and feature-tracing goals: we now have a
trained checkpoint whose decisions we can investigate. It is progress toward the thesis,
not a guarantee about the unconfirmed committee/defense calendar. The full matrix remains
**3/12 completed conditions plus one Spambase training/validation condition**; the new
checkpoint has not yet completed final evaluation or explanations.

Next: freeze a **validation-only predicted-class Grad-CAM tracing protocol** for this exact
checkpoint, map its heatmaps back to all 57 feature names, and explicitly exclude padding.
Then expand the frozen explanation/faithfulness and seed-replication plan before opening
test evaluation. Keep the same architecture/settings; do not start a validation tuning sweep.
Authoritative SOM2IMG collision/allocation choices and committee scope remain separate gates.

The Breast Cancer pipeline/results are unchanged. Source and report changes are local commits;
no GitHub push or external communication was performed. Model/data bundles remain Git-ignored,
so committing source code does not back them up.
