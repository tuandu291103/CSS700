# Session report: SOM2IMG mapping readiness

## What we completed

Reviewed the local six-page SOM2IMG manuscript, including visual checks of its method equations
and evaluation description. The source SHA-256 and implementation gaps are recorded in
`docs/SOM2IMG_READINESS.md`. Added a fixed-settings, training-only native-SOM diagnostic and
strengthened tracing to reject shared pixels, invalid coordinates, duplicate identities and
missing feature indices. Reordered mapping records now trace in original feature-index order.
Existing Grid results are unchanged because their maps are already ordered and one-to-one.

## Question and reproducibility

Can an ordinary native-BMU assignment provide a one-feature-per-pixel map on the existing
Breast Cancer training split? This tests our proposed implementation assumptions, not the
authors' unavailable code. Diagnostic settings were frozen before the first run; no search
for a favorable seed, collision count or model score was performed.

- Run: `artifacts/20260907-003744-449627__breast-cancer__som-bmu__layout-seed-42__audit`.
- Started from clean commit `fd84d36a3ddc50d674665f9c4277041778db9262`.
- Config: `configs/som_mapping_audit.yaml`; SHA-256
  `ee9cea5d528c2c231fcd53237a510703caf2c96f21247a043e1f13ab02cd9177`.
- Split fingerprint:
  `79e85d4ceae692441b84f6921174e89879dc7eca4e27e8e6f6744799a67aa7cb`.
- MiniSom 2.3.6; split and layout seed 42; sigma 1, learning rate 0.5, 1,000 update steps.
- Input: 30 feature vectors, each containing only the 397 standardized training values.
  Held-out feature values and labels are not passed to SOM fitting.
- Manuscript grid equation gives 6 rows by 5 columns; no collision resolution was added.

Local session date is September 6; the artifact timestamp is September 7 UTC. The development
smoke run in `/tmp` failed only the intended clean-worktree check and is not the accepted run.

## Observed result

All seven technical checks passed. Same-seed refitting reproduced the weights and mapping
exactly. The reviewed occupancy figure, all 30 candidate assignments, fitted weights, split
indices, scaler parameters and manifest are retained in the run bundle.

| Mapping measure | Result |
|---|---:|
| Features / available pixels | 30 / 30 |
| Occupied / empty pixels | 22 / 8 |
| Pixels with multiple features | 5 |
| Features involved in collisions | 13 |
| Assignments beyond one per occupied pixel | 8 |

The colliding feature groups were radius/perimeter/area error; mean/worst texture;
mean radius/perimeter/area; worst radius/perimeter/area; and mean/worst smoothness.
Exact groups and zero-based coordinates are in `summary.json` and
`encoding/candidate_bmu_map.csv`; the diagnostic plot is `figures/bmu_occupancy.png`.
Mean training-space quantization error was 1.92116; it is a layout diagnostic, not a
predictive or explanation-quality score.

If these native assignments were rendered by last-write overwrite, eight feature values
would not have independent pixels. We did not render such images, discard values, invent an
aggregation policy or train a CNN. The run deliberately reports `som2img_cnn_ready: false`.

## Interpretation and track check

This is useful new evidence for our implementation planning, not a novel SOM finding and not
evidence that the authors' implementation loses features. Ordinary BMU selection has no
one-to-one constraint. Collision-free capacity is therefore insufficient for traceable
feature-level explanations; a documented rendering and attribution-allocation rule is needed.
One provisional seed, one dataset and StandardScaler rather than the manuscript's MinMax
preprocessing do not support a general claim about SOM2IMG.

The completed Grid three-explainer/three-seed study and tree predictive baselines remain intact.
We are progressing through the planned implementation sequence, but the expected two-dataset,
two-encoding study is not complete. Authoritative SOM2IMG details, the second dataset and the
local SHAP/LIME protocol remain the main decision gates; calendar status requires an agreed
committee schedule.

## Verification and next action

All 39 tests passed, including 15 new mapping/SOM cases. Lint, formatting and diff checks are
clean. Tests cover collisions despite sufficient capacity, exact repeatability, manuscript
orientation, original feature identity, invalid maps, split fingerprint preservation and
held-out-data isolation. The isolation test supplies NaN validation/test values and poisoned
labels and verifies that only training features reach SOM fitting. The SOM audit also imports
without loading PyTorch or XGBoost, preserving the existing native-runtime isolation.

Next: obtain the author code/version and collision policy, then freeze matched scaling,
orientation and model settings before the first SOM2IMG CNN. A committee question is drafted
in the readiness document; no message was sent. A collision-free assignment algorithm would
be a declared methodological adaptation, not an assumed reproduction.
