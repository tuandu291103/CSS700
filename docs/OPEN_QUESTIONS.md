# Open questions for Yu and the committee

These do not block the initial Grid pilot, but they should be resolved before the related gate.

## Highest priority

1. **SOM2IMG implementation:** The local six-page manuscript has now been reviewed. Can you share
   the exact author code/version and recommended hyperparameters from W. D. Selke / the committee?
   The manuscript does not specify collision handling or the full training settings. See
   `docs/SOM2IMG_READINESS.md` and `docs/logs/2026-09-06-som-mapping-readiness.md`.
2. **Second dataset:** Should the second dataset test multiclass behavior, mixed feature types,
   or simply replicate the numerical binary setting? This choice changes preprocessing,
   attribution targets, and the strength of the conclusions.
   **2026-09-19 update:** Yu continued with Spambase as the working second dataset after
   its source audit. V1 remains diagnostic-only because it concentrated repeated groups in
   held-out partitions. Yu then authorized a prospective v2 correction; that split is now
   independently verified and accepted for training-only preflight: 3,220 / 691 / 690 rows,
   about 12.5% repeated-group rows per partition and unchanged original labels/seed.
   No preprocessing or model was fitted at that split checkpoint. See `docs/SPAMBASE_DATA_CARD.md`,
   `docs/logs/2026-09-19-spambase-grouped-split-v2.md` and
   `configs/spambase_split_v2_review.yaml`. Committee scope review is still pending.
   `docs/SECOND_DATASET_DECISION.md` preserves alternatives and scope rationale. The
   committee request in `docs/COMMITTEE_CHECKIN_2026-09-17.md` remains unsent.
   **2026-09-20:** Yu confirmed continuing with Spambase if aligned with the proposal.
   Training-only scaling, all-57-feature Grid mapping and CPU compatibility now pass;
   no held-out transformation, model updates or predictions were evaluated. See
   `docs/logs/2026-09-20-spambase-training-preflight.md`.
   **Subsequent first-model checkpoint:** seed-42 Grid training/validation now passes:
   90.88% validation accuracy, 0.9032 macro F1, epoch 64 selected. Test rows remain
   untransformed/unscored. See `docs/logs/2026-09-20-spambase-grid-first-training.md`.
   **2026-10-01:** validation-only Grad-CAM is implemented and independently checked, but
   its first real run is held: gradient/inference logits exceed the frozen tolerance on
   74 rows, with unchanged classes. Next a versioned numerical-equivalence diagnostic/repair,
   before accepted explanations or faithfulness. Test rows remain unscored; committee
   approval is still not implied. See `docs/logs/2026-10-01-spambase-grad-cam-diagnostic.md`.
3. **Compute:** Will experiments run on Apple Silicon, an NVIDIA GPU, a university server, or a
   cloud notebook? Determinism, package installation, and timing comparisons depend on this.
4. **Explanation target:** Is predicted-class attribution the committee's intended primary
   analysis, with true-class attribution added for error analysis?
5. **Feature ranking sign:** Should primary rankings use absolute attribution magnitude while
   retaining signed values for secondary analysis, or use positive support for the target only?

## Design decisions

6. What exact feature replacement should faithfulness and occlusion use in standardized space:
   zero/training mean, a conditional value, or more than one sensitivity setting?
7. Should Random Forest and XGBoost comparisons use built-in global importance only, or TreeSHAP
   for like-for-like local explanations as well?

   **Provisional implementation (2026-09-12):** Built-in importance remains a separate global
   diagnostic. Yu authorized implementing local TreeSHAP v1 with probability output,
   interventional replacement, 397 training-background rows and a fixed 100-row sensitivity.
   The original v1 attempt failed RF numerical routing; its outputs remain diagnostic.
   A routing-equivalent copy in v1.1 passed all four seed-42 model/background conditions;
   see `docs/logs/2026-09-12-walkthrough-and-tree-shap-repair.md` and the separate numerical
   repair protocol. The subsequent seeds 7 and 2026 and separate stability analysis passed
   all 12 conditions; see `docs/logs/2026-09-14-tree-shap-multiseed-stability.md`.
   Committee review of scope/targets remains outstanding. CNN-logit versus tree-probability
   output remains a comparison limitation. LIME's separate v1 protocol is now frozen;
   its training preflight passed technical checks but often had weak approximation. See
   `docs/LIME_COMPARISON_PROTOCOL.md` and `docs/logs/2026-09-15-lime-protocol-preflight.md`.
   Committee review of representation/locality assumptions remains important. The bounded
   locality diagnostic is complete, with no
   replacement setting selected; see `docs/logs/2026-09-15-lime-locality-diagnostic.md`.
   The full original-v1 LIME test pilot is now complete: 516 fits and 15 separate comparisons;
   see `docs/logs/2026-09-17-lime-test-pilot.md`. Substantial ranking agreement coexists with
   weak approximation; no poor-fit cases were removed and no further tuning is planned.
   The second dataset and authoritative SOM2IMG settings remain the highest-priority decisions.
8. What threshold defines adequate CNN performance beyond beating the majority-class baseline?
9. Should sample-level uncertainty use bootstrap intervals, and does the program expect formal
   hypothesis tests or primarily estimation and effect sizes?
10. For SOM feature collisions, what mapping and aggregation behavior is authoritative?

    **Readiness evidence:** A provisional, training-only native-BMU diagnostic assigned 30
    features to 22 pixels, with 13 features in five collision groups. This is not an author-code
    reproduction. One-to-one tracing now rejects shared pixels; the SOM2IMG CNN remains gated
    on an explicit rendering/allocation policy and matched scaling/orientation decisions.

## Project and writing

11. What is the exact Autumn start date and the committee's preferred progress-meeting cadence?
12. Does the program provide a required thesis Word/LaTeX template, citation style, page limit,
    or repository/data-retention policy?
13. Should the GitHub repository remain private through defense? Private is the safe default.
14. May the final code and derived artifacts be released publicly, and under what license?
15. The LM-IGTD citation appears incomplete on the proposal's reference page (it ends with
    “to leverage the potential of co”). What is the intended full bibliographic entry?
