# Second-dataset decision brief

Original review: **recommendation only, 2026-09-17**. No selection or committee approval
was recorded, and no candidate raw data were downloaded, split, fitted or scored in that
review. Sources supplied metadata/documentation, not instructions or expected model quality.

**2026-09-19 update:** Yu subsequently authorized the bounded Spambase readiness audit.
The official source is now downloaded/pinned and passes structural checks: 4,601 rows,
57 features, no detected missing/invalid values. There are 183 repeated-feature groups
involving 577 rows, including three conflicting-label pairs. See
[`SPAMBASE_DATA_CARD.md`](SPAMBASE_DATA_CARD.md) and
[`the audit report`](logs/2026-09-19-spambase-readiness.md). All rows/labels are retained;
no split, preprocessing or model was fitted. Next decision: inclusion and a prospective
group-aware split/label/evaluation policy. The original September 17 rationale and proposed
sequence below are preserved as history; committee approval remains outstanding.

**Subsequent continuation:** Spambase is now the working second dataset. A frozen v1
grouped split kept every row/label and passed overlap/class-count gates, but assigned all
183 repeated-feature groups to validation/test. Scientific review holds v1 as diagnostic-only;
no preprocessing or modeling followed. See
[`the split-policy report`](logs/2026-09-19-spambase-grouped-split.md). The immediate next
step is a prospective allocation-policy correction, not dataset rejection or seed searching.

**V2 correction completed (September 19 local time):** Yu authorized the prospective
amendment. The first frozen v2 allocation passed all structural gates and independent replay;
3,220 / 691 / 690 rows retain the source's repeated-row share at about 12.5% per partition.
See [`the v2 report`](logs/2026-09-19-spambase-grouped-split-v2.md) and
`configs/spambase_split_v2_review.yaml`. V2 is accepted for training-only preflight; v1's
hold remains. No preprocessing/model fitting or committee scope approval is implied.

**2026-09-20 preflight completed:** Yu confirmed continuing with Spambase as an aligned
second dataset. The reviewed v2 training rows now pass scaling, all-57-feature Grid and
CPU compatibility checks, with no held-out transformation or optimizer updates. See
[`the preflight report`](logs/2026-09-20-spambase-training-preflight.md). This fills the
proposal's open second-public-dataset slot; it does not mean Spambase was named there or
has committee approval. Next freeze the first Grid CNN training/validation protocol.

**First model completed (September 20):** the frozen seed-42 Grid CNN achieved 90.88%
validation accuracy and 0.9032 macro F1; group-equal sensitivity was similar. This is
selection-set evidence of learnable signal, not final test performance or interpretability
novelty. No test rows were transformed/scored and no model settings were changed after
the run. See [`the first-training report`](logs/2026-09-20-spambase-grid-first-training.md).

**Grad-CAM update (October 1):** validation-only feature tracing and independent analytic
checks now run on all 691 saved rows, but the first explanation run is **held**, not accepted:
gradient/inference logits exceeded the frozen tolerance on 74 rows despite unchanged
classes. Preserve this diagnostic and resolve numerical equivalence under a versioned
protocol before faithfulness. See [the diagnostic report](logs/2026-10-01-spambase-grad-cam-diagnostic.md).
Test evaluation remains closed; committee scope remains separate.

**V2 repair accepted (October 4):** the fixed activation-boundary Grad-CAM candidate now
matches original inference logits bit-for-bit and passes independent map/feature checks.
The held v1 output stays unchanged; no retraining or tolerance relaxation. This accepts
validation tracing, not faithfulness or a full-matrix condition. Next a frozen validation
feature-deletion protocol. See [the v2 report](logs/2026-10-04-spambase-grad-cam-v2.md).

**Validation deletion completed (October 4):** the frozen, independently replayed study
finds replacement-dependent Grad-CAM evidence: top-minus-random logit AOPC −0.1163 for
zero/training mean (primary), +0.0730 for training median. Preserve both without changing
dataset, model or primary setting. See [the deletion report](logs/2026-10-04-spambase-faithfulness.md).
Next the input-level validation comparison, not test opening. This remains aligned second-
dataset development work, not committee approval, established novelty or a completed CNN condition.

## Original recommendation and comparison (September 17)

**Audit Spambase first**, subject to Yu's approval; do not yet freeze it as the second dataset.
My reasoning is to extend the study outside medicine while retaining a binary numeric setting,
enough features for nontrivial top-5/top-10 comparisons, and more cases than the pilot.
This is an investigator recommendation based on suitability, not published model performance.
Multiclass or mixed categorical/continuous expansion would require a separate scope decision.

| Candidate | Source metadata | Assessment for this thesis |
|---|---|---|
| [Spambase, UCI 94](https://archive.ics.uci.edu/dataset/94/spambase) | 4,601 cases; 57 numeric features; spam/non-spam; CC BY 4.0 | Preferred audit candidate; named features and useful feature count, but integrity and sampling semantics need checking |
| [Ionosphere, UCI 52](https://archive.ics.uci.edu/dataset/52/ionosphere) | 351 cases; 34 features; good/bad radar returns; no missing values reported | Smaller fallback; less sample support than our 569-case pilot, with pulse-feature semantics rather than everyday names |
| [Sonar, UCI 151](https://archive.ics.uci.edu/dataset/151/connectionist%2Bbench) | 208 cases; 60 real-valued features; mines/rocks; no missing values reported | Small test groups; ordered frequency-band structure and acquisition-angle dependence need interpretation, not an assumption of independent arbitrary columns |
| [Banknote, UCI 267](https://archive.ics.uci.edu/dataset/267/banknote%2Bauthentication) | 1,372 cases; four real-valued features; authentication classification | Not preferred: current top-5/top-10 metrics cannot apply to four features; changing the primary metrics would weaken direct comparability |

No claim that any candidate improves CNN accuracy or resolves the Grid-versus-SOM question.
The binary recommendation deliberately limits scope; it does not establish multiclass generality.

## Spambase: metadata is not an acceptance audit

The [legacy documentation](https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.DOCUMENTATION)
reports 1,813 spam and 2,788 non-spam cases, no missing entries, and warns that personal/work
vocabulary can identify the collection source. Consequently this would be a historical benchmark,
not evidence about contemporary general-purpose spam filtering. Conversely, the current UCI
landing page says missing values are present, while displayed variable rows say otherwise.
Record the discrepancy and check the actual pinned release; do not silently choose a metadata flag.

The [feature dictionary](https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names)
defines 48 word frequencies, six character frequencies, an average capital-run length and two
integer run-length counts. Class 0 is non-spam and class 1 is spam. The last column is a target,
not an input feature. Preserve original feature order/names, including punctuation.

These properties motivate checks, not claims about arrays we have not inspected. In particular,
zero-heavy frequencies may collapse quartile boundaries, and continuous sampling of integer
counts may yield inputs outside their original measurement semantics. Those are prospective
LIME audit risks, not measured failure rates for this dataset.

## Proposed next audit, after authorization

1. Download the official files into ignored storage, retain exact bytes, source URLs, access
   date, SHA-256 hashes, license and attribution. Parse exactly 57 inputs plus one target;
   validate row counts, ordered names, class codes, finite values and documented ranges.
2. Run a **pre-split integrity audit** of missing/sentinel values and exact duplicate feature
   vectors, including conflicting-label groups. This requires raw-data inspection before any
   test partition exists; disclose it. Do not silently drop, impute, resolve labels or split
   identical vectors across partitions. Report source/group identifiers if supplied; absence
   of exact duplicates would not establish absence of related emails or acquisition effects.
3. Review the audit, then freeze inclusion/grouping and split rules before outcomes. Prefer
   keeping duplicate groups together if present; select the actual group-aware procedure
   prospectively. A nominal 70/15/15 allocation is a proposal, not a promised row count or
   frozen stratified-row split when duplicates/groups require different handling.
4. Fit scaling, any approved imputation and any constant-feature rule on **training only**.
   Keep all source identities and exclusion reasons. Do not change feature selection or layout
   in response to test results. If a constant-column or binning policy needs an amendment,
   review it explicitly and give the second-dataset protocol its own version.
5. Run training-only encoding/explainer compatibility checks and a compute feasibility check.
   Freeze model/explainer settings and background/sample budgets before test evaluation.
   A larger full-training SHAP background can materially increase work; do not extrapolate
   the previous LIME run's inconsistent timing or reduce budgets after looking at test scores.
6. Build the Grid seed-42 checkpoint on the accepted new split, then the tracing/baseline
   pipeline and registered training-seed repeats. The new Grid track can progress while
   authoritative SOM2IMG clarification is pending; keep the encoder-comparison gate separate.

## Local implementation readiness

Checked the existing code without changing scientific settings:

- `GridEncoder` gives 8 x 8 with seven padding pixels for 57 features. For comparison, the
  metadata-only counts give 6 x 6/two padding for 34 and 8 x 8/four padding for 60. Synthetic
  tests verify value preservation, feature ordering and exclusion of padding from tracing.
- `SmallTabularImageCNN` uses adaptive pooling; synthetic 6 x 6 and 8 x 8 forward/backward
  tests exercise the same architecture. These are compatibility tests, not trained results.
- Grid and tabular runners call `load_breast_cancer_splits` directly. Source validation and
  frozen LIME runners lock pilot identities/counts. Add dataset-aware entry points and
  dataset-specific configs; do not relax the old accepted pilot's guards or rewrite its hashes.
- LIME's `make_explainer` hard-codes malignant/benign display names. Parameterize class names
  under a new tested adapter contract; never label spam explanations as medical classes.
- Numeric input compatibility is not semantic validity. Check integer features, quartile
  degeneracy and constant columns using the training reference before any new LIME run.

Decision requested: authorize **Spambase readiness audit only**, or supply the committee's
preferred alternative. Authoritative SOM2IMG code/collision guidance is a separate decision.
