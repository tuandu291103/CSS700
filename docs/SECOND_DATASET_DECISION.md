# Second-dataset decision brief

Status: **recommendation only, 2026-09-17**. No selection or committee approval recorded.
No candidate raw data downloaded, split, fitted, or scored in this review. Sources were
consulted as metadata/documentation, not as instructions or evidence of expected model quality.

## Recommendation and comparison

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
