# Spambase data card: training accepted, Grad-CAM held, test gate closed

Status: **2026-10-01 seed-42 Grid training/validation accepted on the reviewed v2 split
and frozen preflight scaler; first Grad-CAM run held for numerical equivalence;
test evaluation closed; split v1 remains diagnostic-only**.
Yu authorized the prospective correction after
the first frozen allocation concentrated every repeated-feature group in validation/test.
V2 balances group structure under unchanged source, inclusion, labels and seed. No committee
approval or final test result is implied. One scaler was fitted on training rows only and
reused without refitting for the first CNN.

## Source, attribution and retained version

- [UCI Spambase, dataset 94](https://archive.ics.uci.edu/dataset/94/spambase), accessed
  September 19, 2026. Citation: Hopkins, M., Reeber, E., Forman, G., and Suermondt, J.
  (1999), *Spambase*, UCI Machine Learning Repository,
  [DOI 10.24432/C53G6X](https://doi.org/10.24432/C53G6X).
- The UCI landing page lists **CC BY 4.0**. Preserve attribution, source/version and a
  description of transformations with any permitted redistribution. No dataset uploaded
  or redistributed in this session.
- Downloaded the [official archive](https://archive.ics.uci.edu/static/public/94/spambase.zip),
  retaining the archive and exact `spambase.data`, `spambase.names` and
  `spambase.DOCUMENTATION` member bytes. Response was HTTP 200 without a redirect;
  Content-Type, ETag and Last-Modified were unavailable in the captured response.
- Raw directory: `data/raw/20260919-090143-663503__spambase__uci-94/`.
  Full SHA-256 pins, including the acquisition receipt, are in
  [`configs/spambase_source_v1.yaml`](../configs/spambase_source_v1.yaml).
  Data-file SHA-256:
  `b1ef93de71f97714d3d7d4f58fc9f718da7bbc8ac8a150eff2778616a8097b12`.
- Acquisition code: `bbed6d7370bcf3c571a6716f9d5b725a0d0d8ee1`.
  Source-lock/audit commit: `f9c2f55a30f40fe8c404a1439c5534c11002ace2`.
  Code was frozen before acquisition; pins were committed before the real-data audit.
  A failed sandbox-network attempt is also retained, not overwritten.

## Schema and empirical integrity

The pinned release contains **4,601 rows, 57 inputs and one target**. Class 0 means
non-spam (2,788 rows); class 1 means spam (1,813 rows). The ordered dictionary specifies
48 word frequencies, six character frequencies, an average capital-run length, longest
capital run and total capital letters. The final two inputs are integer counts.
See the [official dictionary](https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names).
Names/order, including punctuation, were checked exactly; the target is not an input.

All 4,601 rows have 58 fields. No recognized missing token, nonnumeric or nonfinite value,
invalid target, documented-domain violation or noninteger count was found. All 57 features
vary somewhere in the full release; that is not a guarantee of variation in a future
training partition and is not a feature-selection decision.

The current UCI page's global missingness flag disagrees with the
[legacy documentation](https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.DOCUMENTATION),
which reports no missing entries. Our statement concerns the pinned bytes and declared
checks, not a correction to either metadata source. Unknown finite sentinel codes cannot
be excluded by these checks alone. No imputation was performed.

| Exact numerical duplication measure | Observed |
|---|---:|
| Unique feature vectors, target excluded | 4,207 |
| Feature-vector groups with more than one row | 183 |
| All rows belonging to those duplicate groups | 577 |
| Redundant feature rows beyond one per group | 394 |
| Unique fully labeled rows, target included | 4,210 |
| Redundant fully labeled rows beyond the first | 391 |
| Feature groups containing both target classes | 3 |
| All rows in conflicting-label groups | 6 |
| Largest feature-vector group | 69 |

The three conflicting groups have zero-based source row IDs `[63, 3108]`, `[148, 1985]`
and `[478, 3213]`, each with one row per class. One-based line numbers are respectively
`[64, 3109]`, `[149, 1986]` and `[479, 3214]`. These six rows are included in the 577,
not additional to them. No conflicting label was repaired or removed.

Equality means exact parsed float64 values with signed zeros normalized, not identical
email content. Stable group IDs, feature-vector hashes and all row memberships are saved.
The supplied schema has no message/sender/thread identifier to establish real-world
independence; synthetic row/group IDs cannot supply it. Near-duplicates and related
messages are outside this audit.

## Inclusion, preprocessing and scientific limitations

All original rows, labels and feature values are retained unchanged. Whole-release values
and labels were inspected for pre-split integrity/grouping only; a later test partition
must not be described as never previously inspected. The source audit created no partitions.
The subsequent v1 trial created diagnostic row assignments only; no learned scaler,
feature selection, SOM, CNN, tree model or explanation was fitted.

The frozen v1 policy retains all original rows/labels, including conflicting pairs, and
keeps exact feature groups together with a nominal 70/15/15 row allocation and seed 42.
It also specifies class-balance tolerances, training-only preprocessing principles and
row-primary/group-equal sensitivity evaluation. However, its first allocation of
3,221 / 690 / 690 rows routed all 183 repeated groups into validation/test and left training
with only singleton groups. The review in `configs/spambase_split_v1_review.yaml` explicitly
withholds scientific acceptance. See `SPAMBASE_SPLIT_PROTOCOL.md` and the split session
report for the diagnosis. The subsequent prospective v2 correction is documented in
`SPAMBASE_SPLIT_V2_PROTOCOL.md` and `logs/2026-09-19-spambase-grouped-split-v2.md`.
It keeps the original-label policy and seed 42, with 3,220 / 691 / 690 rows and 2,946 / 629 /
632 feature groups. Repeated-group row shares are 12.55% / 12.59% / 12.46%, versus 12.54%
in the source. Every group stays intact; one unchanged conflicting pair happens to occur
in each partition. Group-size bin differences remain visible in the report, including
no size-6-to-10 group in validation. Group disjointness/balance do not establish independent
messages or population representativeness.

The affirmative review `configs/spambase_split_v2_review.yaml` pins the accepted artifacts
and fingerprint `6d0b7926d502366e3f4dff8e905d1ffaa513f44d086bdf37d4509b314aabd39d`.
Only this reviewed split may feed preprocessing/modeling; do not silently use a
fresh replay, change seeds or promote v1. Original run summaries and all v1 pins are unchanged.

The subsequent preflight fitted StandardScaler on the 3,220 training rows only; all 57
columns varied and none required a numerical scale safeguard. Training images use a
fixed row-major 8×8 Grid with seven zero-padding pixels and explicit float32 conversion.
One CPU forward/backward compatibility pass left weights unchanged; no classifier was
trained, no held-out rows were transformed and no explanations were generated. See
`SPAMBASE_PREFLIGHT_PROTOCOL.md`, `logs/2026-09-20-spambase-training-preflight.md` and the
artifact pins in `configs/spambase_preflight_v1_review.yaml`. Separate NumPy calculations
reproduced the saved scaling exactly. This is not evidence of predictive quality.

The subsequent frozen seed-42 CNN trained on 3,220 rows and selected epoch 64 using only
691 validation rows (stopping after epoch 76). Validation accuracy 90.88% and macro F1
0.9032 exceed the training-derived majority baseline's 60.64% / 0.3775; group-equal
sensitivity is 90.86% / 0.9038. These are selection-set results, not unbiased test estimates.
The 690 test rows were not transformed or scored; no explanations were generated. See
`SPAMBASE_GRID_TRAINING_PROTOCOL.md`, `logs/2026-09-20-spambase-grid-first-training.md`
and `configs/spambase_grid_seed_42_v1_review.yaml`.

The first validation-only Grad-CAM run (October 1) reused those saved inputs and traced all
57 features on 691 rows, excluding seven padding cells. Independent analytic-map and
record checks passed, but gradient-enabled logits failed the frozen numerical tolerance
on 74 rows (max 4.28e-5); exact no-gradient replay and all predicted classes matched.
This run remains **diagnostic-only**, with no post-run tolerance adjustment or retry.
See `logs/2026-10-01-spambase-grad-cam-diagnostic.md` and
`configs/spambase_grad_cam_validation_v1_review.yaml`. Next isolate and resolve numerical
equivalence under a versioned protocol, not immediate faithfulness or test evaluation.

Future checks still require training-only references for LIME binning and integer-count
sampling semantics, SHAP background budgets and bounded model/explainer compute settings.
The previous Breast Cancer protocol and artifacts stay unchanged.
Numeric compatibility does not establish meaningful synthetic explanation neighborhoods.

The source documentation describes a 1999 collection with work/personal vocabulary that
can identify its origin. Treat this as a historical benchmark, not evidence for modern
general-purpose spam filtering. Do not infer personal identity, reconstruct messages or
claim population representativeness. No new email collection or contact with people is
part of this study milestone. Institutional requirements and eventual release scope remain
for Yu/the committee to confirm.

## Evidence and gate

Protocol: [`SPAMBASE_AUDIT_PROTOCOL.md`](SPAMBASE_AUDIT_PROTOCOL.md).
Session report: [`logs/2026-09-19-spambase-readiness.md`](logs/2026-09-19-spambase-readiness.md).
Accepted run: `artifacts/20260919-090210-691352__spambase__readiness-audit/`.
An independent NumPy parser/grouping calculation reproduced the counts and verified all
4,601 row memberships, 57 profiles and six recorded input hashes.

Split follow-ups: the [v1 diagnostic](logs/2026-09-19-spambase-grouped-split.md) and
[accepted v2 correction](logs/2026-09-19-spambase-grouped-split-v2.md).
Preflight follow-up: [training-only compatibility report](logs/2026-09-20-spambase-training-preflight.md).
Training follow-up: [first CNN validation report](logs/2026-09-20-spambase-grid-first-training.md).
Explanation follow-up: [Grad-CAM diagnostic hold](logs/2026-10-01-spambase-grad-cam-diagnostic.md).
Decision: **seed-42 training/validation checkpoint accepted; Grad-CAM v1 numerical hold;
split v1 still held and test evaluation not enabled**.
Committee scope review and authoritative SOM2IMG collision/allocation guidance remain
separate gates. Raw data and run outputs are local and Git-ignored; a source commit or push
is not their backup.
