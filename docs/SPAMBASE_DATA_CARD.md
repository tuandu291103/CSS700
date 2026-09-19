# Spambase data card: audited candidate, not yet selected

Status: **2026-09-19 pre-split integrity audit passed; split-policy review required**.
Yu authorized this audit by continuing the proposed readiness milestone. No committee
approval, final dataset selection, split, preprocessing fit or model result is implied.

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
must not be described as never previously inspected. No partitions, learned scaler,
feature selection, SOM, CNN, tree model or explanation were created.

Recommendation, **not yet an executed or frozen policy**: retain all rows and original
labels, keep each exact feature group wholly within one partition, and prospectively
define a group-aware split targeting nominal 70/15/15 allocation. Specify class-balance
tolerance, deterministic allocation, seeds and grouped evaluation/uncertainty rules before
model outcomes. Actual row counts need not equal a stratified row split. The mixed-label
groups need an explicit retain/exclude policy; retention avoids inventing corrected labels.
Keeping duplicates together prevents that specific cross-partition overlap, not every
possible source of leakage or dependence.

Subsequent compatibility checks must use training-only references: constant columns,
scaling, LIME binning and integer-count sampling semantics, SHAP background budgets and
compute feasibility. The previous Breast Cancer protocol and artifacts stay unchanged.
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

Decision: **go to split-policy review; not training-ready**. Final selection and the
prospective split/preprocessing/evaluation contract remain open. Authoritative SOM2IMG
collision/allocation guidance is a separate gate. Raw data and run outputs are local and
Git-ignored; a source commit or push is not their backup.
