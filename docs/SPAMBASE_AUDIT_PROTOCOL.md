# Spambase pre-split integrity audit v1

Authorized scope: Yu's September 19 continuation of the proposed data-readiness audit.
This is **not committee approval, final dataset selection, splitting or model training**.
Commit the audit implementation and synthetic tests before inspecting the downloaded data.

## Acquisition and pinning

Use the official archive linked by the [UCI dataset page](https://archive.ics.uci.edu/dataset/94/spambase):
`https://archive.ics.uci.edu/static/public/94/spambase.zip`. Save a new ignored raw directory,
retain archive bytes and its three named members without executing their contents, and record
URL, access time, response metadata, exact sizes, SHA-256 hashes and clean source commit.
Reject unexpected archive paths/members, duplicate names and oversized input. Do not use
unverified mirrors. Keep failed attempts in separate directories rather than overwriting them.

The UCI page lists CC BY 4.0 and the citation Hopkins, Reeber, Forman and Suermondt (1999),
Spambase, DOI `10.24432/C53G6X`. Retain attribution and the source link in the data card.
Raw files are ignored by Git; hashes are identity checks, not backups or author signatures.

After acquisition, write and commit a tracked `configs/spambase_source_v1.yaml` with archive,
member and receipt hashes **before the first real audit**. Later audits are offline, verify
the pin/receipt, compare retained members to archive bytes, and recheck inputs at completion.
The source lock pins this acquisition, not a guarantee about future UCI releases.

## Fixed checks and definitions

Read all physical source lines; define `row_id` as zero-based line position and separately
retain one-based line number. Expect 4,601 rows, 57 inputs in the exact official dictionary
order, and one binary target (0 non-spam / 1 spam), with published counts 2,788 / 1,813.
The [source dictionary](https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names)
supplies ordered names and feature types. Match names, punctuation and the source class
declaration; never parse the target as an input.

Retain malformed row/field locations. Check empty fields and the declared missing tokens
`?`, `NA`, `N/A`, `NaN`, `null`, `none` (case insensitive), other nonnumeric/nonfinite values,
invalid labels and documented ranges. The 54 frequency features must lie in [0, 100]; the
three run-length measures must be at least 1; the final two must be integers. These are the
published broad domains, not fitted thresholds or outlier trimming rules. Unexpected values
are findings, not permission to impute or drop rows. Undocumented finite sentinels cannot be
ruled out just by parsing/domain checks.

Export per-feature finite/unparsed counts, minima/maxima, zeros, distinct finite values and
whole-release constant flags for integrity only. Do not use them to fit preprocessing or
choose features. No full-data quartiles, scaler, SOM, LIME neighborhood or model is fitted;
those require a future frozen split and training-only reference.

For fully parsed finite data with binary labels, group by exact **parsed float64 feature
vectors**, excluding the label and treating signed zeros as equal. Numeric representations
such as `0` and `0.0` consequently match. This is operational numeric equality, not proof that
the emails are identical; nearby vectors, shared senders and acquisition dependence are not
covered. Keep stable first-occurrence group IDs, normalized little-endian vector hashes,
every member row/line ID, per-group class counts and conflicting-label flags. If parsing or
labels fail, do not quietly group only a cleaned subset.

Report separately: unique feature vectors; groups with size > 1; all rows in those groups;
redundant feature rows beyond the first; redundant fully labeled rows; conflicting feature
groups and their total rows; and the largest group size. Do not sum overlapping counts or
call the redundant count the number of all affected rows. A conflicting group is not proof
of a labeling error: the measured features may not identify the same underlying email.

## Outputs and decision gate

Produce append-only manifest, exact source-lock copy, complete integrity report, feature
profiles, duplicate-group listing, all-row group membership and a concise summary. Fail
closed on dirty code, broken source pin or schema/domain violations. Duplicate/conflicting
groups are scientific data-readiness findings, not software errors. A technically successful
audit returns `go_to_split_policy_review`, with `training_ready: false` in every case.

If structurally usable, recommend a prospectively specified group-aware split when duplicate
vectors exist, and require review of any conflicting-label handling. All rows/labels remain
unchanged during this audit. No majority-vote relabeling, deduplication, test-driven tuning,
formal hypothesis tests or predictive claims. The current-page missingness flag conflicts
with the [legacy documentation](https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.DOCUMENTATION);
report what the pinned bytes show without silently rewriting either source.

This whole-release pre-split audit observes values and labels for integrity/grouping. Disclose
that scope; it is not a claim that the release was unseen before future partitioning. Final
inclusion, split, preprocessing, budgets and dataset selection remain separate decisions.
