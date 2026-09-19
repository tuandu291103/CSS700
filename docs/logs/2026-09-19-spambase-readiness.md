# Spambase readiness: pinned source and pre-split integrity audit

Date: **2026-09-19**. Authorized milestone: readiness audit only, not dataset selection
or modeling. Source documents supplied metadata, not instructions to execute.

## Brief summary

- Implemented and froze official-source acquisition, hash/receipt verification, exact
  schema checks, duplicate/conflicting-label grouping and append-only audit exports.
- Downloaded and pinned the UCI release. All structural checks passed: **4,601 rows,
  57 features**, expected class counts, and no detected missing/invalid values.
- Found **183 repeated-feature groups involving 577 rows**, with **three conflicting-label
  groups involving six rows**. This is a data-readiness finding, not a novelty claim.
- Preserved every row and label. No split, preprocessing fit, model or explanation was run.
  Next gate is a prospective grouped split and explicit label-handling decision.

## Frozen execution and retained evidence

The implementation/protocol and 16 synthetic tests were committed before data acquisition:
`bbed6d7370bcf3c571a6716f9d5b725a0d0d8ee1`. The initial network attempt failed to resolve
the source hostname within the sandbox; its receipt remains at
`data/raw/20260919-090131-369582__spambase__uci-94/receipt.json`.

The approved network retry succeeded from the same clean code. Source files and receipt:
`data/raw/20260919-090143-663503__spambase__uci-94/`. Acquisition occurred at
2026-09-19 09:01:43 UTC. The archive contains exactly the three expected members; their
retained bytes match the archive. Source identity and receipt hashes were committed in
`configs/spambase_source_v1.yaml` at `f9c2f55a30f40fe8c404a1439c5534c11002ace2`
before inspecting the real data through the audit.

Accepted audit directory:
`artifacts/20260919-090210-691352__spambase__readiness-audit/`.
Outputs: manifest, copied source lock, complete integrity report, feature profiles,
duplicate groups, all-row memberships and summary. All ten checks passed; all six pinned
inputs were unchanged at completion. No failed audit result was replaced or relabeled.

Reproduce the offline audit from a clean code state with the retained source:

```bash
make audit-spambase PYTHON=.venv/bin/python \
  SOURCE_DIR=data/raw/20260919-090143-663503__spambase__uci-94
```

New runs create separate timestamped directories. A fresh acquisition gets a different
receipt; it must not silently replace the committed acquisition pin. Raw artifacts remain
ignored by Git and are not backed up by these commits.

## Results and interpretation

The target counts are 2,788 non-spam and 1,813 spam. Dictionary names/order, punctuation,
58 fields per row, numerical finiteness, declared missing tokens, binary targets, broad
feature domains and the two integer count features all pass. There are no whole-release
constant features. These profiles are integrity evidence, not fitted preprocessing.

There are 4,207 unique feature vectors, leaving 394 redundant feature rows beyond the
first occurrence. Including labels gives 4,210 distinct rows and 391 redundant labeled
rows. The 183 repeated-feature groups contain 577 total rows; the largest has 69. Three
groups contain both labels, two rows each. The six conflicting rows remain unchanged and
are included within those 577 rows. Full definitions and exact conflicting row IDs are
in [`SPAMBASE_DATA_CARD.md`](../SPAMBASE_DATA_CARD.md).

The audit supports a split-policy review, **not an ordinary random row split**: exact
feature groups should stay together to prevent identical vectors crossing partitions.
Measured-feature equality neither proves the same underlying email nor identifies which
conflicting label is correct. No sender/message group identifier is supplied in this
schema. This audit does not rule out other acquisition dependence or near-duplicates.

The [UCI page](https://archive.ics.uci.edu/dataset/94/spambase) and
[legacy documentation](https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.DOCUMENTATION)
disagree on missingness. The pinned data have no missing values detected under our checks;
unknown finite sentinels remain untestable without further source knowledge. The data card
records attribution/license, version and the historical collection-bias limitation.

## Verification

**114 tests passed in 25.76 seconds**, including 16 new synthetic cases. Lint, formatting
and whitespace checks passed. Tests exercise source/receipt tampering, retained archive
consistency, missing/invalid values, signed-zero and near-value grouping, conflicting-label
counts, dirty-code rejection and input changes during an audit; tests do not download data.

A separate read-only calculation used `numpy.loadtxt` and `numpy.unique`, with no imports
from the audit implementation. It reproduced every duplicate/class count and verified all
4,601 row IDs, memberships, hashes and labels; every duplicate group's members/counts;
all 57 profiles; the copied source lock, archive member bytes and six recorded input hashes.
This alternative parser/grouping check passed. No source or old research artifact changed.

## On-track check and next step

This readiness milestone is complete and removes uncertainty about the candidate's raw
integrity. It adds **zero trained CNN conditions**: the matrix remains **3 of the expected
12**, not a thesis-completion percentage or assurance about the defense calendar.
Authoritative SOM2IMG code/collision policy and committee scope decisions are still open.
No further LIME tuning or efficiency claim was introduced.

Recommended next bounded milestone: confirm Spambase inclusion and freeze a versioned
group-aware split contract, retaining all original rows/labels as the proposed default.
Define how the three conflicting groups are handled, allocation/class-balance tolerances,
fixed split seed, training-only preprocessing and grouped evaluation/uncertainty. Review
the split manifest and verify zero cross-partition feature-group overlap before fitting
anything. Training/explainer feasibility and the first Grid seed-42 run follow their own
gates; they were not authorized or executed by this audit. No committee message or Git push
was sent.
