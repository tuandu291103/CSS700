# Data policy

Do not commit downloaded datasets, processed arrays, trained weights, or generated attribution
files. Store them under the ignored `data/raw`, `data/interim`, `data/processed`, `checkpoints`,
or `artifacts` directories.

Every dataset added to the study must have a small tracked data card in `docs/` that records:

- authoritative source URL and citation;
- license and access date;
- target definition, features, sample size, and missingness;
- inclusion/exclusion and preprocessing decisions;
- ethical or privacy constraints;
- a checksum or immutable dataset version when available.

The Breast Cancer Wisconsin pilot is loaded directly through scikit-learn and is never written
to the repository by the pilot command.

Spambase candidate data were acquired and audited on September 19, 2026, without splitting,
preprocessing or training. See `docs/SPAMBASE_DATA_CARD.md` for attribution, pinned identities,
duplicate/conflicting-label findings and versioned split decisions. Source and receipt
hashes are tracked in `configs/spambase_source_v1.yaml`; raw bytes and audit outputs remain
ignored. A new download receipt is a different acquisition and must not silently replace
that lock. Hashes and Git source synchronization are not an artifact backup.

The later v1 grouped allocation is **diagnostic-only**, despite its automatic technical pass.
All repeated-feature groups were assigned to validation/test. The scientific hold and exact
run pins are in `configs/spambase_split_v1_review.yaml`; see
`docs/logs/2026-09-19-spambase-grouped-split.md`. Do not use v1 for fitting. A future data loader
must verify affirmative scientific acceptance of a revised split, not just a passing run status.

That separate v2 correction is now accepted **for training-only preflight**, with 3,220 /
691 / 690 rows, intact feature groups and balanced repeated-row shares. The authoritative
review/pins are in `configs/spambase_split_v2_review.yaml`; the report is
`docs/logs/2026-09-19-spambase-grouped-split-v2.md`. Preserve the exact accepted IDs and
require this affirmative review when implementing the new loader. No Spambase preprocessing
or model has been fitted yet, and v1 remains diagnostic-only.
