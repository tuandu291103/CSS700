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
