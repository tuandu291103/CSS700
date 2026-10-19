# CSS700: Interpreting Gene-Expression Models

Yu Dinh's thesis research on tabular-to-image representations and biological interpretation.

**Current stage: Week 2 — literature evidence and dataset selection.**

This repository contains the revised study started on 15 October 2026. It is the active home for the restart. Current deliverables include research documents, references, and a small reproducible metadata audit; dataset acceptance and model implementation are still ahead.

## Research question

Can spatially organized explanations provide stable, biologically meaningful gene-group information beyond well-explained tabular baselines, and what are the predictive-performance tradeoffs?

This is a question to investigate, not a demonstrated result. The study will compare approaches without assuming that CNNs always lose, that tabular feature rankings are useless, or that a visually convincing heatmap proves biological meaning. See the [evidence review](week02/CLAIM_EVIDENCE.md) for supporting sources, counterevidence, and limitations.

## Start here

| Document | Purpose |
|---|---|
| [Claim evidence](week02/CLAIM_EVIDENCE.md) | Assess the main research claims against published work |
| [Proposal wording](week02/PROPOSAL_WORDING.md) | Read the citation-backed draft motivation |
| [Dataset candidates](week02/DATASET_CANDIDATES.md) | Review three candidates, a reserve, and acceptance checks |
| [MILE feasibility audit](week02/GSE13159_FEASIBILITY.md) | See verified counts, specimen-type imbalance, mapping ambiguity, and remaining gates |
| [Reference library](references/REFERENCES.md) | Find research references and their reading notes |
| [BibTeX bibliography](references/references.bib) | Reuse citation entries in thesis writing |
| [Research log](docs/RESEARCH_LOG.md) | Follow completed work and the next milestone |

## Status and next milestone

- **Completed:** initial evidence review, revised motivation, dataset shortlist, and MILE metadata/platform-annotation audit.
- **Latest finding (19 October):** 2,096 sample records and 18 labels verified; strong specimen-type/diagnosis association and ambiguous probe-to-gene mappings identified. These are audit findings, not model results.
- **Next:** resolve MILE patient/center metadata and processing provenance, then draft its acceptance protocol for advisor review.
- **Pending:** address specimen-type confounding, verify expression data and current gene mapping, and audit the remaining candidates before accepting datasets.
- **Later:** agree on comparison methods and evaluation metrics, then implement the new study.

No expression matrices have been acquired for the restart, and no models or predictive experiments have been run. The only code is a standard-library metadata audit with 10 synthetic tests, not a training pipeline. Local metadata/annotation downloads are ignored by Git; a small [checksummed audit summary](week02/GSE13159_AUDIT.json) is tracked. See the [audit report](week02/GSE13159_FEASIBILITY.md#reproduce-this-audit) for reproduction and test commands.

## Repository layout

```text
CSS700/
├── README.md
├── week02/                 # Evidence review, motivation, datasets, search notes
├── references/             # Reference library, BibTeX, open-access reading copy
├── scripts/                # Bounded metadata audit only
├── tests/                  # Synthetic audit checks only
└── docs/                   # Research log and recovery information
```

The [reading and search log](week02/READING_AND_SEARCH_LOG.md) records the review's scope and unresolved questions. Dataset candidates have not passed full acceptance: MILE has a partial feasibility audit; the others remain at the shortlist stage. Published results are kept distinct from our future experimental findings.

## Earlier work

The previous implementation and experiment files are no longer part of the active repository. They remain recoverable through Git history and a verified local archive. See [recovery information](docs/ARCHIVE.md). The weekly meeting presentation remains separate from the thesis repository.
