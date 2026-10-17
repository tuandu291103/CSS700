# CSS700: Interpreting Gene-Expression Models

Yu Dinh's thesis research on tabular-to-image representations and biological interpretation.

**Current stage: Week 2 — literature evidence and dataset selection.**

This repository contains the revised study started on 15 October 2026. It is the active home for the restart. The current deliverables are research documents and references; dataset acceptance and model implementation are still ahead.

## Research question

Can spatially organized explanations provide stable, biologically meaningful gene-group information beyond well-explained tabular baselines, and what are the predictive-performance tradeoffs?

This is a question to investigate, not a demonstrated result. The study will compare approaches without assuming that CNNs always lose, that tabular feature rankings are useless, or that a visually convincing heatmap proves biological meaning. See the [evidence review](week02/CLAIM_EVIDENCE.md) for supporting sources, counterevidence, and limitations.

## Start here

| Document | Purpose |
|---|---|
| [Claim evidence](week02/CLAIM_EVIDENCE.md) | Assess the main research claims against published work |
| [Proposal wording](week02/PROPOSAL_WORDING.md) | Read the citation-backed draft motivation |
| [Dataset candidates](week02/DATASET_CANDIDATES.md) | Review three candidates, a reserve, and acceptance checks |
| [Reference library](references/REFERENCES.md) | Find nine research references and their reading notes |
| [BibTeX bibliography](references/references.bib) | Reuse citation entries in thesis writing |
| [Research log](docs/RESEARCH_LOG.md) | Follow completed work and the next milestone |

## Status and next milestone

- **Completed:** initial evidence review, revised motivation, and dataset shortlist.
- **Next:** review the scope with the advisor and audit GSE13159 / MILE for feasibility.
- **Pending:** verify sample independence, labels, preprocessing, and gene identifiers before accepting a dataset.
- **Later:** agree on comparison methods and evaluation metrics, then implement the new study.

No new expression matrices have been acquired for the restart, and no new models or experiments have been run. This documentation-stage repository does not yet contain an active training pipeline or test suite.

## Repository layout

```text
CSS700/
├── README.md
├── week02/                 # Evidence review, motivation, datasets, search notes
├── references/             # Reference library, BibTeX, open-access reading copy
└── docs/                   # Research log and recovery information
```

The [reading and search log](week02/READING_AND_SEARCH_LOG.md) records the review's scope and unresolved questions. Dataset descriptions are candidates, not completed data audits. Published results are kept distinct from our future experimental findings.

## Earlier work

The previous implementation and experiment files are no longer part of the active repository. They remain recoverable through Git history and a verified local archive. See [recovery information](docs/ARCHIVE.md). The weekly meeting presentation remains separate from the thesis repository.
