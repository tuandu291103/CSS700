# CSS700: Tabular-to-Image Interpretability Research

Yu Dinh's thesis research on tabular-to-image representations and interpretable machine learning.

**Current stage: separate learning experiments — Wisconsin data and fixed-grid encoding.**

This repository contains the revised study started on 15 October 2026. Following the professor's updated guidance, we are learning the data, conversion, CNN, and explanation steps separately before confirmatory thesis experiments. Genomic literature and MILE feasibility work are preserved for later; their acceptance questions remain unresolved.

## Longer-term research question

Can spatially organized explanations provide stable, biologically meaningful gene-group information beyond well-explained tabular baselines, and what are the predictive-performance tradeoffs?

This is a question to investigate, not a demonstrated result. The study will compare approaches without assuming that CNNs always lose, that tabular feature rankings are useless, or that a visually convincing heatmap proves biological meaning. See the [evidence review](week02/CLAIM_EVIDENCE.md) for supporting sources, counterevidence, and limitations.

## Start here

| Document | Purpose |
|---|---|
| [Experiment 01: Wisconsin](experiments/01_wisconsin/README.md) | Start with the dataset card, feature dictionary, protocol, results, and next learning step |
| [Executed walkthrough](experiments/01_wisconsin/walkthrough.ipynb) | Follow one sample from tabular values to pixels and back |
| [Claim evidence](week02/CLAIM_EVIDENCE.md) | Assess the main research claims against published work |
| [Proposal wording](week02/PROPOSAL_WORDING.md) | Read the citation-backed draft motivation |
| [SOM2IMG review](week02/SOM2IMG_REVIEW.md) | Check the reference method, nine benchmarks, interpretation claims, and reproduction questions |
| [Dataset candidates](week02/DATASET_CANDIDATES.md) | Review three candidates, a reserve, and acceptance checks |
| [MILE feasibility audit](week02/GSE13159_FEASIBILITY.md) | See verified counts, specimen-type imbalance, mapping ambiguity, and remaining gates |
| [MILE acceptance protocol](week02/GSE13159_ACCEPTANCE_PROTOCOL.md) | Review proposed acceptance gates, split rules, specimen controls, and preprocessing policy |
| [Reference library](references/REFERENCES.md) | Find research references and their reading notes |
| [BibTeX bibliography](references/references.bib) | Reuse citation entries in thesis writing |
| [Research log](docs/RESEARCH_LOG.md) | Follow completed work and the next milestone |

## Status and next milestone

- **Completed, 23 October:** acquired and checksummed the official Wisconsin files; audited 569 rows and 30 predictors; made a 397/86/86 learning split; built a reversible 5×6 fixed grid with training-only scaling; added an executed notebook, two figures, and ten synthetic tests. See [results and limitations](experiments/01_wisconsin/README.md#results-and-session-summary).
- **Learning observation:** radius, perimeter, and area are strongly correlated in the training data before image conversion. Displaying this relationship alone would not establish added interpretability. This is an exploratory observation, not a new biological discovery.
- **Next:** walk through the data and encoding, then compare simple tabular baselines with a small fixed-grid CNN. Study explanation methods and controlled shortcuts afterward. Spambase is the proposed second learning dataset, not yet acquired.
- **SOM2IMG reproduction gate:** obtain the intended paper/code version, dataset files, preprocessing, splits, and collision rule. This fixed-grid exercise does not reproduce SOM2IMG.
- **Deferred, preserved:** genomic dataset acceptance, including MILE patient/specimen confounding and preprocessing provenance; comparison/metric selection for the eventual thesis study.

No new classifier, predictive score, learned SOM layout, or attribution result has been produced in the restarted study. The 13 existing MILE audit tests remain; ten Wisconsin tests bring the suite to 23. Raw data and `.venv` are ignored by Git. Reproduction commands and the exact learning environment snapshot are linked from [Experiment 01](experiments/01_wisconsin/README.md#reproduce).

## Repository layout

```text
CSS700/
├── README.md
├── experiments/            # Separate learning exercises; start with 01_wisconsin
├── week02/                 # Evidence review, motivation, datasets, search notes
├── references/             # Reference library, BibTeX, open-access reading copy
├── scripts/                # MILE audit and Wisconsin learning pipeline
├── tests/                  # Synthetic audit, mapping, and split checks
├── requirements-learning.txt # Exact local learning-environment snapshot
└── docs/                   # Research log and recovery information
```

The [reading and search log](week02/READING_AND_SEARCH_LOG.md) records the review's scope and unresolved questions. Dataset candidates have not passed full acceptance: MILE has a partial feasibility audit; the others remain at the shortlist stage. Published results are kept distinct from our future experimental findings.

## Earlier work

The previous implementation and experiment files are no longer part of the active repository. They remain recoverable through Git history and a verified local archive. See [recovery information](docs/ARCHIVE.md). The weekly meeting presentation remains separate from the thesis repository.
