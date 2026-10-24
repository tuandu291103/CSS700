# CSS700: Tabular-to-Image Interpretability Research

Yu Dinh's thesis research on tabular-to-image representations and interpretable machine learning.

**Current stage: this week's one-dataset learning cycle completed — explanations and shortcut checks included.**

This repository contains the revised study started on 15 October 2026. Following the professor's updated guidance, we are learning the data, conversion, CNN, and explanation steps separately before confirmatory thesis experiments. Genomic literature and MILE feasibility work are preserved for later; their acceptance questions remain unresolved.

## Longer-term research question

Can spatially organized explanations provide stable, biologically meaningful gene-group information beyond well-explained tabular baselines, and what are the predictive-performance tradeoffs?

This is a question to investigate, not a demonstrated result. The study will compare approaches without assuming that CNNs always lose, that tabular feature rankings are useless, or that a visually convincing heatmap proves biological meaning. See the [evidence review](week02/CLAIM_EVIDENCE.md) for supporting sources, counterevidence, and limitations.

## Start here

| Document | Purpose |
|---|---|
| [Weekly summary and meeting notes](docs/WEEKLY_COMPLETION_2026-10-23.md) | Deliverables, lessons, professor questions and a short self-check |
| [Experiment 04: shape or background?](experiments/04_shortcut_sandbox/README.md) | Controlled shortcut demonstration and explanation-reference pitfalls |
| [Experiment 03: frozen-model explanations](experiments/03_wisconsin_explanations/README.md) | Grad-CAM, IG, SHAP/LIME, feature tracing and diagnostic checks |
| [Experiment 02: model comparison](experiments/02_wisconsin_models/README.md) | Prediction results, CNN layer walkthrough, training diagnostics, and limitations |
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

- **Completed, 23 October:** audited Wisconsin and verified reversible encoding, then trained logistic regression, random forest, and a 1,282-parameter CNN on the saved split, with a majority/prior control. Forest and CNN each use three fixed seeds. Settings were recorded before training; all CNN checkpoints were selected by validation loss before test scoring.
- **Latest learning result:** logistic regression achieved 83/86 correct learning-test predictions; CNN runs achieved 83, 83, and 82; forest runs 80, 79, and 79. CNN accuracy was close to logistic regression but its mean log loss was higher. These are exploratory results from one split, not a general model ranking or an interpretability benefit. See [the full report](experiments/02_wisconsin_models/README.md).
- **Newly completed:** explained the first frozen CNN on 12 preselected validation cases plus one illustrative error, including named-feature tracing, same-model SHAP/LIME comparisons, feature-replacement, reference and parameter-reset checks. IG/SHAP magnitude rankings agreed strongly; the Grad-CAM coordinate proxy performed poorly in our feature-replacement diagnostic. These are bounded learning results, not a universal method ranking. [Explanation evidence](experiments/03_wisconsin_explanations/README.md).
- **Controlled lesson:** synthetic shape classifiers trained on label-correlated backgrounds dropped from 100% to 0% accuracy when backgrounds were reversed; independent-background controls stayed at 100% across all three seeds. A foreground-only attribution map can still result from the reference choice. [Shortcut evidence and limits](experiments/04_shortcut_sandbox/README.md).
- **Next:** review the [weekly walkthrough and professor questions](docs/WEEKLY_COMPLETION_2026-10-23.md), then specify the next isolated layout comparison. Do not tune these models against the observed test scores. Spambase remains a proposed second real dataset, not acquired yet; the synthetic sandbox does not count as one.
- **SOM2IMG reproduction gate:** obtain the intended paper/code version, dataset files, preprocessing, splits, and collision rule. This fixed-grid exercise does not reproduce SOM2IMG.
- **Deferred, preserved:** genomic dataset acceptance, including MILE patient/specimen confounding and preprocessing provenance; comparison/metric selection for the eventual thesis study.

No learned SOM layout or genomic predictive experiment has been produced in the restarted study. The suite now has 50 passing tests (13 MILE audit, ten encoding, ten model/results, nine attribution and eight shortcut checks). Both new experiments reproduced byte-for-byte on the recorded environment. Raw data, `.venv`, and model checkpoints are ignored by Git; protocols, small results, and figures are tracked. Reproduction commands are in each experiment report. The environment snapshot is specific to Python 3.12/macOS ARM, not a universal platform lock.

## Repository layout

```text
CSS700/
├── README.md
├── experiments/            # Separate learning exercises; start with 01_wisconsin
├── week02/                 # Evidence review, motivation, datasets, search notes
├── references/             # Reference library, BibTeX, open-access reading copy
├── scripts/                # Data audit, encoding, training, attribution and shortcut controls
├── tests/                  # Analytic checks, data controls, training and saved-result integrity
├── requirements-learning.txt # Exact local learning-environment snapshot
└── docs/                   # Research log and recovery information
```

The [reading and search log](week02/READING_AND_SEARCH_LOG.md) records the review's scope and unresolved questions. Dataset candidates have not passed full acceptance: MILE has a partial feasibility audit; the others remain at the shortlist stage. Published results are kept distinct from our future experimental findings.

## Earlier work

The previous implementation and experiment files are no longer part of the active repository. They remain recoverable through Git history and a verified local archive. See [recovery information](docs/ARCHIVE.md). The weekly meeting presentation remains separate from the thesis repository.
