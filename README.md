# CSS700: Tabular-to-Image Interpretability Research

**Yu Dinh · Master's thesis research**

This project examines whether explanations from convolutional neural networks trained on tabular-to-image representations can be traced back to the original features and interpreted reliably. The revised study began on 15 October 2026. Following my advisor's guidance, I am conducting separate experiments on data preparation, image encoding, classification and feature attribution before combining them in the thesis study.

For the genomic application, the question is whether spatial explanations provide stable, biologically meaningful information about groups of genes beyond what tabular baselines provide. The [literature review](week02/CLAIM_EVIDENCE.md) supports investigating this question, but does not establish that image-based explanations are inherently more useful or that either model family always performs better.

## Start here

- **Latest study:** [Experiment 09 — reference and model-free controls](experiments/09_reference_controls/README.md).
- **Research history:** [dated research log](docs/RESEARCH_LOG.md).
- **Research rationale:** [evidence review](week02/CLAIM_EVIDENCE.md) and [draft motivation](week02/PROPOSAL_WORDING.md).

## Current findings

Experiment 09 extends the explanation study using the same 21 frozen CNNs and twelve Wisconsin evaluation cases. It compares Integrated Gradients (IG) against a model-free input–reference-distance ranking and adds four real fitting records as references.

With these four references, mean IG magnitude-rank agreement across CNN seeds was **0.901 for SOM, 0.757 for random layouts and 0.626 for the fixed grid**. IG produced a larger absolute score change than the distance control in 334 of 432 SOM model/case/reference comparisons, with one tie. These observations are dependent, not 432 independent cases.

The controls also limit the conclusion. With the median reference, SOM's mean random percentile was almost identical for IG and distance (0.8819 and 0.8821), despite IG winning more pairwise comparisons. Changing the reference class substantially changed signed explanations. One of 1,008 newly computed IG vectors missed the numerical tolerance at the step cap and remains explicitly reported. The findings support further study of stability and reference choice; they do not yet establish biological usefulness or superiority to tabular explanations. See the [full report](experiments/09_reference_controls/README.md) for methods, figures, results and limitations.

## Experiment index

The numbering follows the order of the learning studies. Each report explains its scope, evidence and reproduction requirements.

| Study | Main outcome | Report |
|---|---|---|
| 01 · Wisconsin data and encoding | Documented 30 morphology features and verified recovery of values from a fixed grid | [Data walkthrough](experiments/01_wisconsin/README.md) |
| 02 · Wisconsin classifiers | Compared logistic regression, random forest and a small CNN on the same split | [Classifier comparison](experiments/02_wisconsin_models/README.md) |
| 03 · CNN explanations | Compared Grad-CAM, IG and same-model SHAP/LIME with replacement and reference checks | [Attribution methods](experiments/03_wisconsin_explanations/README.md) |
| 04 · Synthetic shortcuts | Tested shape versus background reliance and the influence of attribution references | [Shortcut experiment](experiments/04_shortcut_sandbox/README.md) |
| 05 · TCGA-BRCA audit | Checked 60,660 expression identifiers, 1,226 samples, participant grouping and metadata | [Data audit](experiments/05_tcga_brca_audit/README.md) |
| 06 · TCGA-BRCA pilot | Compared three classifier families on 256 training-selected genes with grouped and site splits | [Genomic prediction pilot](experiments/06_tcga_brca_pilot/README.md) |
| 07 · SOM2IMG layouts | Compared seven fixed, random and SOM layouts across three CNN seeds; audited encoding | [Layout comparison](experiments/07_som2img_layout/README.md) |
| 08 · SOM2IMG attributions | Traced IG and occlusion to named features; compared seeds, references and weight resets | [Attribution comparison](experiments/08_som2img_attributions/README.md) |
| 09 · Reference controls | Compared IG with model-free distance and four real-record references | [Control study](experiments/09_reference_controls/README.md) |

These are preliminary studies. Wisconsin contains cell-morphology measurements, not gene expression. The TCGA pilot has not evaluated biological interpretation. Synthetic bars do not count as a real dataset, and Xena and TCGA are not independent datasets.

## Next research decisions

1. Review the reference-sensitive findings and numerical exception in Experiment 09 with the advisor before setting a new protocol.
2. Define a group-level interpretation question and a replacement strategy that addresses correlated features and unrealistic hybrid inputs. This remains proposed work.
3. Agree on the genomic cohort, biological interpretation criteria and fair tabular comparators before extending the explanation study to genes. The full multi-dataset thesis evaluation is still outstanding.

The supplied SOM2IMG code resolves several implementation questions, including collision handling. Its README describes a 51-dataset study, whereas the earlier manuscript I reviewed describes nine benchmarks. The intended paper/code version still needs clarification. Experiment 07 uses the [pinned author implementation](https://github.com/Kerbash/tab2img-bench/tree/dc0ef691db733ebcba4d13b53ca58e258f76ef0a) with this project's small CNN and floating-point inputs, rather than reproducing the complete published benchmark.

## Reading and research records

- [Research log](docs/RESEARCH_LOG.md): dated methods, findings and decisions.
- [SOM2IMG review](week02/SOM2IMG_REVIEW.md): manuscript review and a separate update on the supplied code.
- [Evidence review](week02/CLAIM_EVIDENCE.md) and [draft motivation](week02/PROPOSAL_WORDING.md): research rationale and supporting literature.
- [Dataset candidates](week02/DATASET_CANDIDATES.md): proposed datasets and outstanding selection questions.
- [MILE feasibility study](week02/GSE13159_FEASIBILITY.md) and [acceptance protocol](week02/GSE13159_ACCEPTANCE_PROTOCOL.md): specimen confounding, gene mapping and preprocessing checks.
- [TCGA/Xena source review](docs/TCGA_XENA_FEASIBILITY.md): the initial acquisition rationale.
- [Reference notes](references/REFERENCES.md), [BibTeX](references/references.bib) and [reading log](week02/READING_AND_SEARCH_LOG.md).
- [Wisconsin walkthrough](experiments/01_wisconsin/walkthrough.ipynb) and [23 October weekly notes](docs/WEEKLY_COMPLETION_2026-10-23.md).

## Repository organization

| Location | Purpose |
|---|---|
| `experiments/01_*` through `09_*` | One report per study, with its protocol where applicable, result JSON and figures |
| `scripts/` | Data preparation and experiment runners |
| `tests/` | Analytic, data-integrity and saved-result checks |
| `week02/` | Literature and dataset-selection records, retained at their original paths |
| `references/` | Reference notes, bibliography and the documented reading copy |
| `docs/` | Dated research log, source records, weekly notes and archive recovery instructions |
| `requirements-learning.txt` | Recorded Python development environment |

Version control retains code, tests, protocols and the evidence needed to review the results. Detailed result JSON is intentional: it preserves feature vectors, numerical diagnostics and comparisons that the summaries cannot show. Raw data, sample-level genomic manifests and predictions, environments, checkpoints, upstream checkout, duplicate verification runs and presentation exports remain local under the existing ignore rules. Historical reports retain their original results and milestone test counts.

## Reproducibility

Each experiment report contains its inputs and reproduction commands. The environment snapshot records Python 3.12 on macOS ARM; it is not a platform-independent lock. A fresh clone does not contain raw datasets or checkpoints, and some tests explicitly skip checks requiring those local files.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-learning.txt
MPLCONFIGDIR=/private/tmp/css700-matplotlib XDG_CACHE_HOME=/private/tmp/css700-cache .venv/bin/python -m unittest discover -s tests -q
```

On 4 November, **all 130 tests passed** in the existing local environment. Ruff passed for the Experiment 09 script and tests. The saved independent execution of Experiment 09 matches its result JSON and all three figures byte-for-byte; those hashes were rechecked during repository review. Experiments 07 and 08 also have previously documented independent reproductions. No training or full attribution run was repeated during this cleanup.

## Earlier work

The implementation preceding the October restart is recoverable through Git history and a verified local archive. [Recovery instructions](docs/ARCHIVE.md) document those locations. Weekly presentations and the standalone review paper remain separate from the active research repository.
