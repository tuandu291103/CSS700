# CSS700: Tabular-to-Image Interpretability Research

**Yu Dinh · Master's thesis research**

This project examines whether explanations from convolutional neural networks trained on tabular-to-image representations can be traced back to the original features and interpreted reliably. The revised study began on 15 October 2026. Following my advisor's guidance, I am conducting separate experiments on data preparation, image encoding, classification and feature attribution before combining them in the thesis study.

For the genomic application, the question is whether spatial explanations provide stable, biologically meaningful information about groups of genes beyond what tabular baselines provide. The [literature review](week02/CLAIM_EVIDENCE.md) supports investigating this question, but does not establish that image-based explanations are inherently more useful or that either model family always performs better.

## Current progress

### SOM2IMG attribution comparison, 2 November

I compared Integrated Gradients (IG) and single-feature occlusion across all 21 saved fixed, random and SOM CNNs, using the same 12 evaluation cases and class targets. Explanations were decoded to the original feature names. The study also tested two references, matched feature replacement and full model resets.

With the fitting-median reference, IG magnitude-rank agreement across CNN seeds averaged **0.880 for SOM, 0.745 for random layouts and 0.693 for the fixed grid**. Across layout seeds, SOM also had higher agreement than random layouts. However, changing references affected the explanations, some magnitude agreement survived weight resets, and the replacement test gave nearly every layout a strong result. This is preliminary evidence of greater consistency, not proof of better biological interpretation or faithfulness.

The [Experiment 08 report](experiments/08_som2img_attributions/README.md) includes the tables, three figures, sources and limitations. All 112 project tests pass; an independent execution reproduced the results and figures byte-for-byte. Existing models were not retrained.

### SOM2IMG experiment, 31 October

I used the authors' [tab2img-bench implementation](https://github.com/Kerbash/tab2img-bench/tree/dc0ef691db733ebcba4d13b53ca58e258f76ef0a) to compare fixed, lossless random and SOM layouts on Wisconsin Diagnostic. Seven layouts were crossed with three CNN initialization seeds, giving 21 models. Scaling and layout fitting used 318 records, checkpoint selection used 79 records, and evaluation used the existing 86-record validation partition.

All layouts retained the 30 features. SOM placed more similar training-feature profiles next to one another: mean adjacent absolute correlation was 0.580, compared with 0.416 for random layouts. However, the mean overlap of SOM neighbor sets across seeds was 0.211 Jaccard, indicating substantial variation in the exact neighborhoods.

Mean evaluation accuracy was 98.97% for SOM, 98.32% for random layouts and 97.67% for the fixed layout. The SOM–random difference is small on 86 previously examined records. This experiment establishes a working implementation and a basis for explanation comparisons; it does not yet demonstrate improved interpretation. A separate encoding audit reproduced the authors' documented uint8 wraparound behavior. The classifier comparison used floating-point inputs throughout.

See the [experiment report](experiments/07_som2img_layout/README.md) for the protocol, results, figures and limitations, or the [weekly research notes](docs/WEEKLY_FINDINGS_2026-10-31.md) for a shorter discussion.

### Earlier experiments

| Study | Main outcome | Report |
|---|---|---|
| Wisconsin data and encoding | Documented the 30 features and verified recovery of the original values from a fixed grid | [Experiment 01](experiments/01_wisconsin/README.md) |
| Wisconsin classifiers | Compared logistic regression, random forest and a small CNN on the same learning split | [Experiment 02](experiments/02_wisconsin_models/README.md) |
| CNN explanations | Compared Grad-CAM, IG and same-model SHAP/LIME, with feature replacement and reference checks | [Experiment 03](experiments/03_wisconsin_explanations/README.md) |
| Synthetic shortcuts | Tested whether models relied on shape or background and how the attribution reference affected the result | [Experiment 04](experiments/04_shortcut_sandbox/README.md) |
| TCGA-BRCA data audit | Checked 60,660 expression identifiers, 1,226 samples, participant grouping and clinical metadata | [Experiment 05](experiments/05_tcga_brca_audit/README.md) |
| TCGA-BRCA pilot | Compared three classifier families on 256 training-selected genes, with grouped and collection-site splits | [Experiment 06](experiments/06_tcga_brca_pilot/README.md) |

These are preliminary studies. The TCGA pilot has not yet evaluated biological interpretation, and the final genomic cohort and interpretation criteria remain under discussion. The synthetic data do not count toward the proposed real-dataset comparison.

## Next experiment

The next step is to strengthen the explanation controls: compare IG with a model-free input–reference-distance ranking under the same replacement test, and examine references drawn from real fitting records. This will help distinguish useful model-dependent information from consistency caused by the input and reference. Group-level and genomic interpretation remain later stages requiring a separate protocol.

The supplied code resolves several implementation questions, including collision handling. Its README describes a 51-dataset study, whereas the earlier manuscript I reviewed describes nine benchmarks. The intended paper/code version still needs clarification. Experiment 07 uses the authors' layout code with this project's small CNN and floating-point inputs, rather than reproducing the complete published benchmark.

## Reading and research records

- [Research log](docs/RESEARCH_LOG.md): dated methods, findings and decisions.
- [SOM2IMG review](week02/SOM2IMG_REVIEW.md): manuscript review and a separate update on the supplied code.
- [Evidence review](week02/CLAIM_EVIDENCE.md) and [draft motivation](week02/PROPOSAL_WORDING.md): research rationale and supporting literature.
- [Dataset candidates](week02/DATASET_CANDIDATES.md): proposed datasets and outstanding selection questions.
- [MILE feasibility study](week02/GSE13159_FEASIBILITY.md) and [acceptance protocol](week02/GSE13159_ACCEPTANCE_PROTOCOL.md): specimen confounding, gene mapping and preprocessing checks.
- [TCGA/Xena source review](docs/TCGA_XENA_FEASIBILITY.md): the initial acquisition rationale.
- [Reference notes](references/REFERENCES.md), [BibTeX](references/references.bib) and [reading log](week02/READING_AND_SEARCH_LOG.md).
- [Wisconsin walkthrough](experiments/01_wisconsin/walkthrough.ipynb) and [23 October weekly notes](docs/WEEKLY_COMPLETION_2026-10-23.md).

## Reproducibility

Each experiment directory contains its protocol, results and reproduction instructions. The project currently has 112 passing tests; the earlier author-code check passed all 18 upstream tests. Independent local executions of Experiments 07 and 08 reproduced their result JSON and figures byte-for-byte. Reloading all 21 checkpoints reproduced the saved predictions. Experiment 08 retains detailed attribution vectors and random replacement outcomes for inspection.

The environment snapshot in `requirements-learning.txt` records Python 3.12 on macOS ARM. It does not guarantee identical results on other platforms. Raw data, sample-level genomic manifests and predictions, environments, upstream source checkout and model checkpoints remain outside version control.

```text
CSS700/
├── experiments/                # Protocols, results and experiment reports
├── week02/                     # Literature review and dataset selection
├── references/                 # Reference notes and bibliography
├── scripts/                    # Data preparation and experiment code
├── tests/                      # Data, numerical and result-integrity tests
├── requirements-learning.txt   # Recorded development environment
└── docs/                       # Research log and weekly notes
```

## Earlier work

The implementation preceding the October restart is recoverable through Git history and a verified local archive. [Recovery instructions](docs/ARCHIVE.md) document those locations. Weekly presentation files remain separate from the active research repository.
