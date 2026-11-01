# CSS700: Tabular-to-Image Interpretability Research

Yu Dinh's thesis research on tabular-to-image representations and interpretable machine learning.

**Current stage: author-code SOM2IMG layout comparison completed; next is a frozen-model explanation comparison across layouts.**

This repository contains the revised study started on 15 October 2026. Following the professor's updated guidance, we are learning the data, conversion, CNN, and explanation steps separately before confirmatory thesis experiments. Genomic literature and MILE feasibility work are preserved for later; their acceptance questions remain unresolved.

## Longer-term research question

Can spatially organized explanations provide stable, biologically meaningful gene-group information beyond well-explained tabular baselines, and what are the predictive-performance tradeoffs?

This is a question to investigate, not a demonstrated result. The study will compare approaches without assuming that CNNs always lose, that tabular feature rankings are useless, or that a visually convincing heatmap proves biological meaning. See the [evidence review](week02/CLAIM_EVIDENCE.md) for supporting sources, counterevidence, and limitations.

## Start here

| Document | Purpose |
|---|---|
| [Weekly findings — 31 October](docs/WEEKLY_FINDINGS_2026-10-31.md) | Professor-ready findings, connection to last week's work, limitations and discussion questions |
| [Experiment 07: SOM2IMG layouts](experiments/07_som2img_layout/README.md) | Pinned author code; preservation, neighborhood stability, matched CNNs and numerical export audit |
| [Experiment 06: first TCGA-BRCA pilot](experiments/06_tcga_brca_pilot/README.md) | Patient-safe splits, 256-gene grid, three classifier families, BH site holdout, results and professor-alignment check |
| [Experiment 05: TCGA-BRCA data audit](experiments/05_tcga_brca_audit/README.md) | Complete downloads, patient/label/gene checks, collection-site risks and the original cohort options |
| [TCGA/Xena initial source check — 26 October](docs/TCGA_XENA_FEASIBILITY.md) | Historical resource inspection and rationale, with a link to the subsequent full audit |
| [Weekly summary and meeting notes](docs/WEEKLY_COMPLETION_2026-10-23.md) | Deliverables, lessons, professor questions and a short self-check |
| [Experiment 04: shape or background?](experiments/04_shortcut_sandbox/README.md) | Controlled shortcut demonstration and explanation-reference pitfalls |
| [Experiment 03: frozen-model explanations](experiments/03_wisconsin_explanations/README.md) | Grad-CAM, IG, SHAP/LIME, feature tracing and diagnostic checks |
| [Experiment 02: model comparison](experiments/02_wisconsin_models/README.md) | Prediction results, CNN layer walkthrough, training diagnostics, and limitations |
| [Experiment 01: Wisconsin](experiments/01_wisconsin/README.md) | Start with the dataset card, feature dictionary, protocol, results, and next learning step |
| [Executed walkthrough](experiments/01_wisconsin/walkthrough.ipynb) | Follow one sample from tabular values to pixels and back |
| [Claim evidence](week02/CLAIM_EVIDENCE.md) | Assess the main research claims against published work |
| [Proposal wording](week02/PROPOSAL_WORDING.md) | Read the citation-backed draft motivation |
| [SOM2IMG review](week02/SOM2IMG_REVIEW.md) | Historical nine-benchmark manuscript review, with a separate update for the newly supplied code |
| [Dataset candidates](week02/DATASET_CANDIDATES.md) | Review three candidates, a reserve, and acceptance checks |
| [MILE feasibility audit](week02/GSE13159_FEASIBILITY.md) | See verified counts, specimen-type imbalance, mapping ambiguity, and remaining gates |
| [MILE acceptance protocol](week02/GSE13159_ACCEPTANCE_PROTOCOL.md) | Review proposed acceptance gates, split rules, specimen controls, and preprocessing policy |
| [Reference library](references/REFERENCES.md) | Find research references and their reading notes |
| [BibTeX bibliography](references/references.bib) | Reuse citation entries in thesis writing |
| [Research log](docs/RESEARCH_LOG.md) | Follow completed work and the next milestone |

## Status and next milestone

- **31 October author-code experiment:** pinned `Kerbash/tab2img-bench` at `dc0ef691db733ebcba4d13b53ca58e258f76ef0a`. Compared fixed, lossless random and SOM layouts on Wisconsin, crossing seven layouts with three CNN seeds. Used 318 fitting / 79 inner-stop / 86 reused validation records; the old learning-test records were not transformed or scored. [Protocol and evidence](experiments/07_som2img_layout/README.md).
- **New learning findings:** every layout preserved 30/30 features. SOM's adjacent training-profile absolute correlation averaged 0.580 versus 0.416 for random, but its mean neighbor-set overlap across seeds was only 0.211 Jaccard. Mean evaluation accuracy was 98.97% SOM, 98.32% random and 97.67% fixed: small exploratory differences, not general superiority or interpretation quality. An input audit also reproduced the authors' documented uint8 wraparound behavior; the matched CNN experiment avoids it using floating-point inputs.
- **26 October genomic pilot:** adopted the conservative learning cohort (1,059 tumors + 111 normals), froze participant/case-disjoint partitions and a separate BH-site holdout, selected 256 genes using each training partition only, and verified exact 16×16 image/table recovery. Trained logistic regression, random forest and the small CNN plus a prior control. All models in both scenarios were fixed before test scoring. [Protocol, results and limitations](experiments/06_tcga_brca_pilot/README.md).
- **Genomic learning result:** on the 177-sample grouped test, logistic regression and all forest runs scored 177/177; CNN runs scored 171/173/172. On the 221-sample BH holdout, balanced accuracy was 96.24% for logistic regression, 92.01% for forest (mean) and 96.38% for CNN (mean). The tiny CNN/logistic difference is not established superiority. Only 17 normals are in the grouped test and six in BH validation. No biological interpretation benefit or general model ranking is claimed.
- **26 October genomic audit:** downloaded all three TCGA-BRCA data tables and checked 60,660 expression identifiers × 1,226 samples. All samples match clinical records, all gene IDs map exactly, and all 74,369,160 values are finite/nonnegative. The extra Xena identifier is `sampleID`, verified against the complete browser inventory. There are 1,095 participants, not 1,226 independent patients. [Audit evidence](experiments/05_tcga_brca_audit/README.md).
- **Cohort boundary:** the original binary candidate has 1,095 tumors and 113 normals, with all normals paired to tumor participants. The conservative option is now adopted for Experiment 06 only. Clinical/preservation exclusions, collection-site imbalance and the final biological target still need thesis-level decisions; this is not final dataset acceptance.
- **Completed, 23 October:** audited Wisconsin and verified reversible encoding, then trained logistic regression, random forest, and a 1,282-parameter CNN on the saved split, with a majority/prior control. Forest and CNN each use three fixed seeds. Settings were recorded before training; all CNN checkpoints were selected by validation loss before test scoring.
- **Wisconsin learning result:** logistic regression achieved 83/86 correct learning-test predictions; CNN runs achieved 83, 83, and 82; forest runs 80, 79, and 79. CNN accuracy was close to logistic regression but its mean log loss was higher. These are exploratory results from one split, not a general model ranking or an interpretability benefit. See [the full report](experiments/02_wisconsin_models/README.md).
- **Newly completed:** explained the first frozen CNN on 12 preselected validation cases plus one illustrative error, including named-feature tracing, same-model SHAP/LIME comparisons, feature-replacement, reference and parameter-reset checks. IG/SHAP magnitude rankings agreed strongly; the Grad-CAM coordinate proxy performed poorly in our feature-replacement diagnostic. These are bounded learning results, not a universal method ranking. [Explanation evidence](experiments/03_wisconsin_explanations/README.md).
- **Controlled lesson:** synthetic shape classifiers trained on label-correlated backgrounds dropped from 100% to 0% accuracy when backgrounds were reversed; independent-background controls stayed at 100% across all three seeds. A foreground-only attribution map can still result from the reference choice. [Shortcut evidence and limits](experiments/04_shortcut_sandbox/README.md).
- **Next:** keep all existing models frozen and specify a small attribution comparison across the fixed/random/SOM layouts, decoded to the same feature names with reference and perturbation controls. Clarify biological usefulness before returning to genomic interpretation. Do not retune Wisconsin or BRCA models against observed evaluation/test scores. Spambase remains proposed, not acquired; the synthetic sandbox does not count as a real dataset.
- **SOM2IMG version boundary:** the supplied code resolves the API, collision and fold-local-fitting questions for its pinned version. It describes a 51-dataset study, not automatically the same version as our nine-dataset reading copy or final IEEE paper. Experiment 07 uses its actual layout implementation but deliberately retains our CNN, float inputs and learning split; it is not a complete paper-benchmark reproduction.
- **Not yet resolved:** genomic dataset acceptance, including the new TCGA candidate and preserved MILE patient/specimen/provenance questions; comparison/metric selection for the eventual thesis study.

Learned SOM layouts have now been tested on Wisconsin, but no genomic interpretation-quality result has been produced. The suite has **94 passing project tests**; the pinned author repository's **18 tests** also pass. Experiment 07's full independent rerun reproduced its JSON and four figures byte-for-byte, and all 21 checkpoints reload to matching predictions. Earlier genomic and explanation reproducibility checks remain documented in their own reports. Raw data, sample-level genomic manifests/predictions, `.venv`, upstream checkout and model checkpoints are ignored by Git; protocols, compact evidence, mappings and figures are tracked. Reproduction commands are in each experiment report. The environment snapshot is specific to Python 3.12/macOS ARM, not a universal platform lock.

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

The [reading and search log](week02/READING_AND_SEARCH_LOG.md) records the review's scope and unresolved questions. Genomic candidates have not passed final thesis acceptance: TCGA-BRCA now has an audit and exploratory prediction pilot; MILE has a partial feasibility audit; other genomic candidates remain shortlisted. The professor's broader three-dataset/comparative-method/interpretation-evaluation plan remains ahead. Published results are kept distinct from our experimental findings.

## Earlier work

The previous implementation and experiment files are no longer part of the active repository. They remain recoverable through Git history and a verified local archive. See [recovery information](docs/ARCHIVE.md). The weekly meeting presentation remains separate from the thesis repository.
