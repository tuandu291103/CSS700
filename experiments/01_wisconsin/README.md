# Experiment 01: Wisconsin data understanding and fixed-grid encoding

23 October 2026. Dataset review and fixed-grid encoding, completed before classifier training.

## Question and scope

The first experiment examines whether each numeric feature can be placed in a grid and recovered without loss. Following my advisor's recommendation to study the pipeline in separate stages, I began with the feature definitions and encoding rules. The fixed grid serves as a simple reference representation before learning a SOM layout. The genomic feasibility work remains documented under `week02/`.

Start with the executed [walkthrough notebook](walkthrough.ipynb). The [machine-readable results](results.json) contain source checksums, software versions, split membership, feature locations, scaling parameters, and numerical checks. Figures are [one encoded sample](encoding.png) and [training correlations](correlations.png).

## Dataset card

**Source:** [UCI Breast Cancer Wisconsin (Diagnostic), DOI 10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B), credited to W. H. Wolberg, W. N. Street, and O. L. Mangasarian in the [original documentation](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names). UCI lists CC BY 4.0. Retrieved 23 October 2026; raw files stay under ignored `data/raw/wdbc/`.

One record describes a breast-mass fine-needle-aspirate sample through image-derived cell-nucleus measurements. The target is benign (`B`) or malignant (`M`). The record ID is retained for traceability, never as a predictor. This is morphology, not gene expression. The source names Street, Wolberg and Mangasarian's 1993 *Nuclear feature extraction for breast tumor diagnosis* as its introductory paper; that article has not been independently reviewed in this milestone. [UCI record and documentation]

### Feature dictionary

The source order has three blocks of ten: `mean`, `se` (standard error), and `worst` (average of the three largest measurements). These are our explicit column names for the original 30 fields. The full field-to-pixel dictionary is in `results.json`, using zero-based predictor indices and coordinates. Raw file field number = predictor index + 3. [Original documentation, Attribute information]

| Base name | Meaning |
|---|---|
| radius | Center-to-boundary distance summary |
| texture | Gray-level variability |
| perimeter | Boundary length |
| area | Nuclear area |
| smoothness | Local radius variation |
| compactness | Perimeter/area-derived shape measure |
| concavity | Inward boundary indentation |
| concave_points | Concave boundary-point count |
| symmetry | Shape symmetry |
| fractal_dimension | Boundary-complexity measure |

This abbreviated dictionary summarizes the source descriptions. The physical units and full extraction formulas have not been independently verified, so the analysis treats the columns as the supplied measurements rather than raw microscopy pixels.

## Protocol chosen for this exercise

The following settings define this experiment rather than the SOM2IMG benchmark:

1. Validate the exact source checksums, 32 fields, valid IDs/labels, finite measurements, and duplicate predictor rows. Stop on duplicate predictors until a grouping policy is specified.
2. Make a stratified approximately 70/15/15 train/validation/test split, seed `20261023`. Store the precise row indices and IDs. Full-file checks cover integrity and label counts only; distributions and correlations use training rows.
3. Fit `MinMaxScaler(clip=False)` on training predictors only. Reuse it for other partitions. Held-out values outside [0,1] are allowed; clipping would discard information. This follows [scikit-learn's leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).
4. Place the 30 scaled features in source order into a **5×6 fixed grid**, using float64. This orientation is our teaching choice, not SOM2IMG's dimension rule. There is no learned geometry, resizing, interpolation, or quantization.
5. Decode all partitions. Require exact recovery of scaled values, then raw-value recovery within `rtol=1e-12, atol=1e-10`. An occupancy mask distinguishes unused pixels from genuine zero values; this particular grid has no padding.
6. Examine training-only correlations and one deterministic training sample (the first training row in source order). Do not select an example because its picture looks convincing.

Wisconsin also appeared in the earlier archived pilot, so the saved test partition is used for preliminary studies rather than final thesis validation. Unique record IDs and the absence of exact duplicates do not establish patient independence. The recovery checks concern the rounded values in the downloaded file, not the original measurement process.

## Fixed grid versus SOM2IMG

| Question | Our fixed-grid exercise | SOM2IMG manuscript reviewed |
|---|---|---|
| What determines coordinates? | Original column order | Learned SOM assignments of feature profiles |
| What is fitted? | Training-only scaling | SOM layout plus preprocessing; exact fold boundaries need clarification |
| Is placement one-to-one? | Explicitly enforced and tested | Collision-resolution rule still needs author confirmation |
| What does color mean? | Scaled input magnitude | Encoded feature value, not automatically model importance |

SOM2IMG statements refer to the supplied manuscript, not a verified match to the final IEEE version. See the [source-specific review](../../week02/SOM2IMG_REVIEW.md). Neither invertibility nor visual grouping proves faithful attribution or useful domain interpretation.

## Reproduce

From the repository root, use Python 3.12. The learning environment is separate from archived environments. Exact installed versions are recorded in `requirements-learning.txt` (macOS ARM development snapshot; not a cross-platform lock).

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-learning.txt
mkdir -p data/raw/wdbc
curl --fail --location https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data --output data/raw/wdbc/wdbc.data
curl --fail --location https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names --output data/raw/wdbc/wdbc.names
.venv/bin/python scripts/wisconsin_learning.py
.venv/bin/python -m unittest discover -s tests -v
```

Open `walkthrough.ipynb` in a notebook-capable editor and select `.venv/bin/python`. Raw files must be present. No notebook cell downloads data or trains a model. Source changes cause a checksum failure rather than silent substitution.

## Next small experiment

After walking through this notebook, compare logistic regression, random forest, and a small fixed-grid CNN on the recorded learning split. Use validation data for choices, and distinguish exploratory results from later confirmatory work. Then study explanations and shortcut controls one at a time. Spambase remains the proposed second dataset, not acquired in this milestone.

At this stage, reproducing SOM2IMG required clarification of the intended paper/code version, dataset files, preprocessing, split seeds and collision rule. The later author-code comparison is recorded separately in [Experiment 07](../07_som2img_layout/README.md).

## Results and interpretation

Computed from the checksummed source files by `scripts/wisconsin_learning.py`; full evidence is in [results.json](results.json).

| Check | Observed result |
|---|---:|
| Samples / predictors | 569 / 30 |
| Benign / malignant records | 357 / 212 |
| Missing or nonfinite predictor values | 0 |
| Duplicate IDs / exact duplicate predictor rows | 0 / 0 |
| Training rows (B / M) | 397 (249 / 148) |
| Validation rows (B / M) | 86 (54 / 32) |
| Learning test rows (B / M) | 86 (54 / 32) |
| Grid coverage | 30 / 30 features |
| Collisions / padding pixels | 0 / 0 |
| Maximum scaled-value round-trip error | 0, every partition |
| Maximum raw-value round-trip error | 4.55 × 10⁻¹³, every partition |
| Constant predictors in training | 0 |

Training-only Spearman correlations were **0.9996** for mean radius versus mean area and **0.9977** for mean radius versus mean perimeter. These relationships are already present in the table. A later layout comparison should therefore assess what the spatial representation adds beyond a conventional correlation analysis.

The experiment produced a documented feature dictionary, reproducible splits, a reversible mapping and an executed walkthrough with two figures. Classifier training and attribution were deferred to subsequent experiments. At completion, all 23 tests passed, including ten new encoding tests and 13 existing audit tests. Lint and notebook validation also passed, and both figures were checked visually.

A second run in a separate temporary directory produced byte-identical `results.json` under the recorded environment.
