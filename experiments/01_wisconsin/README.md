# Experiment 01: Wisconsin data understanding and fixed-grid encoding

23 October 2026 · Learning milestone · No classifier or explanation method trained

## Question and scope

Can we explain the data fields and trace every numeric feature into a grid and back without loss? This is the first small exercise requested after the professor's learning-first guidance. It is not a SOM2IMG reproduction, a clinical evaluation, or a thesis novelty claim. Earlier genomic work remains available under `week02/`.

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

The brief dictionary is for orientation, not a replacement for the original feature-extraction specification. Physical units and full extraction formulas are not independently established here; do not invent micrometers or interpret these as raw microscopy pixels.

## Protocol chosen for this exercise

These are our learning-design choices, not settings claimed from SOM2IMG:

1. Validate the exact source checksums, 32 fields, valid IDs/labels, finite measurements, and duplicate predictor rows. Stop on duplicate predictors until a grouping policy is specified.
2. Make a stratified approximately 70/15/15 train/validation/test split, seed `20261023`. Store the precise row indices and IDs. Full-file checks cover integrity and label counts only; distributions and correlations use training rows.
3. Fit `MinMaxScaler(clip=False)` on training predictors only. Reuse it for other partitions. Held-out values outside [0,1] are allowed; clipping would discard information. This follows [scikit-learn's leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).
4. Place the 30 scaled features in source order into a **5×6 fixed grid**, using float64. This orientation is our teaching choice, not SOM2IMG's dimension rule. There is no learned geometry, resizing, interpolation, or quantization.
5. Decode all partitions. Require exact recovery of scaled values, then raw-value recovery within `rtol=1e-12, atol=1e-10`. An occupancy mask distinguishes unused pixels from genuine zero values; this particular grid has no padding.
6. Examine training-only correlations and one deterministic training sample (the first training row in source order). Do not select an example because its picture looks convincing.

The recorded learning test partition is not certified as an untouched final thesis holdout: this dataset appeared in the earlier archived pilot and is now used for learning. Unique record IDs and absence of exact duplicates do not establish patient independence or external clinical validity. Source values are already rounded; round-trip preservation concerns the downloaded numeric values, not the underlying measurement process.

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

Ask the professor or previous student for the intended SOM2IMG version, source code, dataset files, preprocessing, split seeds, and collision rule before numerical reproduction. No contact or author-code verification is claimed here.

## Results and session summary

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

Training-only Spearman correlations include mean radius versus mean area **0.9996**, and mean radius versus mean perimeter **0.9977**. These associations predate image conversion. Our interpretation: a meaningful later comparison should ask what the layout or explanation adds beyond a standard correlation view, not claim that spatial placement discovered these relationships. No causal, diagnostic, or biological benefit has been demonstrated.

**Completed:** acquired sources, documented features, checked data quality, stored reproducible splits and mappings, traced values both ways, and executed a walkthrough with figures. **Not completed:** CNN training, SOM2IMG implementation/reproduction, feature attribution, a second dataset, or a thesis finding. All 23 tests passed (ten new synthetic tests plus 13 existing audit tests); new-code lint and notebook validation passed. Both generated figures were visually inspected. These tests establish implementation checks, not research validity.

A second run in a separate temporary directory produced byte-identical `results.json` under the recorded environment.
