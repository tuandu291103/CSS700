# SOM2IMG: focused reference and dataset review

19 October 2026 | Week 2 initial study | Literature review, not a reproduction

## Bottom line

SOM2IMG is a **transformation method evaluated on nine datasets**, not the name of a single dataset. The supplied manuscript motivates arranging similar feature profiles in a grid, but it does not report a gene-expression benchmark or quantify improved biological interpretation. It is a candidate comparison method for our thesis, not evidence that our hypothesis has already been established. [S1, Table I, Sections III, V.C and VI]

## Sources and version boundary

- **S1 — text actually reviewed:** the locally supplied [six-page manuscript](/Users/dudinh/Desktop/SOM2IMG.pdf), titled *SOM2IMG: Self-Organizing Map Based Tabular-to-Image Transformation for Deep Learning*, with an anonymous-author heading. Read all six pages and visually checked the equations, Tables I–II, and Figures 1–3. Page locators below mean PDF pages 1–6. Size: 1,467,643 bytes; SHA-256: `b80c6d67ef05ea7ec806848c102b0b17bd3d0535f90fdcc673e4268031da18f9`. Its PDF metadata records creation on 8 May 2026; that is not a verified publication date. It remains local and preserved in the [existing archive](../docs/ARCHIVE.md); no extra manuscript copy is being published to GitHub.
- **S2 — published bibliographic identity:** W. Selke and W. Kim, IEEE IRI 2026, pp. 171–176, DOI [10.1109/IRI69576.2026.00042](https://doi.org/10.1109/IRI69576.2026.00042), verified against the [author's publication list](https://faculty.washington.edu/kimw6/publication.html). Its link resolves to the professor's [IEEE record](https://ieeexplore.ieee.org/abstract/document/11676815), whose content could not be retrieved. **S1 has not been verified as identical to the final IEEE article.**
- **Other version located, not reviewed:** an [OpenReview manuscript](https://openreview.net/pdf?id=DU5m5X7yu8) is indexed under the same title with an ICLR 2026 submission heading. Direct retrieval returned a browser check. We do not merge its results with S1 or infer its publication status from the title match.

Technical observations below concern **S1 only**. The bibliography keeps its reading-copy key (`som2imgSuppliedManuscript2026`) separate from the published citation (`selke2026som2img`).

## 1. What the method does

For a sample-by-feature matrix, the method transposes the data: each feature is represented by its values across samples. A self-organizing map learns a two-dimensional layout of these feature vectors. Each feature receives the coordinates of its closest map unit, its **best matching unit (BMU)**. Each sample then supplies values at those coordinates; unused pixels are zero-filled. Equation 1 specifies `h = ceil(sqrt(d))` and `w = ceil(d/h)` for `d` features. [S1, Section III.B, p. 2]

This is **feature grouping**, not patient clustering. Nearby features reflect the representation and distance used to learn the layout; proximity alone does not demonstrate a biological interaction. The latter is our interpretation of the mapping's limits, not a reported biological finding.

Two implementation questions matter:

1. **One feature per pixel is not established by the BMU equation alone.** Independent nearest-unit assignments do not impose distinct coordinates: two features can choose the same unit even if the grid has enough cells. This is a mathematical observation about Equation 2, not an observed bug in author code. S1 describes direct feature placement but gives no explicit collision-resolution rule. We need the actual mapping and rule before claiming invertibility or preservation of every feature.
2. **Fold boundaries are unclear.** Section III.E and Figure 2 describe normalization/image generation first, followed by 10-fold cross-validation. They do not establish that normalization and layout learning are refitted within each training fold. This is an unresolved protocol question, **not proof of leakage**. Our proposed inductive evaluation must fit both on training rows only and reuse the frozen coordinates for held-out rows. [S1, pp. 2–4]

The phrase “original values” also needs qualification: the evaluation uses min–max normalized inputs. Stored numeric values, image export precision, and any collision aggregation must be checked separately. [S1, Sections III.B and III.E, Figure 2]

## 2. Which datasets were used?

The middle column reports **S1 Table I**, not independently audited release dimensions. The final column cross-checks official records where available. These are record-level checks; no dataset files were acquired in this session.

| Benchmark in S1 | Reported samples × features | Official-record check / suitability |
|---|---:|---|
| Breast Cancer Wisconsin | 569 × 30 | [UCI Diagnostic dataset](https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic): matching dimensions; cell-nucleus morphology derived from images, not gene expression. Candidate technical pilot only. |
| Heart Disease | 303 × 20 | [UCI](https://archive.ics.uci.edu/dataset/45/heart+disease): 303 × 13 predictors. Need exact release and encoding to explain the feature-count difference. |
| Lung Cancer | 101 × 32 | [UCI dataset 62](https://archive.ics.uci.edu/dataset/62/lung+cancer): 32 × 56 predictors, without individual attribute definitions. Dataset identity/dimensions remain unresolved; do not silently swap counts or infer a transformation. |
| Myocardial Infarction | 1,700 × 184 | [UCI](https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications): 1,700 × 111 predictors. Exact target, prediction time, and encoding need verification. |
| Obesity Levels | 2,111 × 23 | [UCI](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition): 2,111 × 16 predictors; record describes 77% synthetic data. Not 2,111 independently observed people. |
| Heart and Soul | 515 × 100 | S1 Section III.A describes a proprietary clinical cohort. Exact release, target, permissions, and dictionary were not obtained. |
| Dry Beans | 13,612 × 16 | [UCI Dry Bean](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset): 13,611 × 16 morphology features. One-row discrepancy needs a manifest, not a guessed correction. |
| Spambase | 4,601 × 57 | [UCI](https://archive.ics.uci.edu/dataset/94/spambase): matching dimensions; email word/character frequencies and capitalization summaries. Not biological data. |
| Dengue & Chikungunya | 17,172 × 26 | [Mendeley version 1](https://data.mendeley.com/datasets/bv26kznkjs/1): 17,172 records, 27 attributes including the class field; balanced through undersampling. Describes Amazonas and Recife sources, not Recife alone. Clinical/sociodemographic data, not expression. |

Categorical encoding could explain some feature-count differences, but this has **not** been verified against author preprocessing code. S1's UCI citation is to the general repository rather than accession-specific releases. Its conclusion explicitly proposes genomics as a future extension. [S1, Sections III.A and VI]

**Our selection recommendation:** ask whether “use the SOM2IMG dataset” means reusing one benchmark for a technical check or using SOM2IMG on the thesis's genomic datasets. Wisconsin Diagnostic is a plausible small technical check, **not an accepted replacement for one of the three biological datasets**. Keep the [transcriptomic shortlist](DATASET_CANDIDATES.md) conditional; this review does not accept MILE or the other candidates.

## 3. Algorithms and evaluation actually reported

S1 compares SOM2IMG with seven other named T2I approaches (DeepInsight, IGTD, REFINED, NCTD, SuperTML, GramMatrix, correlation-based mapping) and a random-layout control, using a common LeNet-inspired CNN. Tabular comparators are random forest, gradient boosting, XGBoost, RBF SVM, logistic regression, k-nearest neighbors, and an MLP. Tabular settings are described as defaults with minimal tuning. [S1, Sections II.B and III.C–E]

Reported evaluation uses F1 and 10-fold cross-validation; Figure 3 labels its error bars as 95% confidence intervals. Table II reports differences from random layout, win counts, and Cohen's d. For SOM2IMG it lists mean F1 difference **+0.021**, **9/9 wins**, and **d = 1.08**. These are **manuscript-reported predictive results**, not our results and not measurements of explanation quality. [S1, Section IV.A, Table II, Figure 3]

Before reproducing numerical claims, obtain the exact F1 averaging convention, folds/seeds, uncertainty calculation, unrounded scores, and implementation version. S1 itself cautions about only nine dataset-level observations. There is also a visible reporting inconsistency: p. 4 says SuperTML is excluded from Table II, but Table II includes its row. Record the inconsistency without silently repairing the analysis. We did not estimate precise F1 values from bar heights or recompute the reported effect sizes.

## 4. What it supports about interpretability

S1 Section V.C provides a qualitative account of feature neighborhoods, including a Wisconsin cell-morphology example. We found **no quantified user study, gene-set enrichment evaluation, attribution-faithfulness test, or explanation-stability analysis** in the six-page copy. Thus it supplies a motivation for inspecting neighborhoods, not evidence that a researcher understands predictions better or that those neighborhoods reveal genomic mechanisms.

This distinction matters because image-to-gene decoding and enrichment already occur in [DeepFeature](https://doi.org/10.1093/bib/bbab297), and gene-set interpretation is available without images through approaches such as [GSEA](https://doi.org/10.1073/pnas.0506580102). Visual plausibility alone is not sufficient validation; [saliency sanity checks](https://arxiv.org/abs/1810.03292) motivate checking whether explanations depend on the learned model. See the [claim evidence](CLAIM_EVIDENCE.md) for reading depth and limitations.

## 5. Proposed use in our thesis — not an approved experiment protocol

- **Representation comparisons:** SOM2IMG, DeepInsight, and IGTD, plus random layout as a control. Use a matched CNN and feature set where feasible; record any necessary differences. These are three candidate T2I methods, not three newly developed algorithms. [S1; DeepInsight and IGTD in the reference library]
- **Tabular comparisons:** retain strong tree-based and simple linear baselines, with comparable tuning opportunities. Compare their explanations using the **same** downstream biological annotation and gene budgets, not a plain tabular list against an extensively annotated image. [Motivation: S1 Section III.C; R1, R3 and R9 in the reference library; matching policy is our proposal.]
- **Interpretation outcomes to specify next:** mapping coverage/collision rate; gene-space top-k overlap across repeats; and biological coherence under a fixed annotation/background and matched selection budget. These are proposed measurable outcomes, not validated thresholds. More enriched sets alone does not mean a better explanation. A model-faithfulness test and model-randomization control also need a precise, sourced protocol before experiments. Predictive F1 remains a separate outcome.
- **Reproduction gate:** obtain the final authorized paper and author implementation or an adequate specification; inspect collision handling, fold-local fitting, input identifiers, and export precision. No code link was found in S1 or the bounded title/GitHub searches. This does not establish that author code is unavailable.

## Wednesday discussion notes

1. Completed the missing focused SOM2IMG reading and checked its benchmark identities against official records.
2. Its nine benchmarks do not establish a gene-expression interpretation benefit; that remains our research question, not today's discovery.
3. Clarify whether the professor expects a small benchmark reproduction in addition to three genomic datasets.
4. Request the intended final version/code and clarification of BMU collisions and fold-local preprocessing; do not assume the anonymous copy describes the final implementation.
5. Next priority: resolve the intended benchmark and method specification, then continue dataset acceptance checks. No new model training or biological result was produced today.
