# Week 2: evidence for the thesis motivation

15 October 2026 | Targeted primary-source review | No new experiments

## Main conclusion

The biological-interpretation direction is credible, but the original motivation needs qualification. We should investigate an added benefit of spatial gene-group explanations, not begin by declaring that one model family always wins or that another family's explanations have no value.

This review is purposive, not an exhaustive systematic review. Reference IDs R1-R9 resolve in the [reference library](../references/REFERENCES.md); publication locators and reading depth are recorded there.

## Claim-to-evidence table

| Proposed claim | Verdict | Evidence and scope | Defensible thesis wording |
|---|---|---|---|
| T2I plus CNN can never beat traditional tabular ML. | Reject the universal claim. | R1 benchmarks 45 datasets and finds strong tree-based performance on medium-sized tabular tasks. Conversely, DeepInsight reports RNA-seq accuracy of 99% versus 96% for its random-forest comparator (R2, Table 2). Those are published results, not our reproduction. | Tree-based models are essential competitive baselines; relative predictive performance must be measured on the selected task. |
| A traditional ML feature ranking is useless to a researcher. | Reject. | R5 identifies correlation-related problems in particular random-forest importance measures, not universal failure. R6 combines XGBoost importance with regulatory-network context and experimental validation in plants. | A ranking alone may not supply the biological context a researcher needs; its utility depends on the attribution definition, context, and validation. |
| Groups of genes can be more informative than an isolated gene list. | Supported in specific settings. | GSEA analyzes predefined gene sets and demonstrates pathway-level interpretation of expression profiles (R9, Abstract and Introduction). It does not require a T2I model. | Gene-set analysis motivates evaluating explanations at both gene and group levels, with a non-image group-analysis baseline. |
| T2I can place similar features nearby. | Supported as an encoding objective. | DeepInsight positions feature profiles by similarity (R2, Methods); IGTD assigns a distinct pixel to each feature and seeks similar-feature neighborhoods (R7, author implementation description). | Spatial proximity can represent the chosen feature-similarity measure. It is not automatically physical genomic proximity or biological interaction. |
| CNN activation regions can be traced back to genes. | Established prior work, not a new idea by itself. | DeepFeature uses CAM, region accumulation, and an element decoder; Figure 2B evaluates cancer gene-set enrichment (R3). DeepInsight-3D also performs CAM-based gene selection and pathway analysis (R4). | We build on existing image-to-gene decoding and must specify an additional, testable contribution. |
| The resulting image is easier for people to interpret. | Not established by this review. | Saliency-map appearance is insufficient validation (R8). Neither a pathway-overlap score nor our ability to see clusters measures a researcher's task performance. | We will evaluate specified explanation properties. A claim about human understanding requires a separate task-based evaluation. |
| Spatial explanations are especially reliable in very noisy data. | Open hypothesis. | R1 highlights difficulties from uninformative features; R4 reports variability and sparse pathway findings in challenging small datasets. Neither establishes universal T2I noise robustness. | Whether spatial explanations retain useful information under a defined noise process remains to be tested. |
| Back-mapping a heatmap recovers each gene's causal contribution. | Unsupported. | DeepInsight can average colliding features (R2, pipeline); DeepFeature's decoder can return a set of genes for one pixel (R3, Element decoder). Coordinate recovery and individual causal attribution are different tasks. | Traceability identifies represented features or groups; separate evidence is needed for attribution faithfulness or biological causation. |

Sources: [R1](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html), [R2](https://doi.org/10.1038/s41598-019-47765-6), [R3](https://doi.org/10.1093/bib/bbab297), [R4](https://doi.org/10.1038/s41598-023-29644-3), [R5](https://doi.org/10.1186/1471-2105-9-307), [R6](https://doi.org/10.1038/s41467-021-25893-w), [R7](https://github.com/zhuyitan/IGTD), [R8](https://arxiv.org/abs/1810.03292), [R9](https://doi.org/10.1073/pnas.0506580102).

## What exactly is the interpretability problem?

Our proposed operational problem has three parts. These are study definitions and deductions, not claims that the literature has proven our method superior.

1. **Traceability:** Which original probe or gene produced a highlighted pixel? Record the input identifier, annotation version, transformations, coordinate, and any collisions.
2. **Model explanation:** Does a highlighted group actually matter to the chosen prediction under a stated test? A lookup table answers where a gene was placed, not whether a heatmap faithfully explains the model.
3. **Biological interpretation:** Does that group connect to independently defined biological knowledge? This is distinct from predicting a label and from proving a biological mechanism.

For an illustrative collision, if one pixel stores `(a + b) / 2`, then both `(a, b) = (1, 3)` and `(2, 2)` produce the value 2. Inverting that pixel cannot uniquely recover the two values. This is our elementary mathematical illustration of the averaging mechanism described in R2, not an experimental result. Even without collisions, an accurate inverse coordinate lookup does not establish the quality of the explanation assigned to those coordinates.

Also distinguish **feature grouping** from **sample grouping**: the layout organizes gene profiles across samples; each sample supplies values at the fixed feature locations. A gene neighborhood does not, by itself, cluster patients. This distinction follows the representation described in R2 and R7.

## What this means for our contribution

DeepFeature is the closest starting reference found in this review. Its existence rules out presenting image-to-gene decoding plus pathway analysis as wholly new. Our possible contribution is a controlled evaluation of when spatial organization adds reproducible group-level information, with explicit mapping limitations. Whether that is sufficiently novel is still an advisor and broader literature-review decision.

Proposed safeguards for a later protocol, **not implementations completed this week**:

- Compare tabular explanations plus the **same** gene-set analysis against T2I explanations plus gene-set analysis. Comparing a bare ranked list to a richly annotated heatmap would mix modeling and presentation effects.
- Match gene budgets, annotation universe, pathway release, and multiple-testing procedure. Do not equate a larger number of enriched sets with an automatically better explanation.
- If pathways guide the layout, do not use those same pathways as the sole independent evidence of biological coherence. Predeclare separate evaluation knowledge or report the circularity explicitly.
- Keep mapping integrity, model faithfulness, stability, predictive utility, and biological coherence as separate outcomes. Agreement with a different predictor's ranking is not ground truth.
- Separate measurement perturbations, irrelevant features, label noise, and batch effects. A result under one is not evidence for all four.
- Use model-dependence checks motivated by R8, and a random-layout control if later testing whether neighborhood structure adds value.
- Reserve “easier to understand” for measured human-task outcomes; reserve causal or clinical claims for evidence beyond this proposed computational study.

These choices are proposed research controls derived from the identified risks. They do not require us to launch an experiment matrix during Week 2.

## Source-audit note

DeepFeature's Materials and methods states 6,280 downloaded files, 64 removed, and 6,126 remaining. The subtraction gives 6,216. This is an unresolved inconsistency in the published text, verified in both the publisher text and the institutional PDF (printed page 7). We do **not** silently repair it or treat either final count as an audited cohort. If reproducing that cohort later, verify the actual release and sample manifest.

## Week 2 acceptance status

- Evidence for the motivation and counterevidence: documented.
- Defensible draft wording: provided in [PROPOSAL_WORDING.md](PROPOSAL_WORDING.md).
- Dataset candidates, identifiers, access routes, and risks: documented.
- Evidence that our future method improves interpretation: **not yet available**.
- Dataset-level acceptance audit: **partial MILE metadata/annotation and two-profile checks completed on 19 October; full-cohort integrity, independence, processing, and final gene mapping remain unresolved**. See the [feasibility report](GSE13159_FEASIBILITY.md).
- Claims of a first-ever tracing method, guaranteed CNN inferiority, or proven noise robustness: **not adopted**.

## 19 October addition: SOM2IMG

The [focused review](SOM2IMG_REVIEW.md) now covers all six pages of the supplied anonymous manuscript, separately from the final IEEE bibliographic record. Its qualitative feature-grouping discussion supports a reason to inspect spatial layouts, not a demonstrated improvement in biological interpretation. None of its nine reported benchmarks is identified as a gene-expression study. Collision handling and fold-local layout fitting require implementation verification before reuse. These limits concern the reviewed copy, not an uninspected final publication; they neither prove a code defect nor establish our thesis's novelty.
