# Week 2 reference library

Reviewed 15 October 2026. Stable citation keys are in [references.bib](references.bib). Locators identify the material used; “reviewed” does not imply a reproduction or exhaustive review of every supplement.

## R1

L. Grinsztajn, E. Oyallon, and G. Varoquaux. **Why do tree-based models still outperform deep learning on typical tabular data?** NeurIPS 35, Datasets and Benchmarks, 2022.

- [Official proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html); [author preprint](https://arxiv.org/abs/2207.08815).
- Key: `grinsztajn2022trees`. Reviewed abstract and introduction; benchmark details are not independently reproduced.
- Use: scoped motivation for strong tabular baselines, not a universal comparison of every T2I method or a genomics-specific result. The preprint title omits “typical”; the proceedings title is used here.

## R2

A. Sharma, E. Vans, D. Shigemizu, K. A. Boroevich, and T. Tsunoda. **DeepInsight: A methodology to transform a non-image data to an image for convolution neural network architecture.** Scientific Reports 9, 11399, 2019. DOI: `10.1038/s41598-019-47765-6`.

- [Publisher](https://www.nature.com/articles/s41598-019-47765-6).
- Key: `sharma2019deepinsight`. Reviewed Results, Table 2 discussion, and Methods / DeepInsight pipeline.
- Use: reported counterexample to blanket predictive inferiority; similarity layout and collision mechanism. No claim of universal superiority.

## R3

A. Sharma, A. Lysenko, K. A. Boroevich, E. Vans, and T. Tsunoda. **DeepFeature: feature selection in nonimage data using convolutional neural network.** Briefings in Bioinformatics 22(6), bbab297, 2021. DOI: `10.1093/bib/bbab297`.

- [Publisher](https://academic.oup.com/bib/article/22/6/bbab297/6343526); [author code](https://github.com/alok-ai-lab/deepfeature).
- Key: `sharma2021deepfeature`. Reviewed relevant results/methods; visually checked Table 1, Figure 2, Materials and methods, and Element decoder, printed pages 5-8.
- Open-access [reading copy](papers/DeepFeature_2021.pdf), obtained from [Griffith repository](https://research-repository.griffith.edu.au/server/api/core/bitstreams/219755e3-3e8b-4b01-9193-a8660d2cb3bc/content). The added repository cover shifts PDF page numbers by one.
- SHA-256: `c72272a10415cd5f32c4983fa2d6546f5b35c4d37fd6baa4e80d2c8cc1da47a4`.
- Use: close prior art; see the evidence review for the unresolved cohort-count discrepancy. Unequal selected-gene counts constrain interpretation of enrichment comparisons.

## R4

A. Sharma, A. Lysenko, K. A. Boroevich, and T. Tsunoda. **DeepInsight-3D architecture for anti-cancer drug response prediction with deep-learning on multi-omics.** Scientific Reports 13, 2483, 2023. DOI: `10.1038/s41598-023-29644-3`.

- [Publisher](https://www.nature.com/articles/s41598-023-29644-3).
- Key: `sharma2023deepinsight3d`. Reviewed “Feature selection to identify genes of interest,” “Pathway-centric context of discovered gene sets,” and related methods.
- Use: additional biological interpretation precedent. Sparse Reactome findings and subsequent IPA analysis illustrate sensitivity to dataset and knowledgebase; enrichment is not causal validation.

## R5

C. Strobl, A.-L. Boulesteix, T. Kneib, T. Augustin, and A. Zeileis. **Conditional variable importance for random forests.** BMC Bioinformatics 9, 307, 2008. DOI: `10.1186/1471-2105-9-307`.

- [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC2491635/).
- Key: `strobl2008conditional`. Reviewed Abstract and Background, with methods context.
- Use: distinguish marginal from conditional importance under correlated predictors. Do not generalize its particular random-forest results to every attribution algorithm.

## R6

C.-Y. Cheng et al. **Evolutionarily informed machine learning enhances the power of predictive gene-to-phenotype relationships.** Nature Communications 12, 5627, 2021. DOI: `10.1038/s41467-021-25893-w`.

- [Publisher](https://www.nature.com/articles/s41467-021-25893-w).
- Key: `cheng2021evolutionary`. Reviewed Abstract, Figure 1 explanation, and candidate-selection/functional-validation sections.
- Use: a counterexample to “traditional rankings are useless.” The application is plant nitrogen-use efficiency, not human cancer; rankings were contextualized and experimentally tested, not assumed causal on their own.

## R7

Y. Zhu et al. **Converting tabular data into images for deep learning with convolutional neural networks.** Scientific Reports 11, 11325, 2021. DOI: `10.1038/s41598-021-90923-y`.

- [Publisher](https://www.nature.com/articles/s41598-021-90923-y); [author implementation](https://github.com/zhuyitan/IGTD).
- Key: `zhu2021igtd`. Publisher retrieval failed during this review; the author repository's Description, Uniqueness, and Components were inspected. No numerical paper result is used.
- Use: the documented one-feature-per-pixel similarity-layout mechanism. Re-read the paper's detailed protocol before implementing a reproduction.

## R8

J. Adebayo, J. Gilmer, M. Muelly, I. Goodfellow, M. Hardt, and B. Kim. **Sanity Checks for Saliency Maps.** NeurIPS 31, 2018.

- [Author preprint](https://arxiv.org/abs/1810.03292).
- Key: `adebayo2018sanity`. Abstract and stated randomization-check rationale reviewed; no method-by-method empirical failure claim is made here.
- Use: why visual appearance alone does not establish explanation validity. It does not show that every saliency method always fails.

## R9

A. Subramanian et al. **Gene set enrichment analysis: A knowledge-based approach for interpreting genome-wide expression profiles.** PNAS 102(43), 15545-15550, 2005. DOI: `10.1073/pnas.0506580102`.

- [Publisher](https://www.pnas.org/doi/10.1073/pnas.0506580102).
- Key: `subramanian2005gsea`. Reviewed Abstract, Introduction, and Methods overview.
- Use: gene-set context is not unique to image representations. Ranked-list GSEA and a Fisher-test overrepresentation analysis of a selected set are different procedures; do not conflate them in a future protocol.

## Dataset records

| ID / BibTeX key | Primary record | Locators used |
|---|---|---|
| D1 / `geoGSE45827` | [GSE45827](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE45827) | Overall design, Platforms, Samples, download routes |
| D1 processing / `geoGSM1116110` | [GSM1116110](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM1116110) | Characteristics, Data processing, Data table row count |
| D2 / `geoGSE13159` | [GSE13159](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE13159) | Study description, Samples, Platforms, Relations, download routes |
| D3 / `fiorini2016rnaseq` | [UCI dataset 401](https://doi.org/10.24432/C5R88H) | Instances, Features, Variable Information, citation, license |
| Reserve / `geoGSE2034` | [GSE2034](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE2034) | Summary, sample count, platform, outcome description |

GEO series records were available through indexed primary-page content where direct access intermittently returned a browser check. No raw expression matrix or complete clinical metadata audit was performed. Original associated studies should also be read and cited when a dataset is accepted; repository metadata alone is not a full methodological review.
