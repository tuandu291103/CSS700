# Draft thesis motivation

15 October 2026 | Draft for advisor feedback, not a statement of achieved results

Tree-based models remain strong competitors on many tabular prediction tasks, as shown in a benchmark of 45 datasets [R1]. However, their advantage should not be treated as universal: tabular-to-image methods have reported competitive or superior performance against the particular tabular comparators in their studies [R2]. Consequently, this thesis will measure predictive performance rather than assume that either representation must win.

For gene-expression analysis, the research question extends beyond predicting a label to understanding relevant genes and their biological context. Correlated predictors can complicate certain feature-importance measures [R5], while gene-set analysis provides a way to interpret expression patterns at a functional-group level [R9]. Traditional machine-learning explanations can also be useful when gene rankings are interpreted alongside biological context and experimental validation [R6].

Spatial encodings offer another representation of feature relationships. DeepInsight and IGTD arrange features using similarity [R2, R7], and prior work already connects CNN activation regions to genes and pathway analysis [R3, R4]. Building on that foundation, the proposed study will evaluate whether spatially organized explanations provide stable, biologically coherent gene-group information beyond tabular explanations given comparable downstream biological annotation. It will report predictive tradeoffs and mapping ambiguity separately. Visual plausibility alone will not be treated as validation [R8]. Any benefit under noisy conditions remains a hypothesis requiring an explicit noise model and controlled evaluation; pathway associations will not be described as causal discoveries.

## References for the draft

- R1: [Grinsztajn et al., 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html).
- R2: [Sharma et al., DeepInsight, 2019](https://doi.org/10.1038/s41598-019-47765-6).
- R3: [Sharma et al., DeepFeature, 2021](https://doi.org/10.1093/bib/bbab297).
- R4: [Sharma et al., DeepInsight-3D, 2023](https://doi.org/10.1038/s41598-023-29644-3).
- R5: [Strobl et al., 2008](https://doi.org/10.1186/1471-2105-9-307).
- R6: [Cheng et al., 2021](https://doi.org/10.1038/s41467-021-25893-w).
- R7: [Zhu et al., IGTD, 2021](https://doi.org/10.1038/s41598-021-90923-y).
- R8: [Adebayo et al., 2018](https://arxiv.org/abs/1810.03292).
- R9: [Subramanian et al., 2005](https://doi.org/10.1073/pnas.0506580102).

## Writing rules for subsequent thesis sections

I will use the following conventions when developing the thesis:

1. Attach a reference to the specific empirical or technical claim it supports, not merely to the end of a long paragraph containing unrelated claims.
2. State a study's task, comparator, and evaluation context when reporting its scores. Never present published scores as our results.
3. Distinguish inferences, hypotheses, study decisions and measured results. Cite external evidence for claims drawn from prior work, and link original results to the experiment that produced them.
4. Prefer the original research paper or dataset record over a review, blog, or uncited assertion. Read the relevant methods and limitations before relying on a result.
5. Use “gene-expression” or “transcriptomic” for the proposed datasets. Keep conclusions within that scope unless additional evidence supports extending them to other genomic modalities or noisy tabular data.
6. Retain contradictory findings and unresolved source discrepancies in the evidence log.
