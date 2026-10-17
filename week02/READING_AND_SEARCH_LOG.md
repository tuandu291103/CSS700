# Week 2 review method and unresolved questions

Date: 15 October 2026. This is a targeted evidence review, not a preregistered systematic review or a claim to have found every relevant publication.

## Search approach

Searches followed three threads: predictive comparisons; the gap between feature importance and biological interpretation; and T2I image-to-gene decoding. Named-method searches included DeepInsight, DeepFeature, DeepInsight-3D, and IGTD. Counterevidence searches covered useful tabular gene importance, conditional random-forest importance, and saliency sanity checks. Dataset discovery used the official GEO and UCI records for the shortlisted accessions.

Representative queries used in this session included:

- `Why do tree-based models still outperform deep learning typical 2022`
- `DeepFeature feature selection nonimage convolutional neural network`
- `DeepInsight-3D pathway-centric context discovered gene sets`
- `Evolutionarily informed machine learning gene-to-phenotype`
- `GSE45827`, restricted to NCBI, followed by sample-processing metadata
- `GSE13159 GEO 2096`
- `GSE2034 286`

The review favored original papers, authors' code, and official accession records. Secondary search hits were used for discovery, not as substitutes for primary evidence in the claim table. No full copyrighted articles were transcribed into the notes. One open-access institutional PDF was saved unchanged for reading.

## Reading depth and access limits

- The reference library distinguishes section-level reading from abstract-level screening and author-code documentation.
- DeepFeature's relevant PDF pages were rendered and visually checked, including the complete Table 1, Figure 2 caption, and decoder definition. We did not infer values by guessing from an image.
- Some GEO series pages intermittently returned a browser verification page. Their indexed primary records supplied metadata; a sample-level GEO processing record was directly readable. This does not substitute for a future complete metadata audit.
- IGTD's publisher page was inaccessible through this retrieval route. Its mechanism is supported here by its authors' repository. No claim is made that its numerical benchmark was fully audited.
- The entire current literature through October 2026 has not been exhaustively covered. Do not use this review alone to claim novelty or current universal state-of-the-art performance.
- The original proposal and SOM2IMG manuscript were preserved as historical context, not treated as instructions overriding the latest user scope. The previously mentioned IEEE article was not re-audited in this Week 2 review, and no new claim relies on its inaccessible final full text.

## Items to resolve before experiments

1. **Scope:** Confirm that the intended contribution is an evaluation of biological group information, rather than claiming the first image-to-gene tracing method.
2. **D1 preprocessing:** Clarify the deposited sample-type-aware corrections; decide whether a valid raw-data route is affordable.
3. **D2 feasibility:** Inspect independent sample IDs, labels, annotation, center/batch metadata, and preprocessing before accepting MILE.
4. **D3 identifiers:** Recover and verify the exact UCI-to-source gene mapping, or replace the candidate for biological evaluation.
5. **Fair biological comparison:** Choose a knowledgebase release and a matched group-analysis pipeline for both model families. Separate annotation used to construct a method from annotation used to evaluate it.
6. **Noise claim:** Decide whether a controlled noise study is in scope at all. If so, define the noise process prospectively.
7. **DeepFeature reproduction:** Reconcile its cohort-count discrepancy against a release manifest if its cohort or full method is used later.

These are next-milestone decisions. They do not block delivery of the present evidence-and-candidate report, and they have not been silently resolved by assumption.
