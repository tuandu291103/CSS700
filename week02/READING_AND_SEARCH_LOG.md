# Week 2 review method and unresolved questions

Initial review: 15 October 2026; dataset-audit and focused SOM2IMG follow-ups: 19 October. I used a targeted review to examine the thesis motivation, identify relevant methods and assess candidate datasets. The search was not preregistered or exhaustive.

## Search approach

**2 November attribution follow-up:** rechecked the IG and saliency sanity-check arXiv records at abstract level, and the official Captum IG and Occlusion API definitions. The implementation review covered baseline broadcasting, the input-minus-reference factor, numerical completeness, patch shapes and batched replacement. [Experiment 08](../experiments/08_som2img_attributions/README.md) records the study-specific targets, comparisons and limitations. This was a targeted implementation review, not a full-paper expansion of the related-work search.

**31 October author-code follow-up:** inspected `https://github.com/Kerbash/tab2img-bench`. Browser retrieval failed, but a read-only Git clone succeeded. Pinned commit `dc0ef691db733ebcba4d13b53ca58e258f76ef0a`; read the README, license, package/dependency declarations, SOM/base/random/grid/encoding/evaluation implementations, grouping helper and upstream tests. Executed its 18 tests and a separate controlled Wisconsin exercise; the final IEEE full text and full 51-dataset benchmark remain outside this review. The newer repository's title/scope, simplified comparators, spillover policy and explicit numeric-export caveat are separated from the older manuscript. [Experiment 07](../experiments/07_som2img_layout/README.md) records all source hashes and local adaptations.

**26 October genomic pilot implementation:** rechecked official scikit-learn guidance on training-only feature selection/scaling, balanced class weights in logistic regression/random forest, and balanced-accuracy definition; read PyTorch's weighted cross-entropy definition. Used these as implementation/metric sources, not evidence that our settings are optimal. Links, choices, measured results and limitations are in [Experiment 06](../experiments/06_tcga_brca_pilot/README.md). No new full-paper review or biological interpretation claim was added.

**26 October TCGA/Xena follow-up:** followed my advisor's recommendation to investigate TCGA and UCSC Xena. Inspected live Xena GDC BRCA expression/clinical pages and the complete clinical field list; checked exact metadata sidecars and matrix size. Consulted primary Xena hub-selection documentation, GDC pipeline/barcode/sample-code references, the Xena publisher preview and the relevant TCGA breast-cancer gene-expression section. This initial inspection assessed source availability and metadata, before acquisition or acceptance of a cohort. Details and reading boundaries: [TCGA/Xena feasibility](../docs/TCGA_XENA_FEASIBILITY.md).

**Later 26 October acquisition follow-up:** downloaded and audited the complete three data tables in [Experiment 05](../experiments/05_tcga_brca_audit/README.md). Read the complete rendered Xena expression-identifier inventory and verified its gene-ID digest against the file; `sampleID` explains the extra listed identifier. Read [current GDC annotation guidance](https://docs.gdc.cancer.gov/Encyclopedia/pages/Annotations/) and reviewed the clinical source's annotation fields. The historical TCGA annotation page explicitly says it is not the current standard, so it was not used as a mandatory exclusion rule. Our conservative cohort option is a declared research choice, not a rule attributed to GDC. The follow-up was limited to data acquisition, audit and cohort-policy design.

**23 October explanation-learning follow-up:** opened the original arXiv records for Grad-CAM (`1610.02391`), IG (`1703.01365`), SHAP (`1705.07874`), LIME (`1602.04938`) and sanity checks (`1810.03292`). Reading depth is abstract-level. Inspected the official Captum IG/LayerGradCam, SHAP KernelExplainer and LIME tabular API documentation to select explicit settings and interpret baselines, numerical residuals and local coefficients. This supports separate learning exercises, not an exhaustive explanation literature review or a novelty claim. Links and reading limits are in the [reference library](../references/REFERENCES.md#explanation-learning-additions--23-october-2026).

Searches followed three threads: predictive comparisons; the gap between feature importance and biological interpretation; and T2I image-to-gene decoding. Named-method searches included DeepInsight, DeepFeature, DeepInsight-3D, and IGTD. Counterevidence searches covered useful tabular gene importance, conditional random-forest importance, and saliency sanity checks. Dataset discovery used the official GEO and UCI records for the shortlisted accessions.

Representative search queries included:

- `Why do tree-based models still outperform deep learning typical 2022`
- `DeepFeature feature selection nonimage convolutional neural network`
- `DeepInsight-3D pathway-centric context discovered gene sets`
- `Evolutionarily informed machine learning gene-to-phenotype`
- `GSE45827`, restricted to NCBI, followed by sample-processing metadata
- `GSE13159 GEO 2096`
- `GSE2034 286`

The review favored original papers, authors' code, and official accession records. Secondary search hits were used for discovery, not as substitutes for primary evidence in the claim table. One open-access institutional PDF was retained unchanged as a reading copy; the notes summarize the evidence and link to its sources.

## Reading depth and access limits

- The reference library distinguishes section-level reading from abstract-level screening and author-code documentation.
- DeepFeature's relevant PDF pages were rendered and visually checked, including the complete Table 1, Figure 2 caption, and decoder definition. Values were checked against the rendered source.
- Some GEO series pages intermittently returned a browser verification page. Their indexed primary records supplied metadata; a sample-level GEO processing record was directly readable. This does not substitute for a future complete metadata audit.
- IGTD's publisher page was inaccessible through this retrieval route. Its mechanism is supported here by its authors' repository. No claim is made that its numerical benchmark was fully audited.
- The entire current literature through October 2026 has not been exhaustively covered. Consequently, novelty and current state-of-the-art claims require a broader review.
- The original proposal and SOM2IMG manuscript were retained as background for the revised research plan. SOM2IMG was not re-audited in the initial 15 October review. The focused 19 October review below now examines the supplied six-page manuscript; no claim relies on inaccessible final IEEE full text.

## Items to resolve before experiments

1. **Scope:** Confirm that the intended contribution is an evaluation of biological group information, rather than claiming the first image-to-gene tracing method.
2. **D1 preprocessing:** Clarify the deposited sample-type-aware corrections; decide whether a valid raw-data route is affordable.
3. **D2 feasibility:** The 19 October audit verifies sample-accession completeness, labels, and platform mapping ambiguity. Patient/center linkage, specimen-type confounding controls, full processing provenance, and expression-level validation remain open; see the [audit](GSE13159_FEASIBILITY.md).
4. **D3 identifiers:** Recover and verify the exact UCI-to-source gene mapping, or replace the candidate for biological evaluation.
5. **Fair biological comparison:** Choose a knowledgebase release and a matched group-analysis pipeline for both model families. Separate annotation used to construct a method from annotation used to evaluate it.
6. **Noise claim:** Decide whether a controlled noise study is in scope at all. If so, define the noise process prospectively.
7. **DeepFeature reproduction:** Reconcile its cohort-count discrepancy against a release manifest if its cohort or full method is used later.

These questions define the next stage of dataset and evaluation design; the current review leaves them open.

## 19 October follow-up: MILE feasibility

Followed the GSE13159 record to the original MILE study (PMID 20406941), its cited PQN/DQN normalization method (PMID 16889801), a sample-column description (GSM330048), and the GPL570 annotation. Queries included `PMC5569671 normalization`, `GSE13159 18 classes`, and `GPL570 Gene Symbol Gene ID`; third-party tables were not used to establish our counts.

Downloaded the complete official series/sample attribute exports and compressed platform annotation. Exact-input hashes and computed results are in [GSE13159_AUDIT.json](GSE13159_AUDIT.json). A timed-out partial sample transfer was discarded from the analysis; a completed retry matched the series inventory. Expression tables were not downloaded. No full raw archive was requested.

Direct article pages intermittently returned browser checks; the Europe PMC full-text service returned HTTP 500. Read the indexed primary MILE abstract and selected method sections, and the PQN/DQN abstract. Full supplements and normalization specifications remain to be reviewed. These access limits constrain the preprocessing and independence conclusions; they do not invalidate the directly downloaded metadata counts.

The audit revealed a strong specimen-type/label association and an annotation snapshot dated 2016 with substantial unmapped/multi-gene probe content. These are feasibility findings, not evidence of a new biological mechanism or proof that T2I explains data better.

### Same-day continuation: original methods and acceptance protocol

The earlier access limit was partly resolved: direct primary-article HTML was retrieved and Methods, Table 1, relevant Results, and the embedded Appendix (including Tables A1–A3) were inspected. A complete patient/center linkage file was not recovered. The local reading copy and its hash are documented in the [audit follow-up](GSE13159_FEASIBILITY.md#follow-up-original-study-and-two-profile-check); a fresh local copy was retrieved before verification because the earlier temporary copy was no longer available.

Examined the normalization section of US20100130372A1 as a primary technical disclosure, explicitly distinguished from a peer-reviewed paper and from verified release implementation. Reviewed GSE122511's indexed primary description as linked reanalysis, not as independent data or original processing evidence. Full Liu 2006 methods and other possible external supplements remain unreviewed. A further publisher retrieval could not be completed during this review.

Selected the first CLL and MDS accessions in the already-audited inventory before downloading their full individual tables. The two-file schema/value audit is purposive and is not a representative estimate of cohort-wide missingness. No full-cohort matrix, raw CEL archive, model fitting, or tuning was performed. The [acceptance protocol](GSE13159_ACCEPTANCE_PROTOCOL.md) separates our proposed controls and acceptance rules from verified observations and from decisions still requiring advisor review.

## 19 October continuation: focused SOM2IMG review

Read the existing local six-page anonymous manuscript in full, extracted its text with pypdf, and rendered/visually inspected all six pages, especially Equations 1–2, Tables I–II and Figures 1–3. The [review](SOM2IMG_REVIEW.md) records its exact hash and local location. Neither the source file nor the historical archive was modified, and no additional reading PDF was added to Git.

Verified the final IEEE bibliographic identity through the author's faculty publication page; the linked IEEE full text remained inaccessible. An indexed OpenReview version was found but direct retrieval returned a browser check. The versions were therefore recorded separately. Queries included `"SOM2IMG" tabular image`, `"SOM2IMG" github`, and `site:github.com "SOM2IMG"`. No author implementation was located in the supplied manuscript or these targeted searches; this search result does not establish that no implementation existed.

Cross-checked seven corresponding UCI dataset records and the Mendeley Dengue/Chikungunya record. These checks expose release/encoding questions, including Lung Cancer dimensions and the Dry Bean sample count. The exact author releases and preprocessing were not audited. Heart and Soul access and target were not established. No benchmark data was downloaded or trained on.

The review separates qualitative neighborhood interpretation from quantified explanation quality and flags BMU collision handling and fold-local fitting as questions, not demonstrated implementation failures. Candidate methods/outcomes and a short Wednesday discussion list are proposals for review, not a frozen experiment protocol or a new biological result.
