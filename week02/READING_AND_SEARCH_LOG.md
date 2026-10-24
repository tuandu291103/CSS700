# Week 2 review method and unresolved questions

Initial review: 15 October 2026; dataset-audit and focused SOM2IMG follow-ups: 19 October. This is a targeted evidence review, not a preregistered systematic review or a claim to have found every relevant publication.

## Search approach

**23 October explanation-learning follow-up:** opened the original arXiv records for Grad-CAM (`1610.02391`), IG (`1703.01365`), SHAP (`1705.07874`), LIME (`1602.04938`) and sanity checks (`1810.03292`). Reading depth is abstract-level. Inspected the official Captum IG/LayerGradCam, SHAP KernelExplainer and LIME tabular API documentation to select explicit settings and interpret baselines, numerical residuals and local coefficients. This supports separate learning exercises, not an exhaustive explanation literature review or a novelty claim. Links and reading limits are in the [reference library](../references/REFERENCES.md#explanation-learning-additions--23-october-2026).

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
- The original proposal and SOM2IMG manuscript were preserved as historical context, not treated as instructions overriding the latest user scope. SOM2IMG was not re-audited in the initial 15 October review. The focused 19 October review below now examines the supplied six-page manuscript; no claim relies on inaccessible final IEEE full text.

## Items to resolve before experiments

1. **Scope:** Confirm that the intended contribution is an evaluation of biological group information, rather than claiming the first image-to-gene tracing method.
2. **D1 preprocessing:** Clarify the deposited sample-type-aware corrections; decide whether a valid raw-data route is affordable.
3. **D2 feasibility:** The 19 October audit verifies sample-accession completeness, labels, and platform mapping ambiguity. Patient/center linkage, specimen-type confounding controls, full processing provenance, and expression-level validation remain open; see the [audit](GSE13159_FEASIBILITY.md).
4. **D3 identifiers:** Recover and verify the exact UCI-to-source gene mapping, or replace the candidate for biological evaluation.
5. **Fair biological comparison:** Choose a knowledgebase release and a matched group-analysis pipeline for both model families. Separate annotation used to construct a method from annotation used to evaluate it.
6. **Noise claim:** Decide whether a controlled noise study is in scope at all. If so, define the noise process prospectively.
7. **DeepFeature reproduction:** Reconcile its cohort-count discrepancy against a release manifest if its cohort or full method is used later.

These are next-milestone decisions. They do not block delivery of the present evidence-and-candidate report, and they have not been silently resolved by assumption.

## 19 October follow-up: MILE feasibility

Followed the GSE13159 record to the original MILE study (PMID 20406941), its cited PQN/DQN normalization method (PMID 16889801), a sample-column description (GSM330048), and the GPL570 annotation. Queries included `PMC5569671 normalization`, `GSE13159 18 classes`, and `GPL570 Gene Symbol Gene ID`; third-party tables were not used to establish our counts.

Downloaded the complete official series/sample attribute exports and compressed platform annotation. Exact-input hashes and computed results are in [GSE13159_AUDIT.json](GSE13159_AUDIT.json). A timed-out partial sample transfer was discarded from the analysis; a completed retry matched the series inventory. Expression tables were not downloaded. No full raw archive was requested.

Direct article pages intermittently returned browser checks; the Europe PMC full-text service returned HTTP 500. Read the indexed primary MILE abstract and selected method sections, and the PQN/DQN abstract. Full supplements and normalization specifications remain to be reviewed. These access limits constrain the preprocessing and independence conclusions; they do not invalidate the directly downloaded metadata counts.

The audit revealed a strong specimen-type/label association and an annotation snapshot dated 2016 with substantial unmapped/multi-gene probe content. These are feasibility findings, not evidence of a new biological mechanism or proof that T2I explains data better.

### Same-day continuation: original methods and acceptance protocol

The earlier access limit was partly resolved: direct primary-article HTML was retrieved and Methods, Table 1, relevant Results, and the embedded Appendix (including Tables A1–A3) were inspected. A complete patient/center linkage file was not recovered. The local reading copy and its hash are documented in the [audit follow-up](GSE13159_FEASIBILITY.md#follow-up-original-study-and-two-profile-check); the original temporarily downloaded copy disappeared after the interrupted session, so a fresh copy was saved locally before verification.

Examined the normalization section of US20100130372A1 as a primary technical disclosure, explicitly distinguished from a peer-reviewed paper and from verified release implementation. Reviewed GSE122511's indexed primary description as linked reanalysis, not as independent data or original processing evidence. Full Liu 2006 methods and other possible external supplements remain unreviewed. One publisher retrieval was not executed because the approval service reported a usage-limit failure; no access restriction was bypassed.

Selected the first CLL and MDS accessions in the already-audited inventory before downloading their full individual tables. The two-file schema/value audit is purposive and is not a representative estimate of cohort-wide missingness. No full-cohort matrix, raw CEL archive, model fitting, or tuning was performed. The [acceptance protocol](GSE13159_ACCEPTANCE_PROTOCOL.md) separates our proposed controls and acceptance rules from verified observations and from decisions still requiring advisor review.

## 19 October continuation: focused SOM2IMG review

Read the existing local six-page anonymous manuscript in full, extracted its text with pypdf, and rendered/visually inspected all six pages, especially Equations 1–2, Tables I–II and Figures 1–3. The [review](SOM2IMG_REVIEW.md) records its exact hash and local location. Neither the source file nor the historical archive was modified, and no additional reading PDF was added to Git.

Verified the final IEEE bibliographic identity through the author's faculty publication page; the linked IEEE full text remained inaccessible. An indexed OpenReview version was found but direct retrieval returned a browser check. No access control was bypassed, and these versions were not assumed interchangeable. Queries included `"SOM2IMG" tabular image`, `"SOM2IMG" github`, and `site:github.com "SOM2IMG"`. No author implementation was located in the supplied manuscript or those bounded searches; this is not an exhaustive absence claim.

Cross-checked seven corresponding UCI dataset records and the Mendeley Dengue/Chikungunya record. These checks expose release/encoding questions, including Lung Cancer dimensions and the Dry Bean sample count. The exact author releases and preprocessing were not audited. Heart and Soul access and target were not established. No benchmark data was downloaded or trained on.

The review separates qualitative neighborhood interpretation from quantified explanation quality and flags BMU collision handling and fold-local fitting as questions, not demonstrated implementation failures. Candidate methods/outcomes and a short Wednesday discussion list are proposals for review, not a frozen experiment protocol or a new biological result.
