# TCGA through UCSC Xena: first feasibility check

Reviewed 26 October 2026, following my advisor's recommendation on 21 October to investigate TCGA and UCSC Xena.

Subsequent work: the [Experiment 06 prediction pilot](../experiments/06_tcga_brca_pilot/README.md) is complete. This note retains the findings and open questions from the initial source inspection.

Later on the same date, all three full data files were downloaded and audited in [Experiment 05](../experiments/05_tcga_brca_audit/README.md). The verified matrix contains 60,660 expression IDs and 1,226 samples; Xena's extra listed identifier is `sampleID`. Clinical/mapping checks, paired participants and cohort-selection issues are documented there. **The remainder of this note preserves the earlier, metadata-only inspection state**, including its then-uncomputed checks; it is not the current acquisition status.

## Decision and scope

I selected GDC TCGA Breast Cancer (BRCA), STAR–TPM RNA-seq, as the next candidate for a full data audit. At this stage, the review covered the dataset pages, clinical fields and three source-description files totaling 1,560 bytes. Full data acquisition and model training had not yet begun, and acceptance for the final thesis remained undecided.

The two supplied links are complementary resources, not two independent datasets:

- [NCI TCGA program](https://www.cancer.gov/ccg/research/genome-sequencing/tcga): study provenance and the official route to the Genomic Data Commons (GDC).
- [UCSC Xena data browser](https://xenabrowser.net/datapages/): access to organized molecular/phenotype datasets from several sources, including GDC. The [Xena documentation](https://ucsc-xena.gitbook.io/project/faq/advanced-data-and-datasets) distinguishes its GDC, legacy TCGA, Pan-Cancer Atlas, and Toil hubs. Releases from these hubs may differ in processing and share participants, so they are neither interchangeable nor necessarily independent.

## Exact candidate inspected

| Item | Observation on the source page; not a full-file audit |
|---|---|
| Cohort | GDC TCGA Breast Cancer (BRCA) |
| Hub / matrix | `https://gdc.xenahubs.net` / `TCGA-BRCA.star_tpm.tsv` |
| Listed dimensions | 60,661 identifiers × 1,226 samples; genes/identifiers are rows |
| Source version / release | `05-20-2024`; metadata points to GDC release 40.0 |
| Units | `log2(tpm+1)`; do **not** apply another log transformation by default |
| Identifier example | `ENSG00000000003.15`; preserve the version suffix in provenance |
| Annotation route | `gencode.v36.annotation.gtf.gene.probemap`; metadata specifies hg38 / GENCODE release 36 |
| Source processing | Metadata describes averaging multiple measurements belonging to a sample, followed by `log2(x+1)`; exact merge granularity still needs verification |
| Download size | HTTP HEAD reports 342,300,776 compressed bytes, about 342.3 MB; matrix not downloaded |
| Companion phenotype | `TCGA-BRCA.clinical.tsv`; page lists 1,255 samples × 85 fields, version `09-07-2024`, GDC release 41.0 |

Sources: [expression details](https://xenabrowser.net/datapages/?dataset=TCGA-BRCA.star_tpm.tsv&host=https%3A%2F%2Fgdc.xenahubs.net), [clinical details](https://xenabrowser.net/datapages/?dataset=TCGA-BRCA.clinical.tsv&host=https%3A%2F%2Fgdc.xenahubs.net), [expression metadata](https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.star_tpm.tsv.json), [annotation metadata](https://gdc-hub.s3.us-east-1.amazonaws.com/download/gencode.v36.annotation.gtf.gene.probemap.json). Download URLs, hashes, file sizes and evidence categories are in the [source manifest](TCGA_XENA_SOURCE_MANIFEST.json).

These numbers are **not** 1,226 independent patients or 60,661 retained protein-coding genes. The retained cohort, unique patients, feature mapping coverage, missingness, and expression/phenotype overlap remain uncomputed.

## Rationale

This moves from Wisconsin's cell-shape measurements to actual gene-expression features. The explicit Ensembl/GENCODE mapping route is promising for later gene-group interpretation, unlike the unresolved anonymous `gene_XX` identifiers in the UCI RNA-seq candidate. It does not establish that mapping is complete or that this release can repair the UCI dataset's feature identities.

Restricting the initial audit to one breast-cancer cohort makes the data preparation manageable. The original [TCGA breast-cancer study](https://www.nature.com/articles/nature11412) provides biological context; its published results and original cohort are not results or dimensions of our present candidate. The [Xena platform paper](https://doi.org/10.1038/s41587-020-0546-8) supports the access/visualization resource, not a claim that T2I-CNN explanations are superior.

## What we learned before downloading

1. **Join by identifiers, not row position.** Expression and clinical files have different versions and listed sample counts. Audit their actual intersection and exclusions; do not assume all expression samples have usable labels or that every clinical row has expression.
2. **Group by participant before splitting.** The clinical preview already shows tumor/normal sample pairs sharing a participant prefix. GDC documents participant → sample → vial/aliquot identifiers. Confirm barcode-derived grouping against case metadata and keep all samples from a participant in one partition. [GDC barcode specification](https://docs.gdc.cancer.gov/Encyclopedia/pages/TCGA_Barcode/).
3. **Sample type is available, but class counts are unknown.** The inspected 85-field inventory includes `sample_type.samples`, `sample_type_id.samples`, `case_id`, tissue-source-site and preservation fields. No full table was audited. Codes `01` and `11` mean Primary Solid Tumor and Solid Tissue Normal respectively. Normal tissue is not automatically an independent healthy donor. [GDC sample-type codes](https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/sample-type-codes).
4. **PAM50 coverage is unresolved.** The inspected GDC clinical inventory does not list PAM50. The cohort page separately lists an ATAC-seq-associated PAM50 dataset with only 74 samples; that is not a verified label file for all 1,226 expression samples. A broader label release would need its own provenance and overlap audit.
5. **PAM50 introduces a separate evaluation issue.** The original TCGA study used a 50-gene expression-based subtype predictor. If we later predict those labels from expression and reward recovery of the defining genes, we risk circular validation. This is our design caution, not evidence of a flaw in that paper. Independent annotation/clinical evidence and explicit controls would be needed. [Original study, gene-expression analyses](https://www.nature.com/articles/nature11412).
6. **Processed does not mean fully audited.** GDC documents STAR quantification and normalized expression measures; the exact Xena release and its transformations must remain explicit. Keep expression predictors separate from labels, patient IDs and clinical covariates. [GDC mRNA pipeline](https://docs.gdc.cancer.gov/Data/Bioinformatics_Pipelines/Expression_mRNA_Pipeline/).

## Proposed first task — not yet run

Use **primary tumor versus solid-tissue normal** as a possible engineering/learning task, subject to actual label and patient counts. It is not automatically the final biological thesis question, a healthy-person screening task, or a route to proving novel disease mechanisms.

1. Download the selected expression matrix, clinical table and exact gene-map table; save URLs, retrieval dates, sizes and SHA-256 hashes. Keep patient-level tables and full matrices ignored by Git.
2. Audit schema, sample/participant counts, duplicates, missingness, sample-type balance, version overlap, gene-map ambiguity, tissue-source site, and preservation/batch metadata. Report exclusions rather than silently dropping unmatched records.
3. Freeze inclusion criteria and participant-grouped splits before scoring models. Select/filter genes, scale features, and learn layouts using training partitions only. Do not choose genes after seeing held-out performance.
4. Start with a modest, declared feature budget and tabular baseline; then reuse our fixed-grid CNN and feature-tracing checks. A full 60,661-feature CNN is not the first experiment. The feature budget and model settings will be specified in a separate protocol.
5. Only after this pipeline audit, decide with the professor whether the scientific target should remain sample type, use independently sourced clinical labels, or use a carefully framed subtype task. Evaluate gene-group meaning separately from prediction accuracy and explanation faithfulness.

This is a return to genomic **feasibility**, not a replacement of the separate-learning approach. The SOM2IMG code/version/collision questions remain open. Another Xena release of TCGA is not independent external validation; overlap must also be checked against any TCGA-derived UCI cohort.

## Source and verification notes

- Live Xena pages were inspected because the text-only browser cannot render their JavaScript content. The expression page supplied a 10 × 10 preview; no claims about all expression values are based on that preview.
- Three public sidecar descriptions were downloaded and checksummed. The clinical sidecar contains a trailing comma before its closing brace. Its original bytes were preserved; strict JSON parsing would require a separately documented normalization. The source manifest itself is strict JSON.
- The expression matrix received a size-only HTTP HEAD request. Its ETag is not treated as an independently verified SHA-256 or a complete-file integrity check.
- Reading depth: NCI overview; Xena hub-selection documentation; live dataset metadata/previews and full clinical field inventory; selected GDC pipeline/barcode/code documentation; Xena publisher preview and bibliographic record; relevant gene-expression section and bibliographic record of the TCGA breast-cancer paper. No full-paper or supplement review is claimed.

This inspection identified a gene-mapped RNA-seq candidate and the three files needed for acquisition. It also established the main checks required before modeling: release compatibility, sample labels, participant grouping and gene identities.
