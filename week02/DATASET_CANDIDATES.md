# Week 2: transcriptomic dataset candidates

Started 15 October; updated 19 October 2026 | MILE metadata/annotation and two profiles audited; full-cohort expression matrices not downloaded or audited

## Selection principle

Our proposed selection criteria are: a clear prediction target; recoverable gene or probe identities; enough independent samples for the intended evaluation; inspectable preprocessing; and an annotation route for group-level interpretation. We will not select datasets because they already favor a CNN. A repository sample count is not a verified count of independent patients, and a probe count is not a count of unique genes.

The shortlist spans breast-tumor subtypes, hematological diagnoses, and multi-tumor RNA-seq classification. This is domain breadth within transcriptomics, not evidence for all genomic modalities or all noisy tabular problems.

**SOM2IMG clarification (19 October):** the supplied reference evaluates a method on nine benchmarks, none described as gene-expression data. The [focused review](SOM2IMG_REVIEW.md#2-which-datasets-were-used) records their identities and source-count discrepancies. Wisconsin Diagnostic is a possible separate technical check, not a substitute for a gene-expression cohort. None of the three candidates below was established as a SOM2IMG benchmark by that review. Confirm what the advisor means by reusing its dataset before changing the shortlist.

## Shortlist at a glance

| Candidate | Source-reported size | Proposed target | Feature identity | Current decision |
|---|---|---|---|---|
| D1: GEO GSE45827 | 155 deposited samples; proposed tumor-only subset of 130 | Four breast-tumor subtype groups | GPL570 probe IDs; gene mapping needed | Conditional: investigate preprocessing before accepting processed values |
| D2: GEO GSE13159, MILE Stage 1 | 2,096 sample records verified | 18 deposited diagnostic labels verified | GPL570 mapping ambiguity quantified; final gene mapping pending | Conditional after metadata audit; specimen-type confounding and independence/provenance gaps remain |
| D3: UCI gene expression cancer RNA-Seq | 801 samples, 20,531 feature columns | Five tumor types | Anonymous `gene_XX`; verified mapping required | Conditional: no biological claims until identifiers are resolved |

Sources: [D1 official GEO series](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE45827), [D2 official GEO series](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE13159), [D3 official UCI record](https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq).

## D1. GSE45827: breast-tumor subtype expression

**Verified source facts.** The series describes 41 triple-negative, 30 HER2, 29 Luminal A, and 30 Luminal B primary invasive tumors, plus 11 normal-tissue samples and 14 cell lines. It uses the Affymetrix HG-U133 Plus 2.0 platform, GPL570. Processed sample tables and a Series Matrix route are listed; raw CEL files are also available. [GEO series, Overall design and Platforms](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE45827).

An inspected sample record reports 29,873 processed probe rows. Its processing description includes GC-RMA, a mixed model containing batch and sample type, technical-replicate averaging, and expression filtering. That is a sample-record observation, not a full matrix-schema audit. [GSM1116110, Data processing and Data table](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM1116110).

**Why it could fit (our rationale).** A within-organ subtype task could test whether spatial groups convey information beyond a ranked list without relying entirely on differences between organs. The small proposed cohort also makes feasibility checks manageable, though it is a weak setting for confident claims about a large CNN.

**Proposed cohort.** Initially consider the 130 primary tumors only; exclude normal tissue and cell lines. Do not silently equate the series' “TN” wording with every sample-level “Basal” label: reconcile the label dictionary and associated study. The subset has not been constructed.

**Acceptance gates.** Determine exactly which deposited values received sample-type-aware adjustment and cohort-wide filtering. These procedures create a potential leakage/selection concern for a fresh prediction benchmark; their effect is not resolved by later fitting a scaler only on training rows. Prefer an auditable raw-data preprocessing route if feasible, or reject this candidate for the confirmatory benchmark. Identify patients and technical replicates before any split. Map retained probes to versioned gene identifiers and report unmapped or ambiguous cases.

**Decision:** scientifically relevant, but **not a ready-to-train clean starter dataset**. A credible preprocessing plan is required first.

## D2. GSE13159: MILE Stage 1

**Verified source facts.** The GEO record describes expression profiling of 2,096 blood or bone-marrow specimens using GPL570, for leukemia diagnosis/subclassification. It belongs to the MILE study and is a subseries of GSE13204. The record provides processed sample tables and Series Matrix access; the raw CEL archive is listed at approximately 9.3 GB. [GEO record, Summary, Overall design, Samples, and Relations](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE13159).

**Why it could fit (our rationale).** It offers a larger starting cohort than D1, multiple related diagnostic groups, and probe identifiers with an annotation route. That makes it the first candidate to investigate for an end-to-end, gene-aware comparison. Larger sample count alone does not establish suitability or remove confounding.

**19 October audit update.** All 2,096 sample accessions match the official series inventory; all have labels across 18 classes (13–448 samples per class). There are 1,556 marrow and 540 blood specimens. All 448 CLL samples are blood, while all 206 MDS samples are marrow. This association is a potential confounding concern, not proof of a model shortcut. Explicit patient linkage and collection-center fields were not found in the downloaded metadata. The 2016 platform-annotation snapshot has 42,904 single-gene, 2,214 multi-gene, and 9,557 unmapped probe rows. See the [feasibility report](GSE13159_FEASIBILITY.md) and [computed audit/source manifest](GSE13159_AUDIT.json).

**Target and dimensions not yet frozen.** The deposited label dictionary and counts are now documented, but inclusion criteria and the final analysis target remain undecided. Record actual retained probe count and unique gene count separately; metadata-reported rows are not a verified expression matrix. The combined non-leukemia/healthy label must not be simplified to healthy controls.

**Follow-up.** Reviewed original-study methods and the embedded appendix; no complete patient/center linkage map was recovered. Two preselected profiles have exact GPL570 ID coverage, each with 54,630 finite DQN3 values and 45 missing control values. This does not establish full-cohort integrity. The [draft acceptance protocol](GSE13159_ACCEPTANCE_PROTOCOL.md) now defines specimen-type controls and explicit limits on acceptable claims.

**Acceptance gates.** Audit independent patient IDs, duplicate specimens, class imbalance, collection-center/batch metadata, and preprocessing. Determine whether a center-aware evaluation is feasible from the available metadata; do not assume it is. Confirm that training and evaluation data can be processed without outcome-informed transformations or fitting new preprocessing on held-out samples. Map probes with a documented many-to-one/one-to-many policy.

**Access and independence.** Start with metadata and processed-data provenance, not the large raw archive. The parent accession and this subseries are not two independent studies. Freeze a sample manifest before adding any other MILE-derived dataset.

**Decision:** **conditional candidate after bounded metadata/annotation and two-profile checks**, not accepted for confirmatory experiments. Proposed controls and mapping policy are documented for review; patient linkage and full processing verification remain unresolved. No full-cohort expression matrix, raw CEL archive, or model training has been started.

## D3. UCI gene expression cancer RNA-Seq

**Verified source facts.** UCI describes an Illumina HiSeq PANCAN subset with 801 samples, 20,531 columns, and five labels: BRCA, KIRC, COAD, LUAD, and PRAD. Its columns have dummy `gene_XX` names; the record points to Synapse `syn4301332` and states that feature order follows the original submission. The release is 69.5 MB and marked CC BY 4.0. DOI: `10.24432/C5R88H`. [UCI record, Dataset Information, Variable Information, and License](https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq).

**Why it could fit (our rationale).** It provides an RNA-seq contrast to the microarray candidates and a compact distribution for an eventual technical pilot. Its convenience must not take priority over interpretable gene identities.

**Hard identifier gate.** Obtain the exact source-version column list, verify ordering and dimensions against the UCI release, and store a checksummed mapping from each `gene_XX` to a stable identifier. Never guess gene names by column number or attach a generic list from a different PANCAN release. If mapping cannot be verified, this dataset can support only an anonymous-feature technical pilot, **not** the thesis's gene/pathway interpretation claims; select a replacement for the biological study.

**Additional gates.** Audit donor IDs, preprocessing and class counts. Evaluate tissue-of-origin effects as an alternative explanation for biological groups. Do not count a larger TCGA cohort from DeepInsight/DeepFeature as independent validation without checking patient overlap.

**Decision:** conditional candidate. The identifier map has **not** been recovered this week.

## Reserve, not a fourth committed experiment: GSE2034

The GEO series describes 286 lymph-node-negative breast-cancer patients, with 180 relapse-free and 106 developing distant metastasis, on GPL96. It provides a prognostic endpoint rather than a simple tumor-type task. [GEO GSE2034](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE2034).

Our rationale for retaining it as a reserve is its within-disease outcome and identifiable microarray probes. However, event status must not be treated as a uniform fixed-time outcome without auditing follow-up and censoring. A survival or carefully defined time-horizon protocol would add scope. Use only after confirming endpoint feasibility and obtaining advisor agreement; it is not an automatic plug-in replacement for D3.

## Common acquisition checklist for the next milestone

The following are full-dataset acceptance requirements. Partial MILE checks are recorded separately in its audit; no candidate has passed the whole checklist:

- [ ] Save the official accession, exact file URLs, retrieval date, byte sizes, and SHA-256 checksums.
- [ ] Record repository terms and original-study citation; do not assume that every GEO deposit has UCI's CC BY license.
- [ ] Count files, assay rows, feature columns, unique patients, repeated specimens, labels, missing values, and duplicates from the actual release.
- [ ] Preserve a feature dictionary: source ID, mapped gene ID, symbol, annotation source/version, ambiguity, and filtering reason.
- [ ] Audit all deposited preprocessing. Distinguish cohort-wide normalization from a strictly inductive deployment pipeline; declare unavoidable limitations.
- [ ] Freeze inclusion/exclusion criteria and patient/batch grouping before assessing model performance.
- [ ] Fit newly learned filtering, scaling, selection, and layouts within training partitions only; define how each step is applied to unseen samples.
- [ ] Define an eligible-gene background and versioned biological annotation for fair downstream comparisons.
- [ ] Estimate memory/time with a bounded feasibility check before committing to the full study.

**Week 2 outcome:** three evidence-backed candidates, one reserve, and explicit rejection/acceptance criteria. This is not a claim that three datasets have been collected, cleaned, approved, or shown to favor the proposed method.
