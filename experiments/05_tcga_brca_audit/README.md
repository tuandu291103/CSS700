# Experiment 05 — TCGA-BRCA acquisition and data audit

26 October 2026. Acquisition and data-quality review preceding the TCGA-BRCA prediction pilot.

The conservative cohort described below was subsequently used in [Experiment 06](../06_tcga_brca_pilot/README.md). This report records the findings and decisions at the audit stage, before model training.

## Purpose and scope

The audit covers the complete GDC TCGA-BRCA STAR–TPM expression matrix, companion clinical table and GENCODE v36 gene map obtained through UCSC Xena. My advisor suggested TCGA and Xena as resources, leading to the [initial source review](../../docs/TCGA_XENA_FEASIBILITY.md) and this acquisition. Original downloads and sample-level working manifests remain outside version control.

The files support a preliminary genomic classification study of primary tumor versus solid-tissue normal samples. This sample-type task does not represent healthy-person screening, survival prediction or molecular-subtype discovery. Suitability for the final thesis cohort depends on the interpretation question and the selection limitations identified below.

## Measured results

The counts below were calculated from the full downloaded files and recorded in [results.json](results.json).

| Audit item | Result |
|---|---|
| Expression matrix | 60,660 versioned Ensembl identifiers × 1,226 samples |
| Complete numeric scan | 74,369,160 values; no missing, infinite, or negative values |
| Clinical source | 1,255 records; all 1,226 expression samples matched by exact ID; 29 clinical-only records |
| Identity checks | 1,095 participants; barcode/clinical IDs and participant–case UUID linkage agree |
| Expression-matched labels | 1,106 primary tumors, 113 solid-tissue normals, 7 metastatic samples |
| Gene mapping | 60,660/60,660 exact versioned-ID matches; no unmapped expression IDs |
| Duplicates | No identical complete sample-expression profiles; 11 repeated participant/type vials in the binary candidate |
| Unfiltered binary candidate after one-vial rule | 1,208 samples: 1,095 tumors + 113 normals; 1,095 participants |
| Paired participants | All 113 normals have a tumor sample from the same participant |

The downloaded matrix is already `log2(TPM+1)` according to its [Xena metadata](https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.star_tpm.tsv.json). No second log, imputation, scaling, feature filtering, or train/test split has been applied. Sample-type meanings come from [GDC's code table](https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/sample-type-codes); participant/sample/vial distinctions follow [GDC barcodes](https://docs.gdc.cancer.gov/Encyclopedia/pages/TCGA_Barcode/).

### Resolved: why Xena says 60,661

The initial protocol retained the page's 60,661-identifier listing. The full file has 60,660 expression rows, so the first scan correctly raised a discrepancy. We then inspected Xena's **complete** [identifier inventory](https://xenabrowser.net/datapages/?host=https%3A%2F%2Fgdc.xenahubs.net&dataset=TCGA-BRCA.star_tpm.tsv&allIdentifiers=true): 60,660 unique Ensembl IDs plus `sampleID`.

The SHA-256 digest of the sorted browser gene-ID list matches the downloaded expression-ID list exactly, resolving the discrepancy as the additional `sampleID` entry. The [follow-up record](source_count_review.json) describes the comparison. The original acquisition protocol remains unchanged, and the results retain both the initial mismatch and its resolution.

### Gene identity is usable, but names are not unique

There are 59,427 distinct source symbols for 60,660 identifiers. In particular, 110 symbols represent multiple identifiers; examples include `Y_RNA` (756 IDs) and `Metazoa_SRP` (170). Retain the **versioned Ensembl ID as the key** and the source symbol as an annotation. We did not merge symbols or assume every row is a protein-coding gene. A future pathway analysis needs its own identifier-conversion and gene-universe policy.

The scan found 2,612 all-zero rows. They are a data-quality description, **not a selected feature set**: any filtering for modeling must be learned from the training partition only. Identical all-zero feature rows are not duplicate patient records.

## Problems and analysis

1. **Patient dependence.** Splitting the 1,208 candidate samples independently could put one patient's normal tissue in training and tumor in testing. Keep every participant's samples in the same partition; verify disjoint case UUIDs too. The 113 normals are not 113 independent healthy donors.
2. **Clinical annotations.** The candidate contains 31 annotated samples from 29 participants, covering prior malignancy, treatment and study-protocol issues. We reviewed the supplied category/note fields. One sample-specific warning exists in the full clinical table, but its affected vial is absent from the expression matrix; the replacement vial named by the source is present. This does not certify all other samples. [GDC annotation guidance](https://docs.gdc.cancer.gov/Encyclopedia/pages/Annotations/) explains why such notes matter before analysis.
3. **Preservation disagreement.** Five candidate tumors have `preservation_method.samples=FFPE`; only one has `is_ffpe.samples=True`. Four therefore disagree across these fields. Preserve both fields; do not silently choose one as truth. Unknown preservation is also common.
4. **Collection-site imbalance.** The candidate has 40 tissue-source-site codes, but only six supply both classes. There are 701 tumors at sites with no normal samples. Site BH supplies 73/113 normals but only 151/1,095 tumors. This is a measured confounding **risk**, not proof that a model uses site effects. Participant grouping alone does not remove it. Xena also cautions that GDC STAR data may retain more batch effects than its Pan-Cancer Atlas release. [Xena dataset guidance](https://ucsc-xena.gitbook.io/project/faq/advanced-data-and-datasets).
5. **Different releases and limited labels.** Expression metadata names GDC release 40.0; clinical metadata names 41.0. Exact ID/label checks passed for their intersection, but that is not a complete historical provenance reconciliation. No PAM50-named field exists in this clinical file. The exact source aggregation across measurements/aliquots is still not independently reproduced.

## Conservative cohort option — calculated, not yet adopted

Before looking at any prediction results, we evaluated this **metadata-only option**:

- Exclude a participant if any of their clinical records contains a case annotation/category or note, or does not explicitly document `prior_treatment.diagnoses=No`.
- Exclude an affected sample if it has a sample annotation or either preservation field flags FFPE.
- Then select the lexicographically first remaining vial per participant and sample type; keep only codes 01 and 11. Preserve tumor–normal pairing for later grouped splits.
- Retain unknown preservation as **unknown**, not as proven fresh tissue. Do not remove sites based on future scores.

This yields **1,170 samples from 1,059 participants: 1,059 tumors and 111 normals**, including 111 paired participants. No raw records were deleted. Exclusion reasons and candidate sample IDs are saved in the ignored local manifest; aggregate counts are in `conservative_cohort_option` in [results.json](results.json).

This broad exclusion policy is our cautious engineering proposal, **not a GDC requirement or a clinical judgment that every flagged record is invalid**. It changes the population being studied and does not eliminate site confounding. The unfiltered candidate remains documented for a future, prospectively specified sensitivity analysis. No cohort option has been selected by model performance.

## Planned follow-up

1. Adopt and freeze a cohort policy for the learning task, separately from final thesis acceptance. Ask the professor whether this task is a useful bridge to the intended biological question.
2. Specify participant-grouped train/validation/test partitions and a collection-site sensitivity check before fitting anything. Keep all filtering, scaling and learned layouts training-only.
3. Declare a modest gene budget, a majority-class control and a tabular baseline; then compare a fixed-grid CNN on identical features/splits. Include class-aware prediction measures, not accuracy alone. Neither model nor parameters are fixed by this audit.
4. Only after the predictive pipeline works, trace image attributions to stable gene IDs and compare explanation faithfulness, stability and gene-group meaning. A heatmap or high tumor–normal accuracy does not establish a thesis contribution.

**Professor discussion:** Should the primary learning cohort use the conservative exclusions? Should collection-site generalization be a required control? What biological relationship, beyond distinguishing these tissue sample types, should a successful explanation demonstrate?

## Evidence and reproduction

- [Pre-scan acquisition protocol](protocol.json), [source-count follow-up](source_count_review.json), [aggregate results and full-file SHA-256 hashes](results.json).
- [Streaming audit and cohort-option code](../../scripts/audit_tcga_brca.py), [resumable expression downloader](../../scripts/download_tcga_expression.py), [integrity tests](../../tests/test_audit_tcga_brca.py).
- [Source metadata manifest](../../docs/TCGA_XENA_SOURCE_MANIFEST.json), [research log](../../docs/RESEARCH_LOG.md).

The full expression gzip is 342,300,776 bytes. Its complete gzip CRC and every numeric cell were checked; SHA-256 identifies the exact local snapshot, not an independent publisher-signed checksum. Clinical and annotation downloads have their own sizes/hashes. The three original metadata sidecars remain unchanged, including the clinical sidecar's non-standard trailing comma.

With the three full data files at the `raw_directory` in the protocol and Python 3.12.14 / NumPy 2.3.5:

```bash
python -m scripts.audit_tcga_brca
python -m unittest discover -s tests -q
```

This run used the bundled scientific Python for the audit and the repository `.venv` for tests. The offline audit does not contact Xena; it checks against the saved browser-inventory digest. The downloader is pinned to the inspected object's size and ETag and refuses to replace a final file. For a fresh acquisition, retrieve the clinical/map URLs in `results.json` into a new raw directory, and run the downloader with `--raw-dir` pointing there. If the source object has changed, investigate the new release rather than bypassing the pin.

An independent audit run reproduced the aggregate results and local sample-manifest JSON byte-for-byte. All 68 tests available at this stage and Ruff checks for the new code passed. Older MILE files retained their existing style warnings.

The audit established complete sample joins and gene mapping, resolved the source-count discrepancy, and identified participant dependence and collection-site imbalance. It also quantified a conservative cohort for the prediction pilot. These checks provide the data foundation for later modeling without resolving biological interpretation.
