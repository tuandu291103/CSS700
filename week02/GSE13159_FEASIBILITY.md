# GSE13159 / MILE: metadata and annotation feasibility audit

19 October 2026 | Week 2 | **Conditional candidate; not accepted for confirmatory experiments**

## Outcome

We verified the complete deposited sample inventory and a usable platform-annotation route. However, specimen type is strongly associated with diagnosis, patient/center linkage is not explicit in the downloaded metadata, and probe-to-gene mapping is not one-to-one. These are reasons to finish the data protocol before training—not evidence that an image representation solves the interpretation problem.

This is an original **data-quality audit**, not a novel biological finding or a model comparison. No expression matrix, raw CEL archive, model, split, or enrichment analysis was produced. The [machine-readable audit](GSE13159_AUDIT.json) contains computed counts, source URLs, retrieval date, byte sizes, and SHA-256 hashes.

## 1. What we checked

Downloaded from NCBI on 19 October 2026:

| Input | Bytes | Purpose |
|---|---:|---|
| `GSE13159_series.soft` | 66,951 | Official series metadata and expected sample accessions |
| `GSE13159_samples.soft` | 6,291,616 | Attributes of all samples, without expression tables |
| `GPL570.annot.gz` | 8,471,521 | Compressed platform annotation, internally dated **9 August 2016** |

Sources: [GSE13159](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE13159), [sample-attribute export](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE13159&targ=gsm&form=text&view=brief), [GPL570 annotation](https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz).

The 2,096 unique GSM accessions match the series inventory exactly. All 2,096 titles are unique and cover `N1_0001` through `N1_2096`. Every record specifies GPL570, Homo sapiens, a diagnostic label, a specimen type, and 54,675 data rows. **The row count is metadata-reported, not measured from an acquired expression matrix.** Unique assay identifiers do not prove unique patients.

The original study describes retrospective Stage I and an independent prospective Stage II. Its Stage II custom array has 1,480 probe sets, so it is not a drop-in same-feature test matrix. The parent accession GSE13204 must not be counted as a separate independent dataset from GSE13159. [Haferlach et al., 2010, Abstract, preprocessing and custom-chip sections](https://pmc.ncbi.nlm.nih.gov/articles/PMC5569671/).

## 2. Labels and specimen-type imbalance

The following counts are computed from all downloaded sample attributes, not copied from a secondary paper. The deposited historical label wording is preserved; it is not a claim about current clinical classification.

| Deposited diagnostic label | Bone marrow | Peripheral blood | Total |
|---|---:|---:|---:|
| ALL with hyperdiploid karyotype | 39 | 1 | 40 |
| ALL with t(12;21) | 58 | 0 | 58 |
| ALL with t(1;19) | 33 | 3 | 36 |
| AML complex aberrant karyotype | 46 | 2 | 48 |
| AML with inv(16)/t(16;16) | 27 | 1 | 28 |
| AML with normal karyotype + other abnormalities | 330 | 21 | 351 |
| AML with t(11q23)/MLL | 29 | 9 | 38 |
| AML with t(15;17) | 34 | 3 | 37 |
| AML with t(8;21) | 35 | 5 | 40 |
| CLL | 0 | 448 | 448 |
| CML | 66 | 10 | 76 |
| MDS | 206 | 0 | 206 |
| Non-leukemia and healthy bone marrow | 73 | 1 | 74 |
| Pro-B-ALL with t(11q23)/MLL | 55 | 15 | 70 |
| T-ALL | 170 | 4 | 174 |
| c-ALL/Pre-B-ALL with t(9;22) | 111 | 11 | 122 |
| c-ALL/Pre-B-ALL without t(9;22) | 232 | 5 | 237 |
| mature B-ALL with t(8;14) | 12 | 1 | 13 |
| **Total** | **1,556** | **540** | **2,096** |

Source and calculation: [sample export](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE13159&targ=gsm&form=text&view=brief), `samples.label_by_specimen_type` in the [audit JSON](GSE13159_AUDIT.json).

**Our analysis / protocol implications:**

- There are 18 complete labels, but class sizes range from 13 to 448 (34.46:1). Overall accuracy alone would be an inadequate account of performance. Plan per-class results and macro-averaged measures; quantify uncertainty for small classes.
- All CLL samples are blood; all MDS samples are marrow. This is an observed metadata association, **not proof that a model uses a shortcut**. Disease and specimen-source effects cannot be cleanly separated for these classes from this cohort alone.
- Propose a specimen-type-only baseline and specimen-stratified sensitivity analyses. These are diagnostic controls, not a complete cure for confounding. A marrow-only analysis would remove every CLL sample and change the scientific question; do not silently apply that filter.
- Keep the combined “Non-leukemia and healthy bone marrow” label intact. Do not describe all 74 samples as healthy controls; one even has a peripheral-blood specimen annotation requiring reconciliation.

## 3. Independence and processing: unresolved

**Observed:** the sample-characteristic fields contain only diagnostic class and specimen type. The downloaded records do not expose an explicit donor-linkage or collection-center/batch field. `N1_####` titles identify assays; no donor-linkage guarantee is supplied by this export. GEO contact details identify a submitter, not each sample's collection center. All records belong to both GSE13159 and GSE13204. [Complete sample export](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE13159&targ=gsm&form=text&view=brief).

**Needed:** inspect original-study supplements and linked metadata for patient/specimen relationships and center assignments. If unavailable, state the limitation explicitly and agree on an appropriate exploratory scope with the advisor. Do not label a random assay split “patient-independent” or claim external-center generalization. Numerical duplicate-profile checks also await an expression matrix and would not, by themselves, establish patient identity.

All sample records cite Liu et al.'s summarization/quantile-normalization method. A separately inspected [GSM330048 record, data-table descriptions](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM330048) distinguishes `VALUE_DS` from `VALUE`; the latter is DQN3 with theoretical beta quantiles (p=1.2, q=3). The [Liu et al., 2006 abstract](https://pubmed.ncbi.nlm.nih.gov/16889801/) distinguishes empirical and theoretical quantile targets. We therefore **cannot infer cohort-fitted normalization merely from the term “quantile normalization.”** Conversely, this does not certify the entire deposited pipeline as leakage-free. Full processing specification, value scale, selected matrix column, and sample-independent applicability remain to be verified. No log transform or new normalization has been applied.

## 4. Probe-to-gene tracing is feasible but ambiguous

The downloaded GPL570 annotation contains 54,675 unique probe IDs. Splitting the `Gene ID` field on `///` and counting distinct Entrez identifiers per probe gives:

| Mapping in this annotation snapshot | Probe rows |
|---|---:|
| Exactly one gene identifier | 42,904 |
| Multiple gene identifiers | 2,214 |
| No gene identifier | 9,557 |
| **Total** | **54,675** |

The single-gene rows cover **20,848 distinct identifiers**, with 10,924 identifiers represented by more than one such probe. These totals include 62 `AFFX-` control rows (10 single-gene, 52 unmapped). They are **not** a final retained-gene count or proof of up-to-date identifier validity. Source: [GPL570 annotation snapshot](https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz); calculations in the [audit JSON](GSE13159_AUDIT.json).

For example, `1007_s_at` maps to both `100616237` / MIR4640 and `780` / DDR1 in this snapshot. Selecting the first name would conceal ambiguity. Also, matching the platform's row count to the sample-declared row count does not verify an exact expression-ID join.

**Proposed policy, not yet implemented:** preserve probe-level attribution and provenance; flag controls, unmapped probes, and one-to-many mappings; never silently assign an ambiguous probe to one gene. Before gene-set analysis, choose and freeze a versioned annotation, reconcile retired identifiers, and document how multiple probes per gene are combined. Use the same eligible-gene background and mapping rules for image and tabular explanations. Report a sensitivity analysis if mapping choices materially affect findings. The 2016 snapshot is suitable for this feasibility check, not automatically the final biological annotation.

## 5. Decision and next bounded milestone

| Acceptance check | Status |
|---|---|
| Complete deposited sample inventory and label counts | Verified |
| Platform annotation available and ambiguity quantified | Verified for the downloaded 2016 snapshot |
| Independent-patient and center grouping | Unresolved |
| Specimen-source confounding addressed by a declared protocol | Unresolved; strong association identified |
| Processed-value provenance, scale, and inductive applicability | Partially documented; not approved |
| Expression IDs, missing values, and duplicate profiles | Not audited |
| Final gene mapping and enrichment background | Not frozen |
| Reuse terms and original-study requirements | Public download verified; no blanket CC BY assumption or complete terms review |

**Recommendation:** retain MILE as a conditional candidate, not a confirmed thesis benchmark. Next, resolve metadata/provenance gaps and draft a short dataset-acceptance protocol covering specimen-type controls, patient grouping, and mapping policy. If those constraints cannot support the intended claims, narrow the claims or replace the candidate before model development. The other two candidate datasets remain unaudited; three independent datasets have not yet been accepted.

## Reproduce this audit

Python 3.9+ with the standard library is sufficient. From the repository root, the following commands retrieve only metadata/annotation (about 14.8 MB total), not the large raw CEL archive. Existing raw copies are kept locally under ignored `data/raw/`; they are not pushed to GitHub. GEO exports can change: compare byte hashes in the JSON before treating a later run as an exact reproduction. A new retrieval date should be supplied for a new download.

```sh
mkdir -p data/raw/gse13159
curl --fail --location --max-time 240 --max-filesize 1000000 --output data/raw/gse13159/GSE13159_series.soft 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE13159&targ=self&form=text&view=brief'
curl --fail --location --max-time 240 --max-filesize 25000000 --output data/raw/gse13159/GSE13159_samples.soft 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE13159&targ=gsm&form=text&view=brief'
curl --fail --location --max-time 240 --max-filesize 35000000 --output data/raw/gse13159/GPL570.annot.gz 'https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz'
```

For the locally preserved 19 October snapshot:

```sh
python3 scripts/audit_mile_metadata.py --series data/raw/gse13159/GSE13159_series.soft --samples data/raw/gse13159/GSE13159_samples.soft --annotation data/raw/gse13159/GPL570.annot.gz --retrieved-on 2026-10-19 --output /tmp/GSE13159_AUDIT_reproduced.json
diff -u week02/GSE13159_AUDIT.json /tmp/GSE13159_AUDIT_reproduced.json
python3 -m unittest discover -s tests -v
```

The script rejects missing/duplicate sample accessions, inventory mismatches, missing/repeated required fields, malformed or incomplete annotation tables, wrong row counts, invalid gene identifiers, and corrupt gzip input. Ten synthetic tests cover key parser/counting and rejection cases. Successful checks verify the audit software and metadata consistency—not biological validity or leakage-free modeling.

**Access note:** direct article pages intermittently returned browser checks; indexed primary-paper excerpts were available. The Europe PMC full-text endpoint returned HTTP 500. The first 60-second bulk metadata transfer timed out; its partial file was not used. A complete retry passed the exact accession-inventory check. The full original-study supplements and full PQN/DQN paper have not been reviewed.
