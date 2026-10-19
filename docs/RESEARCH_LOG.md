# Research log

## 19 October 2026 — MILE metadata and annotation feasibility

**Completed**

- Retrieved complete GSE13159 series/sample attributes and GPL570 annotation; stored local inputs outside Git tracking, with URLs, date, sizes, and hashes in a compact audit summary.
- Verified the exact 2,096-accession inventory, 18 complete diagnostic labels, and specimen-type counts.
- Quantified missing and ambiguous probe-to-gene mappings in the annotation dated August 2016.
- Added one offline audit script, 10 synthetic tests, a [feasibility report](../week02/GSE13159_FEASIBILITY.md), and the original MILE/PQN-DQN citations with reading-depth limits.

**Findings and decision**

All 448 CLL samples are peripheral blood; all 206 MDS samples are marrow. This creates a potential specimen-source confound that requires explicit controls. Unique sample titles do not establish patient independence, and the exported metadata lacks explicit patient/collection-center linkage. Of 54,675 annotation rows, 2,214 map to multiple Entrez identifiers and 9,557 to none. Counts and primary-source provenance are in the [audit JSON](../week02/GSE13159_AUDIT.json).

MILE remains conditional, not accepted. No expression matrix, new model, prediction result, or biological discovery was produced. Full preprocessing provenance, current identifier reconciliation, expression-level checks, and an appropriate interpretation of confounding remain open. This work stays within Week 2 dataset selection; the other candidates still need their own audits.

**Verification**

- Successful bulk retry matched all accessions in the independent series inventory; the initial timed-out partial transfer was not used.
- Annotation gzip integrity, table boundaries, unique probe IDs, and row counts checked.
- All 10 synthetic tests passed; exact-input rerun and document consistency checked before publication.
- New metadata/annotation inputs are ignored by Git. No archived research was changed or restored.

## 17 October 2026 — repository restart

**Completed**

- Made this repository the active home for the revised thesis study.
- Imported the Week 2 evidence review, proposal wording, dataset notes, reading log, and reference library.
- Replaced the old README with a concise introduction, document index, and current next steps.
- Removed the earlier implementation, experiment configuration, test suite, old plans and logs, and Python CI workflow from the active tree.
- Moved the old local checkout contents into a recovery folder and rechecked the existing ZIP checksum. Git history remains intact.
- Kept the meeting presentation separate, as requested. This entry records only the repository cleanup.

**Verification**

- Confirmed the initial checkout was clean and matched `origin/main` before the cleanup.
- Checked the imported documents and open-access paper against their source files.
- Checked the final document links, bibliography keys, and absence of obsolete installation/test instructions before publication.

**Scope**

This was an organization and publication step. It adds no new literature findings, dataset audits, or experimental results. [Recovery details](ARCHIVE.md) explain what was removed and how to retrieve it.

## 15 October 2026 — Week 2 evidence and dataset candidates

**Completed**

- Archived both earlier workspaces and the supplied proposal/reference PDFs.
- Prepared a claim-by-claim review using nine primary research references, with reading depth and limitations documented.
- Drafted a citation-backed motivation and a reusable BibTeX library.
- Documented three transcriptomic dataset candidates and one reserve, including preprocessing and gene-identifier acceptance checks.

**Findings**

The review does not justify universal claims that CNNs cannot outperform tabular ML or that traditional rankings are useless. Existing work already connects image explanations to genes and biological groups. The revised study asks when spatial explanations add reliable information under a fair comparison. See the [evidence review](../week02/CLAIM_EVIDENCE.md) for the supporting references.

The [dataset notes](../week02/DATASET_CANDIDATES.md) identify unresolved preprocessing and annotation issues. No candidate has yet passed a complete data audit, and no new training has begun.

## Next milestone

1. Review the revised question and evidence with the advisor.
2. Follow up MILE patient/center linkage and preprocessing provenance; its bounded metadata/annotation audit is complete.
3. Draft the acceptance protocol with specimen-type controls, mapping policy, and unresolved limitations; audit the other candidates before committing to three datasets.
4. Agree on comparison methods and evaluation metrics before implementing models.
