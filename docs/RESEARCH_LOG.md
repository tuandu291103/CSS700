# Research log

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
2. Start a bounded metadata and annotation audit of GSE13159 / MILE.
3. Resolve sample independence, label definitions, preprocessing, and gene mapping before accepting a dataset.
4. Agree on comparison methods and evaluation metrics before implementing models.
