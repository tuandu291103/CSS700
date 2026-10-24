# Research log

## 23 October 2026 — learning experiment 02: tabular baselines and small CNN

**Completed**

- Recorded a fixed local protocol before training: unchanged Wisconsin split/map, training-only scaling, identical float32 values, no tuning search, three stochastic-model seeds, and validation-only CNN checkpoint selection.
- Trained logistic regression, three random forests, and three 1,282-parameter CNNs, plus a majority/prior control. Saved every run rather than choosing the best seed. Local checkpoints remain ignored; predictions, histories, metrics, and settings are tracked.
- Added a [brief report and CNN layer walkthrough](../experiments/02_wisconsin_models/README.md), learning curves, and all eight confusion matrices. No SOM2IMG implementation or attribution method was added.
- Installed PyTorch in the isolated environment, updated exact dependency versions, and removed an inadvertently captured pip warning from the prior requirements file. Added a requirements-parser regression test and verified an offline install dry-run.

**Results and interpretation**

On the same 86 learning-test records, logistic regression achieved 83 correct; CNN runs achieved 83/83/82; forest runs achieved 80/79/79. CNN mean accuracy was 96.12%, macro-F1 0.9578, and log loss 0.2480; logistic regression was 96.51%, 0.9619, and 0.1561. Thus similar accuracy did not imply equal probability quality. Forest had the highest mean ROC-AUC (0.9826) despite lower thresholded accuracy. These are measured learning results from [the saved outputs](../experiments/02_wisconsin_models/results.json), not general superiority claims or interpretation findings.

CNN checkpoints were selected at epochs 199, 197, and 142. Two runs reached the 200-epoch cap; the third stopped at 172 and restored its best validation checkpoint. Seed variation is not sampling uncertainty: all runs share the same small, reused split. No post-test retuning was performed. A second full run was only a reproducibility check.

**Verification**

- Full training rerun in a separate temporary directory produced byte-identical `results.json`.
- All 16 saved validation/test metric sets recomputed from stored probabilities; all three CNN checkpoints reloaded to exactly matching saved test predictions.
- Visually inspected both figures. Existing Experiment 01 results and all earlier genomic files remain unchanged.
- Added ten tests for model shapes/gradients, label/metric conventions, deterministic training, checkpoint restoration, dependency syntax, saved-score integrity, and seed coverage, bringing the total to 33.
- All 33 tests and new-code Ruff checks passed. Verified protocol/code/input-audit hashes, installed dependency pins, local links, bibliography keys, and Git whitespace before publication.

**Next**

Freeze the fitted models. Walk through CNN prediction mechanics, then explain a small deterministic validation set, beginning with the first listed CNN seed rather than selecting by test performance. Compare explanation behavior and run perturbation/randomization controls before claiming faithful interpretation. Keep the second dataset, learned layout, and genomic extension as separate later steps.

## 23 October 2026 — learning experiment 01: Wisconsin and fixed-grid encoding

**Direction**

The user relayed the professor's request to understand each stage separately, using one or two SOM2IMG benchmarks before thesis experiments. Started with Wisconsin Diagnostic; Spambase remains a proposed follow-up. Preserved all previous genomic research and archives. This is a learning milestone, not a scope claim that the final thesis is no longer genomic.

**Completed**

- Retrieved official UCI `wdbc.data` and `wdbc.names`, recorded SHA-256 hashes, and kept raw downloads ignored by Git.
- Added a [dataset card and brief report](../experiments/01_wisconsin/README.md), an executed walkthrough notebook, training-correlation and encoding figures, and machine-readable results.
- Verified 569 records, 30 predictors, 357 benign / 212 malignant labels, no missing/nonfinite measurements, and no duplicate IDs or exact predictor rows.
- Stored a stratified 397/86/86 learning split, seed 20261023. IDs and labels are excluded from the encoded predictors. Descriptive statistics/correlations and scaler fitting use training rows only.
- Implemented one-to-one row-major placement into a 5×6 float64 grid. All 30 scaled values recover exactly; inverse scaling has maximum absolute error 4.55e-13. No clipping, 8-bit conversion, or image resizing is used.
- Added an isolated Python 3.12.14 learning environment snapshot and ten synthetic mapping/parser/split/scaling tests alongside the 13 existing MILE audit tests.

**What we learned**

Mean radius versus mean area has training Spearman rho 0.9996; radius versus perimeter is 0.9977. Relationships can therefore be visible in the table before conversion. Later spatial explanations must demonstrate an additional benefit under defined checks. Brightness here is a feature value, not model importance. Exact round-trip recovery validates this encoder, not interpretability. Sources and calculated evidence are linked in the experiment report.

**Limits and next step**

No classifier, SOM layout, attribution method, second dataset, or biological discovery was produced. Unique IDs do not establish patient independence. The learning split is not a new sealed thesis holdout, especially given prior archived Wisconsin work. Obtain author implementation/version/splits/collision details before a SOM2IMG reproduction. Next: walk through the notebook, then train simple tabular baselines and a small fixed-grid CNN as a separate experiment.

**Verification**

- All 23 tests passed, including the unchanged MILE audit suite; new-code Ruff checks passed.
- Executed all five code cells in the notebook using the isolated environment and validated notebook structure and outputs; no error outputs remained.
- Recomputed the audit in a separate temporary directory and confirmed byte-identical `results.json` output.
- Visually inspected both generated figures and checked source links, bibliography-key uniqueness, and Git whitespace.
- Raw data, the environment, and archives stay outside Git tracking. No older research file was deleted or archived result presented as newly computed.

## 19 October 2026 — focused SOM2IMG initial study

**Completed**

- Read and visually checked all six pages of the supplied SOM2IMG manuscript; recorded its checksum and distinguished it from the final IEEE bibliographic record and an unread OpenReview version.
- Added a [focused review and Wednesday discussion notes](../week02/SOM2IMG_REVIEW.md), covering the mapping mechanism, all nine benchmarks, reported evaluation, interpretation limits, and candidate comparison roles.
- Cross-checked seven UCI records and the Mendeley record, without acquiring benchmark data; added source links and separate bibliography keys.
- Updated the README, dataset notes, reading log, and the evidence review's stale audit-status statement. No archived material or research code was changed.

**Findings and limits**

The supplied manuscript does not report a gene-expression benchmark; its feature-neighborhood interpretation is qualitative. A biological interpretation benefit remains a hypothesis. BMU assignments alone do not enforce one-to-one feature placement, and the manuscript does not establish fold-local preprocessing/layout fitting. These are reproduction questions, not observed failures in author code. Dataset dimension discrepancies also require exact release/encoding verification. All locators and primary-record links are in the focused review.

**Next**

Clarify which benchmark and paper version the advisor intends; request the author implementation or missing specification, particularly collision handling and fold boundaries. Consider Wisconsin Diagnostic only as a separate technical check, not a gene-expression cohort. Continue the existing dataset acceptance work; no acceptance, advisor approval, new model training, or biological discovery is claimed.

**Verification**

- Source manuscript left unchanged; relevant tables and equations checked visually rather than inferred solely from extracted text.
- Document links, bibliography keys, Git whitespace, and the existing audit tests checked before publication. This is a documentation-only change; test success does not reproduce SOM2IMG.

## 19 October 2026 — follow-up: acceptance protocol and sample-table checks

**Completed**

- Reviewed the original MILE article's methods, center-count table, and embedded appendix; documented what they do and do not resolve about patient/center linkage.
- Added a clearly labeled primary technical disclosure for fixed-target normalization; did not substitute it for verification of the exact deposited pipeline or the unread full Liu 2006 paper.
- Inspected two preselected profiles (first CLL and first MDS accessions), verified their exact platform probe-ID sets, and recorded numeric/missing-value statistics and input hashes.
- Drafted the [acceptance protocol](../week02/GSE13159_ACCEPTANCE_PROTOCOL.md): separate exploratory/confirmatory claims, patient grouping, specimen-only controls, stratified reporting, and mapping/preprocessing rules.

**Findings and limits**

Each checked profile has 54,675 rows, 54,630 finite DQN3 values in [0, 1], and the same 45 missing `AFFX-` values. This guards against confusing DS with DQN3, zero-filling missing controls, or automatically applying log2. Two purpose-selected profiles do not certify the full cohort. No full-cohort matrix, model training, new predictive score, or biological discovery was produced.

Patient-origin information is better documented, but no complete patient/center linkage map has been recovered. MILE remains conditional; no advisor acceptance is assumed. Source details and reproducible calculations are in the updated [feasibility report](../week02/GSE13159_FEASIBILITY.md) and [audit JSON](../week02/GSE13159_AUDIT.json).

**Verification**

- Thirteen synthetic tests passed, including new profile identity, missingness, truncation, nonfinite-value, and probe-join checks.
- Checked exact rerun reproducibility, unchanged metadata-only output, bibliography/local links, and Git whitespace before publication.
- Raw inputs and the article reading copy remain ignored; archived work was not changed.

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

1. Review the completed model comparison and CNN layer walkthrough; separate predictive metrics from explanation quality.
2. Explain frozen-model predictions on deterministic validation examples, then test explanation faithfulness and shortcut behavior. Avoid retuning against the observed learning-test scores.
3. Obtain the intended SOM2IMG implementation/version and exact benchmark/preprocessing details before reproduction. Consider Spambase after the first pipeline is understood.
4. Return to the preserved genomic acceptance work and agree on the final thesis protocol before confirmatory experiments.
