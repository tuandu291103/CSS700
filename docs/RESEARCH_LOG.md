# Research log

## 31 October 2026 — documentation revision and publication

I published the completed TCGA audit, genomic prediction pilot and controlled SOM2IMG layout experiment in commit [`df812d9`](https://github.com/tuandu291103/CSS700/commit/df812d9). I then revised the repository documentation for clearer academic writing, keeping the measured results, sources, dated decisions and limitations intact. The presentation was outside this editorial pass. No experiment was rerun or research claim added as part of the wording changes.

The review covered 21 Markdown documents and the walkthrough notebook's explanatory text. All 94 tests passed. Checks confirmed valid local document links, 49 unique and unchanged bibliography entries, unchanged numerical result-table cells and reproduction commands in the experiment reports, and unchanged notebook code, outputs and metadata.

## 31 October 2026 — author-code SOM2IMG experiment

**Completed:** inspected the [author repository](https://github.com/Kerbash/tab2img-bench) and pinned commit `dc0ef691db733ebcba4d13b53ca58e258f76ef0a`. Confirmed its fit/transform interface, saved feature map, default spillover collision rule, training-fold-local fitting and documented uint8 wraparound behavior. The current README describes a newer 51-dataset study, so it is recorded separately from the prior anonymous manuscript and IEEE citation. Preserved upstream unmodified under an ignored cache; added only pinned missing dependencies, without upgrading the earlier modeling libraries.

Recorded [Experiment 07's protocol](../experiments/07_som2img_layout/protocol.json) before execution. Split the saved training partition into 318 fit / 79 inner-stop records and used the old 86 validation records only for exploratory scoring after model freezing. Compared one fixed, three lossless random and three SOM layouts, crossed with three identical CNN seeds (21 models). All use the same float inputs, features, CNN and optimizer settings. The old learning-test partition and genomic models were left untouched. This experiment evaluates layouts and predictions; attribution comparison is the next stage.

**Findings:** all layouts retained 30/30 features with exact float64 scaled-value recovery. SOM neighbor-profile absolute Pearson correlation averaged 0.580 versus 0.416 for random; neighbor-set Jaccard across SOM seeds averaged 0.211. Same-measurement distance remained within the descriptive random reference interval, providing no clear evidence of semantic enrichment under this measure. Mean evaluation accuracy: 98.97% SOM, 98.32% random, 97.67% fixed. The 0.65-point SOM/random difference is small on 86 previously inspected records; all 21 runs reached the fixed epoch cap. A separate numeric audit confirmed 28 values in nine evaluation samples differ by over 0.5 between default and clipped uint8 export. The authors already document this issue, and our matched comparison avoids it with float32. The audit did not test quantization's effect on prediction. The small reused evaluation set also limits conclusions about model ranking, and the experiment does not evaluate biological interpretation.

**Deliverables:** [full experiment report with four figures](../experiments/07_som2img_layout/README.md), protocol, computed evidence, mappings, runner, 12 tests, version-specific references, and [weekly advisor discussion notes](WEEKLY_FINDINGS_2026-10-31.md). The notes connect the layout findings to the previous explanation and shortcut controls. No faculty meeting took place this week.

**Verification:** 94 project tests and 18 upstream tests pass; new-code Ruff and dependency checks pass. Independent execution produced byte-identical JSON and four figure files. All 21 local checkpoints reloaded to identical saved evaluation probabilities, and source hashes matched. Figures visually inspected. The upstream test using bundled results is not a retraining of 51 datasets. Earlier experiments and archives were preserved. Publication is recorded separately in the entry above.

**Next:** keep models frozen, agree on the intended code/paper version and define a matched input-attribution comparison across layouts, including reference sensitivity and feature replacement. Evaluate attributions by feature names rather than image coordinates. Biological usefulness and confirmatory genomic evaluation remain later work.

## 26 October 2026 — first TCGA-BRCA prediction pilot and guidance check

**Completed:** adopted the conservative learning cohort, recorded [Experiment 06's protocol](../experiments/06_tcga_brca_pilot/protocol.json) before training, and froze two metadata-only participant/case-disjoint splits. The primary split has 816/177/177 samples; the BH-site sensitivity scenario has 806/143/221. Selected 256 genes independently using each training partition's variance, standardized with training statistics, and verified identical float32 table/grid values in a 16×16 fixed layout. Trained logistic regression, three forests, three small CNNs and a majority/prior control per scenario. All fitting/checkpoint selection in both scenarios preceded test scoring.

**Results:** grouped-test logistic regression and forests classified 177/177 correctly; CNNs scored 171/173/172. On BH, logistic regression scored 215/221, forests 210/209/209, and CNNs 209/213/214. BH balanced accuracy: 96.24% logistic, 92.01% forest mean, 96.38% CNN mean. These show different class-error tradeoffs, not a general winner or biological interpretation quality. [Full report and sources](../experiments/06_tcga_brca_pilot/README.md).

**Limits:** one selected cohort and two dependent splits; only 17 normal samples in the grouped test and six in BH validation; five CNN runs reached the fixed epoch cap. The layout is arbitrary, not similarity-preserving or SOM-based. The observed test results were not used for retuning. Genomic attribution, pathway validation and independent external evaluation remain future work. Class-weighted scores are not calibrated clinical probabilities.

**Problems/verification:** a float32 reduction assertion was corrected to accumulate diagnostics in float64 before training; partitions/protocol remained unchanged and the preflight file was preserved. All 82 tests and new-code Ruff checks pass. Full independent preparation and model reruns match byte-for-byte; all 48 metric sets were recomputed from saved local predictions, all classifiers reloaded to matching validation/test predictions, and the learning-curves figure visually inspected. Bundled NumPy handled preparation; the existing pinned modeling environment supplied the unavailable PyTorch/scikit-learn packages.

**Relation to the research plan:** this remains part of the separate-learning phase: Wisconsin first, conversion/CNN/explanation controls, then the professor-suggested TCGA/Xena extension. Three classifier families do not complete the broader three-dataset/T2I-comparison/interpretation-metrics plan. The report includes questions on the final biological target, eligibility/site controls, useful gene-group interpretation and SOM2IMG author materials. These questions remain open for advisor discussion.

**Next:** freeze the fitted models and specify a small deterministic validation-only genomic explanation exercise. Compare input tracing and reference/perturbation behavior before testing biological meaning or learned layouts. Preserve the now-observed test sets as learning results rather than repeatedly tuning against them.

## 26 October 2026 — complete TCGA-BRCA acquisition audit

**Completed:** acquired the full expression, clinical and gene-map files. Added a streaming full-value/identity/mapping audit, a pinned resumable downloader, 18 tests, [Experiment 05 report](../experiments/05_tcga_brca_audit/README.md), pre-scan protocol, source-count reconciliation and aggregate results. Raw data and sample-level manifests remain ignored; no previous experiment or archive was removed.

**Measured findings:** 60,660 versioned expression IDs × 1,226 samples, all 74,369,160 values finite/nonnegative, complete clinical-ID matching and exact gene-map coverage. There are 1,095 participants. Excluding seven metastatic samples and selecting one vial per participant/type gives 1,095 tumors and 113 normals; all 113 normals are paired with tumor participants. The initial 60,661 source-page listing was reconciled by inspecting the complete browser inventory: its extra entry is `sampleID`, and the sorted gene-ID digest matches the downloaded file.

**Issues and options:** 31 candidate samples/29 participants carry case annotations. Preservation fields disagree on four candidate samples. Only six of 40 collection-site codes supply both classes, so a participant-only split cannot rule out site effects. A documented metadata-only conservative option yields 1,059 tumors and 111 normals from 1,059 participants. It excludes annotated/prior-treatment-not-explicitly-No cases and sample-level annotation/FFPE flags before choosing vials. At this audit stage, the conservative option was a proposed cohort policy. It did not establish that flagged records were clinically invalid or settle the final thesis cohort. GDC annotation guidance and exact data sources are linked in the report.

**Verification:** gzip integrity, full-file sizes/SHA-256, complete numeric scan, independent row-count/ID-set checks, full browser-ID digest comparison, 68 passing tests and clean new-code Ruff checks. A second full audit produced byte-identical aggregate and private-manifest JSON. Whole-cohort zero-row counts remain diagnostic only; no genes were selected, no split was made, and no genomic model was trained.

**Next:** freeze a learning-cohort policy and participant-grouped partitions, specify a collection-site sensitivity check, then compare a modest tabular baseline and fixed-grid CNN on the same training-selected genes. Ask the professor how this tumor–normal exercise should connect to the final biological explanation target. The immediate objective is pipeline feasibility; biological usefulness requires a separate evaluation.

## 26 October 2026 — professor-suggested TCGA/Xena source inspection

**Completed:** followed my advisor's 21 October recommendation to investigate TCGA and Xena. Distinguished the TCGA research program from the Xena access platform; inspected the live GDC BRCA RNA-seq details, clinical preview and complete 85-field inventory. Saved and checksummed three small description files (1,560 bytes) outside Git tracking. Added a [feasibility report](TCGA_XENA_FEASIBILITY.md), [source manifest](TCGA_XENA_SOURCE_MANIFEST.json), references and an acquisition plan.

**Findings:** the expression page lists 60,661 identifiers × 1,226 samples, already in `log2(TPM+1)`, with a GENCODE v36 mapping route. A header-only request reports a 342,300,776-byte compressed matrix. Expression and clinical metadata point to GDC releases 40.0 and 41.0 respectively. The clinical file lists sample-type/case fields but no PAM50; the separate cohort PAM50 listing covers only 74 ATAC-associated samples. These are source observations, not computed full-cohort counts or proof of unusable data.

**Decision/limits:** propose BRCA as the next acquisition audit, with primary tumor versus solid-tissue normal as a possible learning target. This source inspection preceded full data acquisition. Patient grouping, release overlap, gene identities and class balance still required audit before choosing a modeling target. The Wisconsin and MILE studies remained available; Xena was treated as an access platform for TCGA, not a second independent dataset.

**Verification:** checked metadata sizes/hashes, source units/releases and Git-ignore status; recorded the source clinical sidecar's trailing comma without altering its bytes. The provenance manifest is strict JSON. Documentation-only change; previous model scores and the recorded 50-test milestone are not newly computed findings.

**Next:** acquire and audit the three data tables under the documented protocol, then agree on the biological prediction/evaluation target before training. Actual SOM2IMG reproduction still requires the intended author implementation details.

## 23 October 2026 — completed the week's separate learning cycle

**Scope and work completed**

I completed the week's planned learning exercises: one benchmark pipeline, explanation comparisons and a controlled shortcut experiment. The [weekly summary](WEEKLY_COMPLETION_2026-10-23.md) records the deliverables and questions prepared for my advisor. A second real dataset, author-code SOM2IMG experiments and genomic evaluation remained later milestones.

- Recorded Experiment 03/04 settings before their runs. Preserved the fitted Wisconsin classifiers, earlier outputs, MILE work and archives.
- Added [frozen-model explanation comparisons](../experiments/03_wisconsin_explanations/README.md): Grad-CAM, IG, Kernel SHAP and a clearly labeled LIME baseline-relative local effect, using the same fixed CNN score and feature mapping. Checked 12 deterministic validation cases plus one extra incorrect case, without test-set explanations or retraining.
- Added feature replacement against matched random controls, reference sensitivity, two SHAP/LIME sampling seeds, head/full model resets, numerical completeness, and an exact linear-model attribution control. Kept raw vectors and diagnostics, including negative results.
- Added a [synthetic shape-versus-background experiment](../experiments/04_shortcut_sandbox/README.md), with two training regimes, three initialization seeds and paired test backgrounds. Explained eight fixed examples per regime with two IG references.
- Added source-backed reports, five figures, 17 new tests, exact dependency additions, a short student self-check and questions for the professor. Original paper reading is labeled abstract-level; detailed implementation settings were checked against official APIs.

**Findings and limits**

On 12 learning cases, IG/SHAP absolute-magnitude Spearman averaged 0.985. Five-feature replacement exceeded the matched random mean in all 12 cases for IG, SHAP and LIME local effects, but only five for the Grad-CAM coordinate proxy. This questions our naive use of a regional map as input-feature ranks; it is not a general comparison or rejection of Grad-CAM. Correlated features and unrealistic replacement combinations limit this diagnostic. [Exact explanation results](../experiments/03_wisconsin_explanations/results.json).

All three correlated-background synthetic models achieved 100% / 50% / 0% accuracy on correlated / independent / reversed backgrounds; all three independent-background controls achieved 100% on all views. The background-only IG reference nevertheless gave 100% foreground absolute mass in all 16 explained model/example pairs. Its unchanged-background reference mathematically excludes those pixels' contributions; this is a different contrast, not an IG software failure. Absolute mass also discards sign. [Exact synthetic results](../experiments/04_shortcut_sandbox/results.json).

The findings inform the experimental design but do not establish biological usefulness or novelty. They also give no basis for universal claims about CNN inferiority or the value of tabular feature rankings. The next advisor discussion needs to clarify the interpretation target and intended SOM2IMG implementation.

**Verification**

- All 50 tests passed. Analytic controls check IG/SHAP linear effects, LIME coefficient scaling, Grad-CAM layer behavior, fixed targets, paired data identity and exact saved metrics.
- Full independent reruns produced byte-identical Experiment 03 and 04 JSON results. Frozen Experiment 01/02 outputs and model hashes were checked; no Wisconsin model was retrained.
- All five new figures visually inspected. Documentation links, bibliography-key uniqueness, code/protocol hashes, dependency pins, package consistency, offline installation dry-run and Git whitespace checked before publication.
- New-code Ruff checks pass. A broader check found five pre-existing style findings in `audit_mile_metadata.py` and its tests; those files were left unchanged. This is not a claim of repository-wide lint cleanliness.
- Raw data, checkpoints and the environment remain ignored. Only code, protocols, compact results, figures and documentation are published.

**Next steps**

Review the weekly learning guide with the professor. Define the next fixed/random/SOM layout comparison only after clarifying the intended author implementation and mapping/collision details. Spambase and accepted genomic experiments follow as separate milestones. I will review the mechanisms and limitations of these exercises before expanding the study; the related-work review also remains incomplete.

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

Following my advisor's recommendation, I began with Wisconsin Diagnostic to understand the pipeline one stage at a time. Spambase remained a possible second benchmark. This learning exercise supports the planned genomic study rather than replacing it; previous research and archives were preserved.

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

This stage covered data acquisition and fixed-grid encoding only. Unique IDs do not establish patient independence. The learning split is not a new sealed thesis holdout, especially given prior archived Wisconsin work. Obtain author implementation/version/splits/collision details before a SOM2IMG reproduction. Next: walk through the notebook, then train simple tabular baselines and a small fixed-grid CNN as a separate experiment.

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

Clarify which benchmark and paper version the advisor intends; request the author implementation or missing specification, particularly collision handling and fold boundaries. Consider Wisconsin Diagnostic only as a separate technical check, not a gene-expression cohort. Continue the dataset acceptance work alongside this technical exercise. The review alone does not resolve cohort eligibility or establish an interpretation benefit.

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

Patient-origin information is better documented, but no complete patient/center linkage map has been recovered. MILE remains conditional pending resolution of these data-quality and independence questions. Source details and reproducible calculations are in the updated [feasibility report](../week02/GSE13159_FEASIBILITY.md) and [audit JSON](../week02/GSE13159_AUDIT.json).

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

MILE remains a conditional candidate. This stage was limited to metadata and platform annotation, before expression-matrix acquisition or model training. Full preprocessing provenance, current identifier reconciliation, expression-level checks, and an appropriate interpretation of confounding remain open. This work stays within Week 2 dataset selection; the other candidates still need their own audits.

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

1. Review the [31 October findings](WEEKLY_FINDINGS_2026-10-31.md), focusing on the distinction between input values, feature neighborhoods and model attributions.
2. Confirm which SOM2IMG code/paper version the professor intends. The author-code layout exercise is now complete; define a frozen-model explanation comparison without tuning against observed scores.
3. Preserve the completed TCGA-BRCA audit and pilot, and resolve biological target/cohort questions before genomic interpretation or confirmatory experiments.
4. Consider a second real learning dataset only after the first pipeline's limitations are understood; do not count synthetic bars as a real dataset or Xena and TCGA as independent datasets.
