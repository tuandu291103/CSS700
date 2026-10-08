# Thesis project plan for Autumn 2026 Winter 2027 and Spring 2027

Yu Dinh | CSS700 | Revised 7 October 2026 | Draft for advisor review

This plan responds to the professor's meeting notes by expanding the study to three public datasets, three primary image encodings, three explanation algorithms, and quantitative tests of explanation quality. It retains the original thesis topic: evaluating feature-level interpretability for tabular-to-image CNN classifiers. It does not assume that images are inherently interpretable or that the proposed framework will outperform every comparator.

## Research problem and proposed contribution

Converting named tabular variables into an image changes how their contributions are represented. A heatmap over a location must be connected to the original feature name, its value, the target class, and the exact transformation. If several variables share a pixel, a regional score does not identify their separate contributions. Even with a one-to-one mapping, CNN receptive fields and spatial heatmap interpolation can spread relevance across neighboring features. Correct coordinate lookup is therefore necessary but insufficient for a faithful explanation.

This is a specific, testable problem, not a claim that every transformation loses information. A bijective value-preserving encoding can preserve the scaled input exactly. It does not make Grad-CAM an exact decomposition of the prediction; Grad-CAM was designed as a coarse localization method [11].

The proposed contribution is a mapping-aware tracing and evaluation framework that records feature provenance, detects ambiguous mappings, and tests the reliability of feature-level explanations across encodings. On a one-to-one map, ordinary inverse lookup is already a baseline, not a novel algorithm. On collisions, the framework should report identifiable feature groups or reject unsupported individual scores; arbitrary division of one pixel's score is not a recovered explanation.

## Evidence and research questions

DeepInsight explicitly averages features that occupy the same pixel, providing a concrete example of lossy spatial encoding [2]. IGTD instead assigns unique pixels and arranges similar features nearby [3]. These establish different representation mechanisms, not evidence that every image explanation fails. Adebayo et al. show why attractive saliency maps require model-dependence checks [6]. Our inference is that tabular-to-image attribution needs both a mapping audit and independent behavioral evaluation.

1. Under which mappings can image attributions be traced unambiguously to named features?
2. How do Grad-CAM, Integrated Gradients, and Occlusion differ in faithfulness, stability, and cost on the same predictor?
3. Does preserving feature neighborhoods improve explanation quality relative to fixed and random layouts?
4. Which conclusions persist across datasets, training seeds, and replacement references?

SOM2IMG is an encoding algorithm, not one dataset. It arranges feature columns according to their behavior across training rows. This can group similar variables; it does not directly cluster patients, emails, or beans, and proximity does not establish causality. Human comprehension is a separate outcome and would require an approved, task-based user study before making an “easier for people to understand” claim.

<!-- pagebreak -->

## Dataset selection and collection

Use the following three public numeric classification datasets. They cover three application settings and introduce a multiclass task without immediately adding categorical encoding complexity. They are all listed in Table I of the locally supplied SOM2IMG manuscript [1b]. Membership and preprocessing in the final IEEE version must still be checked.

| Dataset | Rows | Features | Classes | Purpose and status |
|---|---:|---:|---:|---|
| Breast Cancer Wisconsin Diagnostic [8] | 569 | 30 | 2 | Small biomedical pilot with related measurements; existing Grid experiments |
| Spambase [9] | 4,601 | 57 | 2 | Email statistics, sparse values and duplicate groups; existing validation-only Grid study |
| Dry Bean [10] | 13,611 | 16 | 7 | Larger multiclass morphology task; proposed third dataset, not yet acquired for this study |

Use the tabular Dry Bean measurements, not the original photographs. The manuscript reports 13,612 rows, whereas UCI reports 13,611; reconcile the downloaded file, version and row count before accepting a dataset manifest. Breast Cancer and Dry Bean both contain image-derived measurements, so these are three application settings, not three wholly independent data-generation mechanisms. Spambase adds a non-morphological contrast.

Collection deliverables are a source URL and license, retrieval date and checksum, schema and class counts, missing-value and duplicate audit, feature dictionary, and immutable split manifest. UCI identifies these releases as CC BY 4.0 [8–10]. Verify actual files rather than treating catalog metadata as a completed audit. Fit scaling, imputation if needed, and learned layouts on training rows only. Retain Spambase's accepted grouped split; audit Dry Bean for duplicates before splitting. Keep new held-out test rows closed until the protocol is frozen.

## Comparative algorithms

| Comparison layer | Required methods | What the comparison isolates |
|---|---|---|
| Image encoding | Fixed Grid, SOM2IMG, IGTD | Effect of spatial organization with a matched CNN and preprocessing |
| Explanation | Grad-CAM, Integrated Gradients, feature-wise Occlusion | Regional, input-gradient and perturbation-based explanations of the same predictor |
| Layout control | Random feature permutation | Whether learned neighborhoods help beyond arbitrary placement |
| Tabular context | Random Forest and XGBoost with existing SHAP and LIME analyses | Predictive context and cross-model disagreement, not explanation ground truth |

Additionally, use model-agnostic SHAP on the exact frozen table-to-image-to-CNN predictor for a bounded same-model comparison. Freeze the target output and background first; do not equate TreeSHAP for a different model with validation of CNN explanations. Direct inverse-map Grad-CAM is the tracing baseline against which additional audit safeguards are evaluated.

<!-- pagebreak -->

## Quantitative evaluation

Three primary explanation measures are deletion faithfulness, seed stability, and known-signal recovery on controlled synthetic tests. Mapping checks and neighborhood preservation answer additional, distinct questions. No single number establishes complete interpretability.

| Measure | Quantification | Interpretation and limitation |
|---|---|---|
| Deletion faithfulness | Mean area over the perturbation curve for top-ranked features minus repeated random rankings; also compare bottom-ranked deletion | Larger positive contrasts support the chosen ranking under the specified intervention; not causal proof |
| Attribution stability | Per-sample Spearman correlation and top-k intersection divided by k across model seeds; separately vary layout seeds | Higher agreement indicates reproducibility, not correctness; report ties and constant vectors |
| Known-signal recovery | Precision at the known number of relevant variables and relevance-detection AUROC in controlled functions | Tests attribution against explicitly constructed relevance; does not supply ground truth for real datasets |
| Mapping integrity | Unique feature coverage, collision and drop counts, reconstruction error for invertible encodings | Technical acceptance gate; exact tracing is not evidence of faithfulness |
| Neighborhood preservation | Fraction of training-profile nearest neighbors retained among spatial neighbors, against random layouts | Tests grouping of similar features; does not establish semantic or causal groups |
| Predictive utility and cost | Macro F1, balanced accuracy, explanation seconds per sample and peak memory | Context and feasibility; prediction accuracy is not explanation quality |

For deletion, retain the original unperturbed target class throughout each curve. Define each point as the original target score minus its score after replacing a specified number of ranked features; average over the predeclared deletion fractions. Compare methods on identical rows, scores, fractions and replacement values. Use training-mean replacement as primary and training-median replacement as sensitivity unless a prospective protocol explicitly changes this before the new study.

Separate positive-support ranking from absolute-magnitude ranking: a large negative contribution is not positive evidence for a class. Existing pilot magnitude results remain historical. Freeze new ranking semantics before collecting new confirmatory results. Use both logit and probability diagnostics, but compare effect sizes primarily within the same predictor; raw logits are not calibrated across independently trained models.

Known-signal tests will first use independent inputs and a specified linear or additive function, with irrelevant variables and known signs. Add a separate correlated-input stress test with explicit group-level relevance. First test the known function itself, then distinguish learned-model error from explanation error. Do not use the data generator as ground truth for a CNN that learned another function.

Randomized-weight sanity checks [6], replacement sensitivity, and a small distribution-shift audit accompany deletion. Occlusion may be advantaged by evaluation using the same masking rule; report that dependence. ROAR provides a related removal-and-retraining benchmark [7], but a full retraining matrix is optional and must be budgeted separately.

<!-- pagebreak -->

## Experimental design and prior work comparison

Use one audited split per dataset, three training seeds (7, 42 and 2026), three primary encodings, and matched CNN capacity, optimization budget, scaling and checkpoint selection. Fit each learned layout on training data only and hold its layout seed fixed for the main training-seed comparison. This is a controlled comparison, not a reproduction of each paper's best published score.

The primary matrix contains 27 CNN fits: three datasets, three encodings and three training seeds. One fixed random-layout control per dataset adds nine fits, for 36. A separate layout-sensitivity check adds two further layout seeds for SOM, IGTD and random placement at training seed 42 on each dataset: 18 fits, for a planned total of 54 CNN fits. Explanation methods reuse checkpoints; they are not additional CNN fits. Hyperparameter search, failed diagnostics and any full PermGrad reproduction are outside this count and require separate budgeting.

Budget 18 tabular baseline fits across two model families, three datasets and three seeds, excluding tuning. Before launching the matrix, benchmark one training and explanation job per dataset, then estimate total wall time and memory. Freeze a small equal-budget validation search. For costly same-model SHAP, preselect a reproducible, class-stratified subset of at most 100 eligible evaluation rows per dataset, shared across methods, with a fixed training-only background and query budget. Retain approximation failures and report coverage.

## Related frameworks and fair comparison

PermGrad already combines Grad-CAM from a TINTO image branch with tabular-branch permutation importance and projects spatial saliency back to original features [4]. Its public repository is available. Therefore, “the first image-to-feature tracing framework” is not a defensible claim. Our proposed study emphasizes auditable local tracing, collision identifiability, and controlled explanation-quality tests; whether that combination is sufficiently novel remains an advisor and literature-review decision.

Compare the direct mapping stage under the same CNN, targets and samples. Compare the full hybrid PermGrad model separately if its reproduction passes an Autumn feasibility gate, because it changes both architecture and global attribution aggregation. Do not label a simple CNN back-map as a full PermGrad replication. A literature comparison is mandatory even if the full hybrid benchmark is not feasible.

DeepInsight and IGTD are established transformation comparators [2,3]. Table2Image v2 described an image/tabular interpretation approach; its August 2026 v3 reframes the contribution around learned proxy representations and reliability diagnostics [5]. Cite the exact version. It is related work, not assumed to provide a bijective feature map.

Report paired effects on the same rows with group-aware bootstrap intervals for duplicate groups, and report every training seed separately. Intervals from a single split are conditional on that split; three seeds and three datasets do not support broad population-level claims. Use error and class strata, and disclose all exclusions. Compare only common targets or use a predeclared common reference target across seed comparisons.

If the framework merely relabels an unchanged heatmap, it cannot improve faithfulness by relabeling alone. Its demonstrated benefit may instead be detecting invalid individual-feature claims and making failures reproducible. Negative or mixed results remain valid thesis outcomes.

<!-- pagebreak -->

## Quarter schedule and deliverables

The schedule uses approximately 32 working weeks across the three requested quarters. Blocks are research milestones, not asserted university calendar dates. Autumn is already underway; completed work is credited as pilot development, not silently counted as completed new-protocol conditions.

| Quarter | Work blocks | Required deliverables and exit gate |
|---|---|---|
| Autumn 2026 | Weeks 1–3: literature, scope and dataset audits. Weeks 4–7: multiclass loader, SOM and IGTD adapters, mapping tests. Weeks 8–10: explanation preflights, pilot timing and protocol freeze | Advisor-reviewed scope; three dataset cards; verified source versions; three encoders and three explainers pass pilot checks; frozen matrix and resource estimate |
| Winter 2027 | Weeks 11–13: remaining validation pilots and equal-budget model selection. Weeks 14–17: main matrix and locked-test execution. Weeks 18–20: controls, stability and sensitivity analyses | Complete 36-fit main/control matrix plus 18 layout-sensitivity fits or documented advisor-approved revision; comparable attribution bundles; metrics with uncertainty; first results chapter |
| Spring 2027 | Weeks 21–24: synthesis, related-work comparison and gap-directed replication. Weeks 25–28: full thesis and advisor revisions. Weeks 29–32: defense preparation, corrections and archive | Final evidence tables; limitations and defensible contribution statement; submitted thesis; defense slides and rehearsal; reproducibility package |

Write throughout: Autumn produces introduction, related work and methods drafts; Winter updates methods and writes results; Spring completes discussion, limitations and conclusion. Each research session ends with a short log of work, evidence, limitations and the next step. Push accepted code and documentation after verification; maintain a separate backup plan for ignored datasets and run artifacts.

## Decision gates and risk control

1. **Scope gate in Autumn:** confirm the third dataset, IGTD, required same-model comparison and feasible compute with the advisor. The meeting establishes minimum counts, not approval of these specific choices.
2. **Mapping gate:** obtain the final SOM2IMG paper and implementation; verify collision, padding and feature-preservation behavior. Permit group-level explanations when individual attribution is not identifiable.
3. **Technical gate:** preserve inference/gradient numerical equivalence, target consistency and feature-order tests. Finish the Spambase IG preflight before accepted IG results. Validate multiclass output and target selection on Dry Bean.
4. **Evidence gate:** freeze splits, backgrounds, sign conventions, metrics, query budgets and exclusions before opening new tests. Previously inspected Breast Cancer test results remain development evidence for decisions made afterward.
5. **Resource gate:** retain three datasets, three core encoders and three explainers. Cut optional hybrid reproduction, extra explanation methods and human studies before weakening core controls. Any reduction of the 54-fit plan is explicit, prospective and advisor-reviewed.
6. **Completion gate:** success requires traceable artifacts and an honest answer to the research questions, not a positive result. Claim superiority only for prespecified comparisons supported by effect sizes, uncertainty and sensitivity checks.

<!-- pagebreak -->

## Progress and immediate next steps

The repository already contains a leakage-aware pipeline, Fixed Grid encoding, named-feature tracing, three explanation implementations, perturbation evaluation, and tabular comparison code. Breast Cancer Grid experiments span three training seeds, with mean test accuracy 91.86% and macro F1 0.9141 under the earlier pilot protocol. Spambase has an accepted grouped split and a seed-42 validation model; its 690 test rows remain unscored. The latest coding session recorded 284 passing tests; this planning session does not claim a fresh test run.

The current findings motivate the study but do not resolve it. Spambase Grad-CAM deletion effects change sign between mean and median replacement: top-minus-random logit AOPC is −0.1163 and +0.0730 respectively. Grad-CAM and Occlusion rankings have low mean Spearman agreement under both references (−0.0777 and +0.0071). Those are validation-only, fixed-model results with differing attribution semantics, not proof that either method is wrong or that a new method is superior.

Source logs are `docs/logs/2026-09-06-grid-multiseed-stability.md`, `docs/logs/2026-10-04-spambase-faithfulness.md`, and `docs/logs/2026-10-05-spambase-occlusion.md`. A native SOM best-matching-unit diagnostic found collisions, but was not an author-code reproduction; see `docs/SOM2IMG_READINESS.md`.

We are on track in infrastructure and pilot development. We are not yet done with the expanded study: Dry Bean, accepted SOM2IMG and IGTD training, matched multiclass explanations, same-predictor SHAP and the new controlled matrix remain outstanding. Old pilot runs count toward the revised matrix only if their complete contracts match; otherwise they remain informative pilot evidence.

The next implementation milestone is a three-dataset readiness package: acquire and audit Dry Bean, pin the final SOM2IMG source/code, and specify compatible encoding adapters. Then complete the numerical explanation preflights and freeze the comparative experiment protocol. This session creates the plan; it does not download a third dataset, train a new model, open Spambase tests or contact the advisor.

## Reference verification and review request

The author's bibliography links the professor's IEEE document 11676815 to SOM2IMG and lists its DOI [1]. The final IEEE full text was not accessible in this review. Technical dataset and mapping details here are drawn from the separately supplied six-page manuscript [1b], not assumed identical to the final publication. Obtain the final PDF before asserting final-paper dataset counts, implementation settings or reproduced results.

SOM2IMG implementation remains gated on authoritative collision handling. If author code is unavailable, a documented SOM-based variant may be evaluated after advisor agreement, but must not be labeled a faithful reproduction. No collision policy will be silently invented.

Please review the proposed three datasets, three encoding methods, primary explanation metrics, 54-fit resource envelope, and the distinction between an empirical evaluation contribution and a new attribution algorithm. Confirm whether a full hybrid PermGrad benchmark is required or a controlled mapping-stage comparison plus explicit literature comparison is sufficient.

<!-- pagebreak -->

## References

Primary online sources checked 7 October 2026. Version-specific descriptions are intentional.

1. W. Selke and W. Kim. SOM2IMG Self-Organizing Map Based Tabular-to-Image Transformation for Deep Learning. IEEE IRI, 2026, pp. 171–176. DOI: 10.1109/IRI69576.2026.00042. [Author bibliography](https://faculty.washington.edu/kimw6/publication.html) links the [professor's IEEE reference](https://ieeexplore.ieee.org/abstract/document/11676815). Bibliographic identity verified; final full text pending.

1b. Locally supplied `SOM2IMG.pdf`, six-page anonymous manuscript. Table I and Methods on page 2; interpretability discussion on page 5. SHA-256: `b80c6d67ef05ea7ec806848c102b0b17bd3d0535f90fdcc673e4268031da18f9`. Local version, not a verified final IEEE PDF.

2. A. Sharma et al. DeepInsight A methodology to transform a non-image data to an image for convolution neural network architecture. Scientific Reports, 2019. [Article and mapping procedure](https://www.nature.com/articles/s41598-019-47765-6).

3. Y. Zhu et al. Converting tabular data into images for deep learning with convolutional neural networks. Scientific Reports, 2021. [IGTD article](https://www.nature.com/articles/s41598-021-90923-y); [author implementation](https://github.com/zhuyitan/IGTD).

4. F. Escalera-González et al. PermGrad Interpretable Hybrid Neural Networks with synthetic images for tabular data. Knowledge-Based Systems 350, 116507, 2026. [Article](https://doi.org/10.1016/j.knosys.2026.116507); [public author repository](https://github.com/manwestc/PermGrad). See the feature mapping and aggregation in Section 5; repository title differs from final article title.

5. S. Lee et al. Table2Image. [January 2025 v2](https://arxiv.org/abs/2412.06265v2) uses the title Interpretable Tabular Data Classification with Realistic Image Transformations. [August 2026 v3](https://arxiv.org/html/2412.06265v3) is titled Lightweight Tabular Learning with Generated Proxy Representations and Reliability Diagnostics. Do not interchange their claims.

6. J. Adebayo et al. Sanity Checks for Saliency Maps. NeurIPS, 2018. [Paper](https://arxiv.org/abs/1810.03292).

7. S. Hooker et al. A Benchmark for Interpretability Methods in Deep Neural Networks. NeurIPS, 2019. [Paper](https://papers.neurips.cc/paper/9167-a-benchmark-for-interpretability-methods-in-deep-neural-networks.pdf); [author ROAR implementation](https://github.com/google-research/google-research/tree/master/interpretability_benchmark).

8. UCI Machine Learning Repository. [Breast Cancer Wisconsin Diagnostic](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). DOI: 10.24432/C5DW2B.

9. UCI Machine Learning Repository. [Spambase](https://archive.ics.uci.edu/dataset/94/spambase). DOI: 10.24432/C53G6X.

10. UCI Machine Learning Repository. [Dry Bean](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset). DOI: 10.24432/C50S4B.

11. R. R. Selvaraju et al. Grad-CAM Visual Explanations From Deep Networks via Gradient-Based Localization. ICCV, 2017. [Primary paper](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html).
