# MILE acceptance protocol — draft v0.1

19 October 2026 | For advisor review | **No approval to begin confirmatory training**

## Decision

Keep GSE13159 as a **conditional candidate**. Permit file/provenance checks now. A processed-data exploratory pilot may become reasonable after the technical gates below pass and its limitations are agreed with the advisor. Do not yet claim patient-independent validation, center generalization, or improved biological interpretation.

This document specifies our proposed design choices; they are not published findings or completed experiments. The [feasibility report](GSE13159_FEASIBILITY.md) separates observed evidence from assumptions, including the follow-up on the original article and two sample tables.

## Evidence now available

- All 2,096 accession records and 18 historical labels are accounted for. Specimen type and diagnosis are associated: all 448 CLL samples are blood and all 206 MDS samples are marrow. [Computed metadata audit](GSE13159_AUDIT.json).
- The original paper's Table 1 provides center-by-class counts, not an assay-level center map. Center 10 contributed 199 samples, all CLL. The Methods/Appendix describe patient-origin specimens and laboratory processing, but the reviewed article/appendix does not provide a complete patient/center linkage file. [Haferlach et al., Table 1 and Appendix](https://pmc.ncbi.nlm.nih.gov/articles/PMC5569671/).
- Two preselected full sample tables have exact GPL570 probe-ID coverage. Each contains 54,630 finite DQN3 values in [0, 1] and 45 missing values, all on `AFFX-` probes. These two checks do not establish the full cohort's missingness or independence. [GSM330929](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM330929), [GSM331453](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM331453), [computed statistics](GSE13159_AUDIT.json).

## 1. Acceptance gates and permitted claims

| Gate | Evidence required before passing | Current status / consequence |
|---|---|---|
| File integrity | Full processed matrix with source URL/hash; exact sample/probe identities; finite/missing-value inventory; duplicate-profile checks | Only two individual profiles checked; no training yet |
| Processing | Confirm selected column is DQN3; document quantile target, probe universe, and any extra transformations; distinguish supplied processing from our train-fitted steps | DQN3 definition supported; full release and exact upstream implementation not certified |
| Patient independence | Authoritative anonymized donor/specimen grouping or a sufficiently explicit one-profile-per-patient guarantee for the release, with documented limitations | Study-level patient origin supported; release-level grouping not independently verified |
| Center effects | Sample-to-center mapping with class support assessed for proposed held-out centers | Aggregate counts only; no center-held-out or external-center claim |
| Gene interpretation | Frozen annotation/version, retired-ID reconciliation, control/ambiguity rules, and eligible-gene background | 2016 snapshot characterized; final annotation not approved |
| Reuse and scope | Record access/reuse requirements; agree exploratory versus confirmatory scope and inclusion criteria | Public access established, full requirements/scope decision pending |

If patient grouping stays unavailable, an advisor-approved exploratory **assay-level** analysis must say so; it must not be relabeled patient-independent. If center IDs stay unavailable, within-cohort results cannot establish center robustness. These are separate limitations: missing center IDs do not by themselves prove duplicate patients. No identifying patient information is needed or should be sought.

If processing or identity concerns cannot support the intended thesis claims, replace MILE for the confirmatory study. Do not repeatedly relax criteria to retain it. A dataset's convenience or CNN performance is not an acceptance criterion.

## 2. Proposed cohort and split rules

1. Start with the 18 deposited labels as the descriptive reference cohort. Preserve original labels and a separate versioned analysis-label dictionary. Do not rename the combined non-leukemia/healthy group as entirely healthy or adopt modern clinical labels by guesswork.
2. Freeze any exclusions before performance comparisons. Keep an accession-level reason for each exclusion. Do not silently remove CLL by selecting only marrow, or merge rare groups merely to improve scores.
3. If verified patient/specimen groups are available, keep related assays together in every outer and inner partition. If grouping is unavailable, explicitly record that fact; a duplicate-expression screen is not a substitute for donor linkage.
4. Define fold count, seeds, and rare-class support after the independent-group counts are known. Store the exact split manifest before tuning. Never split by accession order or use test results to select the cohort.
5. Fit new feature filtering/selection, scaling, image layouts, and hyperparameters only within the corresponding training partition. Use identical held-out groups and eligible features for compared model families. Do not present the original study's reported score as a directly comparable baseline under our different protocol.

## 3. Required specimen-type and center controls

These controls are proposed tests; none has been run.

- **Majority-class control:** learn the majority label from training data only.
- **Specimen-type-only control:** learn class frequencies conditional on blood/marrow using training labels only; use fixed additive-one smoothing and the training class prior for an unseen specimen type. This uses no gene expression and quantifies how informative specimen type already is.
- **Stratified reporting:** report per-class support and metrics separately for blood and marrow where the target class has observations. Mark unsupported class/type combinations as unavailable, not as successful or failed validation.
- **Restricted-cohort sensitivity:** if approved, run a separately named marrow-only analysis with its own target definition and splits. Compare methods *within* that cohort; do not interpret differences from the full-cohort score as the causal effect of specimen type.
- **Center sensitivity, only if IDs are recovered:** evaluate center-held-out predictions only with explicit training/test class support. Do not synthesize center IDs from class labels, accession order, GEO submitter contact details, or aggregate table counts.

The CLL/MDS specimen imbalance cannot be fully resolved by stratification or adjustment: the missing class/type combinations remain unobserved. A specimen-only baseline is a diagnostic check, not evidence that all remaining signal is disease-specific.

For predictive reporting, propose macro-F1, balanced accuracy, and per-class precision/recall with support and uncertainty; overall accuracy is secondary. Define handling of absent classes before running comparisons. Any resampling-based uncertainty must respect verified patient groups; repeated folds are not independent patients. These are project choices, not claims that any one metric establishes interpretation quality.

## 4. Proposed preprocessing and gene-mapping rules

- Identify columns by name. Keep `VALUE` (DQN3), `VALUE_DS`, detection p-values, and detection calls distinct. Never select a numeric column by position.
- Do not apply log2 or a second quantile normalization by default. The two checked DQN3 profiles already include zeros and ones; a transform requires an explicit rationale, finite-output checks, and a train/test application rule.
- Preserve missing entries during the audit. Flag all `AFFX-` rows and exclude them from the primary biological gene analysis under a declared control rule; 45 missing control values are not 45 missing patients. Inspect any non-control missingness in the full release before choosing an imputation/filtering policy.
- Keep a feature dictionary from original probe ID to every candidate gene ID, symbol, annotation source/version, mapping status, and exclusion reason. Never select the first gene in a multi-gene mapping without disclosure.
- Proposed primary biological comparison: exclude controls, unmapped probes, and multi-gene probes under the frozen annotation, then combine multiple eligible probes per gene by a declared within-sample median. Keep probe-level results as a sensitivity analysis. Apply the same input/mapping policy to image and tabular models; this rule has not been implemented or validated yet.
- Freeze the gene-set knowledgebase and eligible-gene background independently of held-out outcomes. Separate annotation used to construct a model from annotation used to evaluate explanations. Enrichment alone is not proof of a causal mechanism or added interpretability; the broader evaluation must also assess stability, faithfulness, and randomization checks. See the [claim evidence review](CLAIM_EVIDENCE.md).

Fixed-target normalization can be sample-independent in principle. A primary technical disclosure describes beta-target DQN normalization, but multiple implementation variants are possible; it is not a reproduction certificate for this GEO release. See the preprocessing follow-up in the [audit](GSE13159_FEASIBILITY.md#follow-up-original-study-and-two-profile-check).

## Next bounded work

1. Present this draft and the specimen/center limitations to the advisor. Ask whether a clearly labeled exploratory MILE pilot is useful if complete linkage remains unavailable. No advisor approval is recorded yet.
2. Check the complete processed release's size and provenance before acquisition; then audit IDs, missingness, and duplicate profiles without training. Freeze the final annotation policy separately.
3. Continue feasibility checks on the other candidate datasets. Do not let unresolved MILE metadata substitute for the proposal's requirement for three accepted independent datasets.

**Stop before model training:** record the acceptance decision, permitted claim scope, remaining limitations, cohort/feature manifests, split rules, and evaluation protocol. This draft is a concrete preparation step, not a completed experimental protocol or a positive model result.
