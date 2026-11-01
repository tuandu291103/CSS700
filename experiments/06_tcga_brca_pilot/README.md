# Experiment 06 — first TCGA-BRCA prediction pilot

26 October 2026. This preliminary genomic comparison follows the [TCGA-BRCA acquisition audit](../05_tcga_brca_audit/README.md) and evaluates a fixed-grid CNN alongside tabular classifiers.

## Relation to the research plan

My advisor recommended studying the pipeline through separate experiments and later suggested TCGA and UCSC Xena. This pilot extends that work to gene-expression data, following the Wisconsin and synthetic-vision studies.

| Research objective | Progress in this pilot |
|---|---|
| Understand 1–2 SOM2IMG benchmark datasets first | [Wisconsin audit/walkthrough](../01_wisconsin/README.md) completed one benchmark. TCGA is a professor-suggested genomic extension, not another claimed SOM2IMG benchmark. |
| Understand tabular-to-image conversion | Exact fixed-grid round trips on Wisconsin and now 256 named genomic identifiers. This layout does **not** group similar genes; learned layouts remain separate work. |
| Understand CNNs and compare explanation techniques | [Wisconsin explanations](../03_wisconsin_explanations/README.md) and [shape/background controls](../04_shortcut_sandbox/README.md) completed. No genomic explanations were computed in this pilot. |
| Run isolated experiments before thesis claims | Data audit, preparation, prediction and later interpretation remain distinct stages with saved protocols. |
| Support conclusions with sources and quantitative evidence | Source versions, code/protocol hashes, class-aware metrics and references accompany the results. |

The broader plan for at least three datasets, comparative methods and quantitative interpretation measures remains incomplete. The three classifier families here share one cohort and one fixed encoding. Learned-layout comparisons, biological usefulness criteria and independent validation still require a thesis-level protocol.

## Cohort and preprocessing

The pilot uses the audit's conservative cohort policy: 1,170 samples from 1,059 participants, comprising 1,059 primary tumors and 111 solid-tissue normals. Clinical annotations, prior-treatment fields and sample-level preservation flags determine eligibility independently of expression scores or model performance. The [protocol](protocol.json) and preceding audit describe the exclusions and their implications for the study population. This cohort choice applies to the pilot; final thesis eligibility remains under discussion.

Two scenarios were specified before model fitting:

| Scenario | Training: tumor / normal | Validation: tumor / normal | Test: tumor / normal |
|---|---:|---:|---:|
| Patient-grouped split | 739 / 77 | 160 / 17 | 160 / 17 |
| BH collection-site holdout | 774 / 32 | 137 / 6 | 148 / 73 |

The first scenario has 816/177/177 samples and 739/160/160 participants. The second has 806/143/221 samples and 774/137/148 participants. Participant and case-UUID overlap between partitions is zero in each scenario. Tumor–normal pairs stay together. BH was chosen from the **earlier metadata audit**, because it supplies most normal samples, not because of model results. All BH participants are absent from training and validation in that scenario. These scenarios reuse participants in different roles and are dependent checks within TCGA, **not external validation**. Barcode hierarchy: [GDC documentation](https://docs.gdc.cancer.gov/Encyclopedia/pages/TCGA_Barcode/).

Each scenario independently selects the 256 highest-variance nonconstant expression identifiers **using its training rows only**, then standardizes using training means/population standard deviations. Source values were already `log2(TPM+1)`; no second log, imputation, clipping or outcome-based gene selection was used. This training-only boundary follows [scikit-learn's leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage). The two selected sets share 243 identifiers, a descriptive preparation result, not an attribution-stability score.

The selected IDs are sorted lexicographically into a **16×16 row-major grid**. This is an arbitrary fixed layout, not genomic coordinate order, a similarity map or SOM2IMG. The float32 table and flattened image recover exactly. [Preparation evidence](preparation.json) records every gene ID, source symbol, coordinate and training statistic. Patient IDs, labels and clinical covariates are excluded from predictors.

## Models and evaluation

- Majority/prior control, predicting the training-majority class with training class-prior probabilities.
- Logistic regression: `C=1`, L2 default, `lbfgs`, at most 2,000 iterations, balanced class weights.
- Random forest: 300 trees, minimum leaf size 2, square-root feature sampling, balanced class weights, three fixed seeds.
- Fixed-grid CNN: the existing 1,282-parameter two-convolution architecture, weighted cross-entropy, Adam, learning rate 0.003, at most 150 epochs, patience 25, three fixed seeds.

Training class counts determine weights. The CNN restores the lowest **validation weighted cross-entropy** checkpoint; no test arrays are accepted by its fitting function. Every model in **both** scenarios was fixed before either test set was scored. The threshold is fixed at `P(tumor)>0.5`, with no tuning/calibration or best-seed selection. Sources for implementation: [logistic class weights](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [forest settings](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), [PyTorch weighted loss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html).

### Results

All figures below are measured in this exploratory run, from [results.json](results.json). Forest/CNN entries are **means across three seeds**, not an ensemble. SD below describes initialization variation on the same split, not sampling uncertainty or a confidence interval. Balanced accuracy is the average of the two class recalls. [Metric definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html).

| Scenario / model | Accuracy | Balanced accuracy | Macro-F1 | ROC-AUC | Log loss ↓ |
|---|---:|---:|---:|---:|---:|
| Patient-grouped: majority/prior | 90.40% | 50.00% | 0.475 | 0.5000 | 0.3163 |
| Patient-grouped: logistic regression | 100.00% | 100.00% | 1.000 | 1.0000 | 0.0049 |
| Patient-grouped: random forest | 100.00% | 100.00% | 1.000 | 1.0000 | 0.0309 |
| Patient-grouped: grid CNN | 97.18% | 98.44% | 0.928 | 0.9972 | 0.0857 |
| BH holdout: majority/prior | 66.97% | 50.00% | 0.401 | 0.5000 | 1.0928 |
| BH holdout: logistic regression | 97.29% | 96.24% | 0.969 | 0.9989 | 0.0618 |
| BH holdout: random forest | 94.72% | 92.01% | 0.938 | 0.9994 | 0.1348 |
| BH holdout: grid CNN | 95.93% | 96.38% | 0.955 | 0.9927 | 0.1321 |

CNN accuracy sample SD is 0.56 percentage points on the grouped test and 1.20 points on BH; balanced-accuracy SD is 0.31 and 0.51 points respectively. Forest BH accuracy SD is 0.26 points and balanced-accuracy SD 0.40 points. All individual runs and other SDs remain in the results file.

**Exact error counts:** on the grouped test, logistic regression and all forests score 177/177; the CNNs score 171/177, 173/177 and 172/177, missing six/four/five tumors and none of the 17 normals. On BH, logistic regression scores 215/221 (five normals and one tumor missed); forests score 210/209/209 out of 221, missing 11/12/12 normals and no tumors; CNNs score 209/213/214, missing 0/3/2 normals and 12/5/5 tumors. Confusion-matrix order is `[normal, tumor]`, rows true and columns predicted.

## Analysis and limitations

1. **Representation.** The table and reversible image contain identical features, with participant-disjoint partitions. The fixed grid does not encode biological neighborhoods.
2. **Class imbalance.** The majority control achieves 90.40% grouped-test accuracy while recognizing no normal samples. The BH forest also combines high AUC with lower normal recall at the fixed threshold. These results support reporting class-aware measures alongside accuracy.
3. **Model comparison.** Logistic regression has higher accuracy and lower log loss than the CNN in both scenarios. The CNN's mean BH balanced accuracy is approximately 0.14 percentage points higher than logistic regression, a small difference that does not establish superiority. Forest and CNN errors differ by class.
4. **Generalization.** The grouped test contains only 17 normal samples from one selected cohort. Identity and preprocessing checks address specific sources of leakage, but do not rule out acquisition or biological confounding. Holding out one collection site also provides limited evidence about other sites or clinical settings.
5. **Interpretation.** This pilot evaluates prediction rather than explanation quality. The thesis will need separate comparisons of faithfulness, stability and gene-group meaning using appropriate tabular and T2I baselines.

The BH development set has only **six validation normals**, and its CNN validation curves fluctuate substantially. Five of six CNN runs reached the 150-epoch cap, so optimized convergence is not claimed. Grouped checkpoint epochs were 144/149/150; BH epochs were 126/132/94. We did not increase the budget after inspecting test scores. Class-weighted probabilities are not calibrated clinical risks. See [all learning curves](learning_curves.png).

## Problems fixed and verification

- A pre-training standardization assertion exposed float32 **reduction** rounding (~1.3×10⁻⁶), not an incorrect scaler. Diagnostic reductions now accumulate in float64; the model inputs, inclusion policy, splits and protocol did not change. The original preflight partition file is retained locally and is byte-identical to the completed one.
- Synthetic tests change held-out values to extremes and verify that selected genes, training means/scales and training inputs are unchanged. Other checks cover site/patient/case separation, deterministic tie-breaking, minority-class metrics and weighted checkpoint restoration.
- All **82 tests available at this stage passed**, including 14 new pilot tests. Ruff checks for the new code and Git whitespace checks passed; older MILE style warnings remained unchanged.
- Preparation was rerun from the raw files to a temporary directory: public preparation evidence and private partitions match byte-for-byte. Full training was independently rerun with the same settings; aggregate results and private predictions also match byte-for-byte.
- All 48 train/validation/test metric sets were recomputed from saved local predictions. Every saved classifier was reloaded and checked against its validation/test predictions. The learning-curves figure was visually inspected.
- Raw data, patient-level partitions/predictions, arrays and checkpoints stay ignored. Public artifacts contain aggregate results and gene-level mappings, not individual clinical histories. Older experiment results were preserved.

## Follow-up and advisor questions

The proposed follow-up uses the existing models for a small validation-only explanation study. CNN input attributions will be traced to Ensembl IDs and compared with tabular explanations under specified reference and perturbation conditions. Since the fixed-grid order is arbitrary, biological relationships cannot be inferred from neighboring positions. Testing that possibility requires a separate comparison of fixed, random and learned layouts.

Discuss with the professor:

1. Is primary-tumor versus solid-tissue-normal classification a useful **learning bridge**, or should the eventual thesis target change to a more informative biological question?
2. Is the conservative eligibility policy appropriate for that final question, and what collection-site/external-cohort controls are required?
3. What exactly counts as useful interpretation: known gene groups, pathway consistency, expert usefulness, or another quantifiable target? Which tabular gene-group baseline should we compare against?
4. Can we obtain the intended SOM2IMG implementation/version and collision policy before claiming reproduction?

This first genomic pilot established participant-disjoint evaluation, training-only gene selection and reversible 16×16 encoding. It compared all planned classifier runs and documented collection-site sensitivity. The results provide a basis for explanation experiments, rather than evidence of a new biological finding.

## Reproduction and evidence

- [Protocol](protocol.json), [preparation/mapping](preparation.json), [aggregate results](results.json), [training figure](learning_curves.png).
- [Preparation code](../../scripts/prepare_tcga_pilot.py), [model code](../../scripts/tcga_pilot_models.py), [tests](../../tests/test_tcga_pilot.py).
- [TCGA/Xena provenance](../../docs/TCGA_XENA_SOURCE_MANIFEST.json), [full input audit](../05_tcga_brca_audit/results.json), [research log](../../docs/RESEARCH_LOG.md).

Preparation used bundled Python 3.12.14/NumPy 2.3.5. Modeling used the existing [pinned research environment](../../requirements-learning.txt), including NumPy 2.5.4, scikit-learn 1.9.1 and PyTorch 2.14.1, because the bundled runtime lacks the modeling libraries. Both stages record versions and hashes. Reproducibility claims apply to this recorded local environment, not every platform.

With raw files available, run from the repository root. The default commands refuse to overwrite frozen artifacts; for an independent rerun, supply fresh `--output-dir`, `--public-output`, `--data-dir`, `--preparation` and `--artifact-dir` paths as appropriate.

```bash
/Users/dudinh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m scripts.prepare_tcga_pilot
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLCONFIGDIR=.cache/matplotlib XDG_CACHE_HOME=.cache .venv/bin/python -m scripts.tcga_pilot_models
MPLCONFIGDIR=.cache/matplotlib XDG_CACHE_HOME=.cache .venv/bin/python -m unittest discover -s tests -q
```
