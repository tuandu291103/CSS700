# Weekly learning milestone — 23 October 2026

Scope recorded 23 October 2026 from the current research notes and the user's instruction to finish this week's work. This is the one-dataset learning cycle, not the full thesis. One dataset satisfies the professor's "1–2" starting direction; Spambase and author-code SOM2IMG reproduction were explicitly recorded as later milestones.

| Deliverable | Evidence / status |
|---|---|
| Obtain and understand one SOM2IMG benchmark | Complete: [Wisconsin card and walkthrough](../experiments/01_wisconsin/README.md) |
| Understand tabular-to-image placement and feature tracing | Complete: fixed-grid round trip and [SOM2IMG methodology review](../week02/SOM2IMG_REVIEW.md) |
| Understand and train CNN plus tabular baselines | Complete: [model comparison and CNN walkthrough](../experiments/02_wisconsin_models/README.md) |
| Learn Grad-CAM and input-level attribution; compare SHAP/LIME | Complete: [Experiment 03](../experiments/03_wisconsin_explanations/README.md) |
| Test explanations through perturbation, reference sensitivity and model randomization | Complete: [saved diagnostics](../experiments/03_wisconsin_explanations/results.json) |
| Demonstrate an object-versus-background shortcut under controlled conditions | Complete: [Experiment 04](../experiments/04_shortcut_sandbox/README.md) |
| Brief findings, limitations, references and professor discussion notes | Complete: this summary and the linked reports |
| Verify outputs and publish completed work | 50 tests pass; both new experiments reproduce exactly; final publication check recorded in the session handoff |

Completion means delivering reproducible exercises and a learning guide. It cannot certify the student's personal mastery, demonstrate thesis novelty, or resolve unavailable author-code details. No final thesis dataset acceptance, genomic model, clinical finding, or actual SOM2IMG reproduction will be claimed.

## Brief meeting summary

- **Data and conversion:** audited 569 Wisconsin Diagnostic records and 30 morphology features; verified exact scaled-value tracing through a 5×6 fixed grid. This is a SOM2IMG benchmark learning dataset, not gene-expression data or a learned SOM layout. [Dataset, provenance and encoding evidence](../experiments/01_wisconsin/README.md).
- **Prediction:** logistic regression scored 83/86 correct on the learning test set; three CNN runs scored 83/83/82. Similar accuracy did not imply equal probability quality, and neither proves better interpretation. These are fixed-settings exploratory results, not a universal model ranking. [Model report](../experiments/02_wisconsin_models/README.md).
- **Explanation:** on 12 preselected validation cases, IG and Kernel SHAP agreed strongly in magnitude rankings (mean Spearman 0.985). Using Grad-CAM coordinates directly as feature rankings gave much weaker feature-replacement results. We traced an incorrect prediction back to named features as well as explaining correct predictions. [Explanation report and exact outputs](../experiments/03_wisconsin_explanations/README.md).
- **Shortcut:** three synthetic background-correlated CNNs went from 100% to 0% accuracy after background reversal; independent-background controls stayed at 100%. A background-only IG reference nevertheless placed 100% of absolute attribution on the shape in both groups. That is a reference-dependent contrast, not proof of object-based reasoning. [Controlled experiment](../experiments/04_shortcut_sandbox/README.md).
- **Research implication:** coordinate tracing, model faithfulness, and domain meaning must be evaluated separately. A visually clear map alone does not settle all three. This is a lesson informing our thesis design, not a claim that we invented tracing or proved a biological benefit. Related foundations: [Grad-CAM](https://arxiv.org/abs/1610.02391), [IG](https://arxiv.org/abs/1703.01365), [sanity checks](https://arxiv.org/abs/1810.03292).

## Are we on track?

**Yes, for the professor's initial separate-experiment learning phase.** We now have one end-to-end benchmark exercise and a separate controlled vision demonstration, with protocols, code, named-feature tracing, diagnostics and sources. This does not mean the autumn–spring thesis is complete or that later milestones should be skipped.

The useful result is methodological: in this small setting, naive spatial-to-feature attribution and reference-dependent overlap can be misleading. We have not established that T2I improves biological interpretation, that CNNs always underperform traditional models, or that these observations are novel. Existing work already links image explanations to nonimage features; see [DeepFeature and the prior-art review](../week02/CLAIM_EVIDENCE.md).

## What changed in our understanding?

1. **Conversion is not explanation.** A reversible mapping names each pixel; its brightness is just a scaled measurement.
2. **Localizing a region is not attributing one input.** A convolutional activation mixes neighbors, even when its grid size happens to match the input.
3. **SHAP and LIME are not restricted to tabular classifiers.** Here they query the same CNN through a tabular-to-image wrapper, separating method differences from model differences. [SHAP](https://arxiv.org/abs/1705.07874), [LIME](https://arxiv.org/abs/1602.04938).
4. **Ask “compared with what?”** The reference changes the question; an explanation may account correctly for that contrast without demonstrating robustness. [IG reference definition](https://captum.ai/api/integrated_gradients.html).
5. **Good accuracy is not enough.** The paired synthetic intervention exposed a failure that the original test view concealed. Its results apply to that controlled task, not automatically to Wisconsin or genomics.

## Questions for the professor

1. Can we obtain the intended SOM2IMG code, exact paper version, Wisconsin release, fold-local preprocessing and collision rule? Our fixed grid is only a teaching baseline. [Outstanding reproduction details](../week02/SOM2IMG_REVIEW.md).
2. Should the next isolated experiment compare fixed/random/SOM layouts on Wisconsin, with a clearly defined regional-versus-input explanation question, before adding Spambase?
3. For the eventual thesis, what would count as useful **domain** interpretation beyond model faithfulness: known feature groups, expert review, pathway annotations, or another explicitly defined target? Biological claims still require accepted genomic data and suitable controls.

No meeting or advisor approval is implied by this prepared discussion note.

## Five-minute walkthrough to practice

Try answering without reading the reports first:

1. Why can a valid scaled validation value fall outside [0, 1]? (Training-only extrema; no clipping.)
2. What does feature index 21 map to, and what does a positive score mean for the incorrectly predicted benign case? (`texture_worst`, row 3/column 3; support for the model's original benign target.)
3. Why is a Grad-CAM cell not necessarily the contribution of the input at that coordinate? (Convolutional receptive fields mix neighboring inputs.)
4. Why can a background-dependent model have zero IG on all background pixels? (Those pixels equal the selected reference, so their input-minus-reference factor is zero.)
5. Why are these not thesis discoveries yet? (Small learning subsets, selected references/perturbations, one synthetic design, no accepted biological evaluation or novelty study.)

## Verification and next boundary

All 50 tests pass, including 17 new analytic/control/output checks. Full reruns of Experiments 03 and 04 generated byte-identical JSON results. All five new figures were visually inspected; dependencies are pinned and an offline installation dry-run and package check passed. Earlier experiment outputs and genomic/archived work were preserved.

New-code Ruff checks pass. A repository-wide check also revealed five pre-existing style findings in the untouched MILE audit script/tests; they are not new experiment failures and were not silently changed in this milestone.

This week's scoped deliverables are complete. Next is a guided review and a separately specified layout experiment once the SOM2IMG reproduction details are available—not further tuning of the already observed learning-test results.
