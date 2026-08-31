# Proposal summary and traceability

## Proposed contribution

Build a common feature-attribution tracing framework for tabular-to-image CNN classifiers. The
framework preserves each encoder's feature-to-pixel mapping, converts image-space explanation
values into original-feature scores, and compares explanation methods under controlled
conditions.

## Research comparison

| Dimension | Planned values |
|---|---|
| Datasets | Breast Cancer Wisconsin pilot plus at least one public classification dataset |
| Encodings | Fixed Grid Encoding; SOM2IMG; IGTD only if schedule permits |
| CNN explanations | Grad-CAM; Integrated Gradients; Occlusion Sensitivity |
| Tabular comparisons | SHAP; LIME; Random Forest; XGBoost |
| Repeats | At least three random seeds at expected scope |
| Predictive metrics | Accuracy; macro F1; precision; recall; confusion matrix |
| Explanation metrics | Faithfulness; Spearman correlation; top-k overlap; stability; runtime; group analysis |

## Acceptance levels

### Minimum

- one public dataset;
- documented Grid Encoding and feature-to-pixel mapping;
- one CNN that beats the majority-class baseline;
- all three feature-tracing methods;
- complete feature rankings and evaluation outputs.

### Expected

- at least two public datasets;
- Grid Encoding and SOM2IMG;
- at least three random seeds;
- SHAP, LIME, Random Forest, and XGBoost comparisons;
- the full proposed evaluation suite.

### Aspirational

- IGTD as a third encoding;
- broader Integrated Gradients baseline and robustness studies;
- publication preparation.

## Required deliverables

The reproducible pipeline must record preprocessing, feature mapping, model parameters, seeds,
explanation settings, and Integrated Gradients baselines. It must export trained models,
artificial-image examples, explanation maps, named-feature rankings, CSV results, performance
and faithfulness tables, stability results, comparison tables, and thesis-ready figures.

## Ethical position

The study uses public, non-PII datasets and makes methodological claims rather than clinical,
financial, or security decisions. Convincing-looking maps are not assumed valid; explanations
must be traced, perturbed, repeated, and critically analyzed.
