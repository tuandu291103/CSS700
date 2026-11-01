# Weekly research notes: 31 October 2026

**Yu Dinh**

No faculty meeting took place this week. These notes summarize the SOM2IMG experiment and the questions I would like to discuss at the next meeting.

## Work completed

This week I used the supplied [SOM2IMG repository](https://github.com/Kerbash/tab2img-bench/tree/dc0ef691db733ebcba4d13b53ca58e258f76ef0a) to examine how a learned feature layout affects the Wisconsin Diagnostic representation. I first checked the implementation, including feature scaling, collision handling and image encoding. I then compared one fixed layout, three lossless random layouts and three SOM layouts using three CNN initialization seeds for each layout, giving 21 models.

The experiment used 318 records for fitting, 79 for checkpoint selection and the existing 86-record validation partition for exploratory evaluation. The previous learning-test partition was excluded from transformations and scoring. I recorded the [protocol](../experiments/07_som2img_layout/protocol.json) before running the comparison and retained every model result rather than selecting the best seed.

The [full report](../experiments/07_som2img_layout/README.md), [results](../experiments/07_som2img_layout/results.json) and [runner](../scripts/som2img_learning.py) contain the methods and supporting evidence. All 94 project tests and 18 upstream tests passed. A separate execution reproduced the JSON and four figures exactly in the recorded environment.

## Findings

### Feature organization and stability

All seven layouts retained all 30 features and allowed exact recovery of the scaled float64 values. SOM produced more similar neighboring feature profiles than the random controls. Mean adjacent absolute Pearson correlation was 0.580 for SOM and 0.416 for random layouts.

The exact neighborhoods were less stable than this similarity result alone might suggest. Mean neighbor-set Jaccard overlap across SOM seeds was 0.211, compared with 0.035 for random layouts. The mean/SE/worst versions of the same measurement were closer in SOM than in the fixed grid, but their distances remained within the descriptive random-layout reference interval. I therefore cannot conclude that the layout provides meaningful domain grouping from this check alone. [Layout methods and results](../experiments/07_som2img_layout/README.md#3-feature-preservation-and-layout-structure).

![Layout diagnostics across all seeds](../experiments/07_som2img_layout/layout_diagnostics.png)

### Prediction

| Layout | Mean accuracy | Mean macro F1 | Mean log loss |
|---|---:|---:|---:|
| Fixed | 97.67% | 0.9751 | 0.0543 |
| Lossless random | 98.32% | 0.9820 | 0.0744 |
| SOM2IMG | 98.97% | 0.9890 | 0.0536 |

SOM exceeded random-layout accuracy by 0.65 percentage points on average, equivalent to less than one additional correct prediction per run. The runs share the same 86 previously examined validation records, so they do not provide independent evidence of general superiority. All models reached the 200-epoch training limit. I did not refit tabular baselines in this experiment, and the scores should not be compared directly with last week's different test partition. [Prediction results](../experiments/07_som2img_layout/README.md#4-prediction-promising-small-exploratory-differences).

### Image values before classification

The authors document that their default uint8 conversion can wrap values outside the training range. I reproduced this behavior on our evaluation inputs: 28 values across nine records differed by more than 0.5 between default and clipped export. For example, a scaled `area_worst` value of 1.030766 became 0.023529 under the default conversion. [Author encoding code](https://github.com/Kerbash/tab2img-bench/blob/dc0ef691db733ebcba4d13b53ca58e258f76ef0a/src/tab2img/pipeline.py).

The classifier comparison used float32 inputs for every method, preserving the values to ordinary casting precision. I did not train quantized comparison models, so this audit does not measure how conversion changes accuracy or attribution. It does show why checking feature locations alone is insufficient: the values stored at those locations also need verification. [Encoding audit](../experiments/07_som2img_layout/README.md#5-separate-numerical-audit-a-pixel-can-change-before-the-cnn-sees-it).

## Relation to the thesis

Last week's [explanation comparison](../experiments/03_wisconsin_explanations/README.md) showed strong IG/SHAP magnitude agreement on 12 selected validation cases, with mean Spearman correlation of 0.985. In the five-feature replacement check, IG exceeded the random mean in all 12 cases, while the naive Grad-CAM coordinate proxy did so in five. The [synthetic shortcut experiment](../experiments/04_shortcut_sandbox/README.md) also showed that a reference leaving the background unchanged could produce foreground-only attribution even when prediction depended on the background.

This week's experiment adds a check of the representation itself. I can now trace features through the authors' SOM layout and compare trained models under controlled conditions. The remaining question is whether the more organized layouts produce explanations that are more stable, more faithful to the model, or more useful for understanding the data. None of those benefits has yet been established for the new SOM models.

## Questions for the next meeting

1. Should the thesis use the supplied 51-dataset code version as its reference, or also reproduce the configuration in the earlier manuscript?
2. Is attribution stability and faithfulness across the fixed, random and SOM layouts an appropriate next experiment?
3. For later genomic work, what evidence should define a useful interpretation: predefined pathway coherence, expert assessment, robustness, or another criterion?

## Next steps

I will keep the fitted models unchanged and write an attribution-comparison protocol using common cases, target scores and references. The comparison will decode explanations to feature names and include reference sensitivity and matched feature-replacement controls. The TCGA/Xena pilot remains available for the later genomic study, once the interpretation target and cohort limitations have been agreed.

The current work follows the separate-experiment learning phase discussed with my advisor. The broader multi-dataset study and biological validation remain to be completed.
