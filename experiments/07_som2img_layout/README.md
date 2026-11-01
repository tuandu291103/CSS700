# Experiment 07: SOM2IMG feature layouts and CNN classification

Completed 31 October 2026. This preliminary study uses Wisconsin Diagnostic to examine feature preservation, spatial organization and classification with the authors' SOM2IMG implementation.

[Protocol recorded before execution](protocol.json) · [All computed evidence](results.json) · [Runner](../../scripts/som2img_learning.py) · [Tests](../../tests/test_som2img_learning.py) · [Weekly discussion brief](../../docs/WEEKLY_FINDINGS_2026-10-31.md)

## Main finding

All three SOM layouts preserved the 30 features and placed more similar training-feature profiles next to one another than the three lossless random layouts. However, the exact neighbor relationships varied across SOM seeds. The mean accuracy difference between SOM and random layouts was 0.65 percentage points on 86 evaluation records. This difference is too limited to establish a general advantage, and explanation quality has not yet been evaluated for these models.

These results motivate separate evaluations of feature preservation, neighborhood structure, prediction and explanation quality. In particular, an organized layout may provide a basis for interpretation, but its usefulness depends on whether the explanations are faithful, stable and meaningful in the application domain.

## 1. What the newly supplied repository resolves

The experiment uses [Kerbash/tab2img-bench](https://github.com/Kerbash/tab2img-bench) at commit **`dc0ef691db733ebcba4d13b53ca58e258f76ef0a`** (30 October 2026). An unchanged checkout is stored locally under `.cache/tab2img-bench/`, with source-file SHA-256 hashes recorded in `results.json`. The repository is MIT-licensed. This project calls its implementation without maintaining a separate copy of the authors' source code.

| Earlier question | Evidence in this specific code version |
|---|---|
| Can we run the authors' method? | Yes: `create("SOM2IMG").fit(X_train)` and `transform(X)`; `feature_map_` exposes feature indices per pixel. [API and quick start][repo] |
| What happens when features share a best-matching unit (BMU)? | Default `spillover` sequentially moves a feature to the nearest free pixel by Manhattan distance. Ties follow the first free coordinate in array order. With our sufficient 5×6 grid, every feature is retained. A user-specified undersized grid can overwrite features; preservation must be checked, not assumed for every configuration. [SOM implementation][som] |
| Is fitting within each fold? | The current benchmark fits its outer scaler and each layout on the training fold. This resolves the code-version question; it does not retroactively prove how the older manuscript was run. [Benchmark implementation][evaluation] |
| Which random baseline isolates placement? | `SnakeRandom` places every feature once with the same featurewise scaling. `Random` samples with replacement and applies different, within-row scaling; it confounds placement, feature coverage and value transformation. [Controls][random] |
| Are benchmark method names exact author implementations? | The README describes several comparators as simplified implementations. Its `IGTD`, for example, differs from the original iterative pixel-swapping method. A comparison with the original methods would therefore require checking their respective implementations. [README][repo] |
| Is this identical to our six-page reading copy? | Not established. This repository describes *Which Tabular-to-Image Method, for Which Data? A Regime Map from 51 Datasets*, whereas the older supplied copy describes nine benchmarks. We keep manuscript, IEEE citation and code-version evidence separate. [README][repo]; [historical reading notes](../../week02/SOM2IMG_REVIEW.md). |

All 18 upstream tests passed locally. One checks rankings in the supplied benchmark outputs; passing it does not reproduce the underlying 51-dataset training study.

## 2. Our controlled setup and deviations from upstream

The experiment reuses the audited [UCI Wisconsin Diagnostic data](../01_wisconsin/README.md), comprising 569 samples, 30 cell-morphology measurements and benign/malignant labels. IDs and labels are excluded from the image inputs. The ten measurement families and their mean/SE/worst summaries follow the [original feature dictionary][dictionary]. This morphology dataset provides a small, documented example before a later genomic application.

- Split the old 397 training records into **318 fitting / 79 inner-stop** records, stratified with seed 20261031. Fit all scalers and SOM layouts on the 318 only.
- Use the old **86 validation records** for evaluation after all models are frozen: 54 benign, 32 malignant. These records were inspected previously, so they are exploratory evaluation data, not a new sealed holdout.
- Exclude the old **86 learning-test records** from this experiment's scaling, transforms, fitting and scoring. No new learning-test result is reported.
- Use one fixed row-major layout, three author `SnakeRandom` layouts, and three author `SOM2IMG` layouts (layout seeds 42/43/44). All have 5×6 cells and identical scaled feature values.
- SOM settings: 5,000 updates, sigma 1.0, learning rate 0.5, random initialization, rectangular topology, spillover assignment, no whitening or grid expansion.
- Cross each layout with CNN seeds 20261023/24/25: **21 models**, not 21 independent datasets. Reuse our 1,282-parameter CNN, Adam 0.003, weight decay 0.0001, batch 32, cap 200 epochs and patience 30. Select the lowest inner-stop cross-entropy, never an evaluation score.
- CNN inputs are **one-channel float32, without clipping or 8-bit quantization**. This deliberately differs from the upstream default three-channel uint8 pipeline, larger default CNN, 30 epochs and 10-fold weighted-F1 benchmark. It is a controlled adaptation using the author's layout code, not an exact paper-score reproduction.

The settings were recorded before training and remained unchanged after evaluation. Every seed is included in the report. All 21 runs reached the 200-epoch limit, with many selecting late checkpoints, so the experiment does not establish convergence beyond this budget.

## 3. Feature preservation and layout structure

Decoding each of the seven layouts recovered all 30 scaled values exactly in float64 on the fitting, inner-stop and evaluation partitions. The subsequent float32 cast introduces a small precision difference shared by all methods.

For SOM seeds 42/43/44, the 30 features initially selected **23/24/25 distinct best-matching units (BMUs)**. Spillover retained all features but moved **11/12/11** away from their own BMUs. A previous spillover can occupy a later feature's preferred cell, which explains why the number moved can exceed the number of duplicate BMUs. The largest moves were 5/3/3 Manhattan grid steps. Collision handling therefore affects the final neighborhoods even when it preserves every feature.

![First predeclared layout seed per method; colors identify measurement families, not values or importance](feature_layouts.png)

### Neighborhood diagnostics

Neighbor pairs are the 49 horizontal/vertical grid edges. Profile difference is the root mean square difference between two scaled features over the **318 fitting records**, averaged over edges. Absolute Pearson correlation is also computed on those fitting profiles. Both quantify training-layout organization, not independent biological validation.

Same-measurement distance averages the Manhattan distances for the 30 pairs sharing one measurement name across mean/SE/worst summaries. These groups are a modest dictionary-based check, not disease mechanisms or gene pathways.

| Layout | Neighbor profile RMSE ↓ | Neighbor absolute Pearson r ↑ | Same-measurement grid distance ↓ |
|---|---:|---:|---:|
| Fixed | 0.2149 | 0.4166 | 4.800 |
| SnakeRandom, mean of 3 layouts | 0.2222 | 0.4160 | 3.844 |
| SOM2IMG, mean of 3 layouts | 0.1662 | 0.5803 | 3.244 |
| Descriptive reference, 1,000 random layouts | 0.2273 | 0.4035 | 3.682 |

SOM improves both profile-similarity measures in this experiment, consistent with its training objective. This result is an implementation and organization check rather than independent evidence of domain meaning. The same-measurement distances for SOM (3.20–3.27) also fall within the random reference's central 95% interval (3.07–4.20), so this diagnostic does not establish semantic enrichment. The reference band describes variation among random layouts; it is neither a performance confidence interval nor a corrected hypothesis test.

![All seeds and random-layout reference](layout_diagnostics.png)

### Does the same neighborhood recur?

Compare the unordered sets of adjacent feature pairs using Jaccard overlap, `intersection / union`. This avoids treating a pure rotation/reflection as different neighborhood structure. Across the three pairwise seed comparisons:

- SOM2IMG mean neighbor Jaccard: **0.211**, range **0.153–0.256**.
- SnakeRandom mean neighbor Jaccard: **0.035**, range **0.021–0.054**.

The SOM layouts are more consistent than these random controls, although their exact neighbor sets still vary substantially. Three seeds provide only a limited estimate of layout variability. Further work should assess broader neighborhoods and the stability of feature explanations, rather than relying on one selected map.

## 4. Prediction: promising, small exploratory differences

Every model uses the same 86 evaluation records and threshold `P(M) > 0.5`. Means include **all** runs; ranges and raw predictions are retained in `results.json`.

| Layout | Models | Accuracy | Balanced accuracy | Macro F1 | Log loss ↓ |
|---|---:|---:|---:|---:|---:|
| Fixed | 3 | 97.67% | 97.30% | 0.9751 | 0.0543 |
| SnakeRandom | 9 | 98.32% | 98.03% | 0.9820 | 0.0744 |
| SOM2IMG | 9 | 98.97% | 98.96% | 0.9890 | 0.0536 |

The mean SOM–random accuracy difference is **0.65 percentage points**, equivalent to less than one additional correct prediction per 86-record run. SOM scored higher in four of the nine paired layout/CNN-seed comparisons and tied in five. Correct predictions ranged from 83–85/86 for fixed, 84–85/86 for random and 84–86/86 for SOM. Because these runs share one previously studied split, the differences do not establish statistical superiority or performance on another dataset.

![All model runs rather than a selected best seed](prediction_comparison.png)

The fitting sizes, checkpoint partitions and evaluation records differ from last week's learning-test comparison. Direct comparison with those scores would therefore be inappropriate. Tabular baselines were not refitted here, so the experiment does not compare SOM-CNN with tabular ML.

## 5. Separate numerical audit: a pixel can change before the CNN sees it

The authors document that the default `quantize=True, clip=False` settings can wrap out-of-range values during conversion to uint8. The audit reproduced this behavior using their unchanged `encode_images` function and measured its occurrence on the Wisconsin evaluation inputs. [Encoding implementation][encoding]; [README reproducibility notes][repo].

On our 86 evaluation rows × 30 features:

- **34/2,580 values**, across **11 records**, fell outside the range established by the 318 fitting rows. Such extrapolation is possible with training-only min–max scaling and is not, by itself, a data error.
- **28 values across nine records** differed by more than 0.5 between the upstream default uint8 output and clipped uint8 output.
- Actual example: ID **899987**, `area_worst`, scaled **1.030766** becomes **0.023529** under default uint8 conversion; clipped uint8 gives **1.0**. Our float32 route retains approximately **1.030766**.
- The synthetic documented example also reproduces: **1.02 → 4/255 ≈ 0.015686**; clipping gives 1.0.
- Maximum float32-vs-float64 discrepancy across evaluation values: **5.78×10⁻⁸**. Clipping avoids wraparound but still loses above-range magnitude; quantization also loses precision within range.

![Largest numerical changes; these examples do not estimate model performance effects](encoding_audit.png)

No matched clipped or quantized models were trained, leaving the effect on accuracy and attribution unmeasured. Every model in the layout comparison uses floating-point values and therefore avoids this uint8 conversion. Any later reproduction of the authors' benchmark will need to record whether it retains the historical encoding or changes it.

## 6. How this connects to last week's work

Our [fixed-model explanation exercise](../03_wisconsin_explanations/README.md) found strong IG/SHAP magnitude agreement but poor results when Grad-CAM coordinates were naively treated as individual feature rankings. Our [synthetic shortcut experiment](../04_shortcut_sandbox/README.md) showed that an apparently foreground-only attribution can coexist with background-dependent prediction when the reference leaves the background unchanged.

The present experiment examines an earlier stage of that process: the information entering the model and the stability of its spatial arrangement. Feature tracing now works with the authors' SOM layouts. The next study will test explanations from these models using the principles in [Grad-CAM](https://arxiv.org/abs/1610.02391), [Integrated Gradients](https://arxiv.org/abs/1703.01365) and [saliency sanity checks](https://arxiv.org/abs/1810.03292). Improved attribution or biological interpretation remains an open question.

## Reproduce and verify

Use the existing Python 3.12 learning environment and [pinned dependencies](../../requirements-learning.txt). Only MiniSom 2.3.5, PyYAML 6.0.2 and pinned upstream-test dependencies were added; previous scientific-library versions were not upgraded. This differs from the upstream Windows/CUDA version lock, so cross-environment identity is not claimed.

If the ignored checkout is absent, obtain the exact commit:

```sh
git clone https://github.com/Kerbash/tab2img-bench.git .cache/tab2img-bench
git -C .cache/tab2img-bench checkout --detach dc0ef691db733ebcba4d13b53ca58e258f76ef0a
.venv/bin/python -m pip install -r requirements-learning.txt
PYTHONPATH=.cache/tab2img-bench/src MPLCONFIGDIR=/private/tmp/css700-matplotlib .venv/bin/python -m pytest .cache/tab2img-bench/tests -q
```

The verified Wisconsin raw files must already exist; see [Experiment 01](../01_wisconsin/README.md). Choose fresh output/checkpoint directories; the runner refuses existing result/checkpoint targets:

```sh
MPLCONFIGDIR=/private/tmp/css700-matplotlib XDG_CACHE_HOME=/private/tmp/css700-cache .venv/bin/python -m scripts.som2img_learning --output-dir outputs/som2img-repeat --checkpoint-dir artifacts/som2img-repeat
cmp experiments/07_som2img_layout/results.json outputs/som2img-repeat/results.json
MPLCONFIGDIR=/private/tmp/css700-matplotlib .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m ruff check scripts/som2img_learning.py tests/test_som2img_learning.py
```

All **94 project tests** and **18 upstream tests** passed, along with lint checks for the new code and `pip check`. A separate full execution reproduced the JSON and four PNGs byte-for-byte. Reloading all 21 saved checkpoints reproduced the stored evaluation probabilities. The figures were also checked visually. Checkpoints, raw data and the unchanged upstream checkout remain outside version control; the repository contains the experiment code, compact evidence and report.

## Next experiment

The next experiment will use the existing models without retraining. It will specify common evaluation cases, target scores and a training-median reference, then compare input-level attributions after decoding them to feature names. Reference sensitivity and matched feature replacement will test whether the resulting explanations reflect model reliance. All SOM seeds will remain in the analysis.

[repo]: https://github.com/Kerbash/tab2img-bench/blob/dc0ef691db733ebcba4d13b53ca58e258f76ef0a/README.md
[som]: https://github.com/Kerbash/tab2img-bench/blob/dc0ef691db733ebcba4d13b53ca58e258f76ef0a/src/tab2img/methods/som.py
[random]: https://github.com/Kerbash/tab2img-bench/blob/dc0ef691db733ebcba4d13b53ca58e258f76ef0a/src/tab2img/methods/random_layout.py
[evaluation]: https://github.com/Kerbash/tab2img-bench/blob/dc0ef691db733ebcba4d13b53ca58e258f76ef0a/src/tab2img/evaluation.py
[encoding]: https://github.com/Kerbash/tab2img-bench/blob/dc0ef691db733ebcba4d13b53ca58e258f76ef0a/src/tab2img/pipeline.py
[dictionary]: https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names
