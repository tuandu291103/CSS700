# Experiment 02: learn the models before explaining them

23 October 2026 · Exploratory learning experiment · Not a SOM2IMG reproduction

## Brief result

The small fixed-grid CNN learned a useful classifier on this split, with accuracy close to logistic regression. It did not establish better interpretability, a biological discovery, or a general ranking of model families. Our next question is what evidence each model used to reach its predictions.

All models use the same 397 training, 86 validation, and 86 learning-test records from [Experiment 01](../01_wisconsin/README.md). Values below are computed results, not numbers copied from a paper. See the [complete results and saved predictions](results.json) and the [settings recorded before the first training run](protocol.json). This local protocol is not an external preregistration.

### Learning-test results

| Model | Runs | Accuracy | Balanced accuracy | Macro-F1 | ROC-AUC | Log loss ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Majority/prior control | 1 | 62.79% | 50.00% | 0.3857 | 0.5000 | 0.6601 |
| Logistic regression | 1 | 96.51% | 95.31% | 0.9619 | 0.9792 | 0.1561 |
| Random forest | 3 | 92.25% | 91.49% | 0.9167 | 0.9826 | 0.1659 |
| Fixed-grid CNN | 3 | 96.12% | 95.00% | 0.9578 | 0.9632 | 0.2480 |

Three-run entries are arithmetic means, not an ensemble. Logistic regression and the prior control have one fixed run. Forest/CNN seeds are 20261023, 20261024, 20261025; the data split never changes. CNN accuracy ranges from 95.35% to 96.51% (sample SD 0.67 percentage points); forest accuracy ranges from 91.86% to 93.02% (SD 0.67 points). Three shared-split seeds do not provide a confidence interval for population performance.

Individual correct counts: logistic regression **83/86**; CNN **83, 83, 82/86**; forest **80, 79, 79/86**. All three CNN runs missed three malignant records; the last run also classified one benign record as malignant. Equal confusion matrices do not necessarily mean equal individual predictions. [All eight confusion matrices](confusion_matrices.png)

Metric meanings: accuracy is overall correctness; balanced accuracy averages the two class recalls; macro-F1 averages classwise F1; ROC-AUC evaluates score ranking across thresholds; log loss evaluates probability assignments, with lower values better. These measure **prediction**, not explanation quality. Definitions: [scikit-learn evaluation guide](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics).

## What we learned

1. **Similar accuracy can hide different behavior.** Logistic regression and the CNN have similar hard-label performance, but CNN mean log loss is worse (0.2480 versus 0.1561). This is a difference in probability quality, not a completed calibration analysis or an attribution result.
2. **One metric is not a universal ranking.** Forest has the highest mean ROC-AUC here despite lower accuracy at the fixed threshold. We did not optimize a threshold after inspecting test labels.
3. **Checkpoint selection matters.** CNN seeds selected epochs 199, 197, and 142. The first two reached the 200-epoch cap; convergence is not established. The third stopped at 172 after 30 epochs without a better validation loss, then restored epoch 142. At that selected epoch train/validation losses were 0.0741/0.0605; at the last epoch they were 0.0709/0.0841. This is a concrete example of training loss improving while validation loss worsens. [Learning curves](learning_curves.png)
4. **A gap needs investigation, not a diagnosis by itself.** Forest training macro-F1 is 0.9946 while mean test macro-F1 is 0.9167. All three families score well on this small validation set. Training/test gaps can reflect fitting and split difficulty; this exercise cannot isolate the cause. No automatic retuning was performed.
5. **We still do not know which features caused a particular model decision.** Predictive success and visually organized inputs do not establish faithful explanations. Fixed-grid neighborhood positions are arbitrary. Interpretation and shortcut tests remain separate experiments.

## How our CNN works

A convolution applies learned weights to local neighborhoods, reusing those weights across positions. ReLU introduces a nonlinear transformation. Here two 3×3 convolutions give each second-layer spatial unit a nominal 5×5 input receptive field, truncated by boundaries. With only a 5×6 input, spatial responses already combine many original features. This matters when we later trace an activation or explanation back to a feature. [Stanford CS231n convolution notes](https://cs231n.github.io/convolutional-networks/)

The following dimensions and parameter counts come from our actual implementation; `N` is the batch size:

| Stage | Output shape | Learned parameters | Role |
|---|---|---:|---|
| Scaled feature grid | N×1×5×6 | 0 | One channel, 30 measurements |
| 3×3 convolution + ReLU | N×8×5×6 | 80 | Eight local response maps |
| 3×3 convolution + ReLU | N×16×5×6 | 1,168 | Sixteen combined response maps |
| Global average pooling | N×16×1×1 | 0 | Average each map over spatial positions |
| Flatten + linear layer | N×2 | 34 | Produce benign/malignant scores (logits) |
| Softmax at prediction time | N×2 | 0 | Convert logits to class probabilities |

Total: **1,282 trainable parameters**. Padding preserves spatial dimensions; it is not a measured feature. There is no resize, spatial max-pooling, augmentation, dropout, or batch normalization. Global pooling is a small-model teaching choice, not a claim of optimal architecture. Layer/module and logit concepts: [PyTorch model tutorial](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html).

During training, cross-entropy compares the logits with the known class. Backpropagation computes parameter gradients; Adam updates the weights. A lower loss rewards better predictions, but the objective does not explicitly require a scientifically meaningful reason. An activation map is an internal response, not automatically a feature-importance map.

**Concrete forward-pass check:** using the first listed CNN seed and the same first training record as Experiment 01 (ID 842302, label M), the selected model outputs logits approximately `[-5.9268, 4.2315]`, then probabilities `[0.0000388, 0.9999613]`. This is a training example for tracing computation, not evidence of generalization or calibrated clinical confidence. Reloading the saved checkpoint reproduces its outputs.

Questions to answer before continuing: What changes when weights are trained? Why does a 5×6 activation map not mean that each activation belongs to exactly one original feature? Why can a confident prediction be wrong? How would we test the reason rather than just inspect a picture?

## Fixed protocol and controls

- Reuse the exact Experiment 01 audit (SHA-256 checked), raw-file hashes, row identities, split membership, and feature map. Fit scaling only on training rows and verify its extrema against the original audit. No feature selection or learned layout.
- Provide identical float32 values to all models. Only the CNN reshapes them to N×1×5×6. The float64-to-float32 cast changes scaled values by at most 1.12e-7; this is distinct from the exact float64 mapping round trip in Experiment 01. No clipping or 8-bit image export is used as model input.
- Logistic regression: C=1, default L2 regularization, lbfgs, maximum 2,000 iterations; convergence-limit check. Forest: 300 trees, sqrt feature subsampling, minimum leaf size 2, unlimited maximum depth. These are teaching settings, not a competitive tuning search. [Logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [random forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
- CNN: Adam, learning rate 0.003, weight decay 0.0001, batches of 32, at most 200 epochs, patience 30, minimum validation cross-entropy checkpoint. Its training function receives no test-data argument. All models/checkpoints are fixed before test scoring.
- Fixed decision rule: M only when P(M)>0.5; ties go to B. No class weighting, calibration fitting, hyperparameter search, best-seed selection, or post-test retuning. CNN checkpoint selection consumes validation feedback; baseline settings are fixed, so this is not a matched-tuning-budget study.
- CPU, one PyTorch thread, deterministic algorithms and fixed seeds. A full independent rerun produced byte-identical results on this machine. PyTorch does not guarantee identical behavior across platforms/releases. [Reproducibility guidance](https://docs.pytorch.org/docs/2.14/notes/randomness.html)

## Reproduce and verify

Use Python 3.12 and the repository environment. Raw Wisconsin files must already be present; see [Experiment 01 acquisition commands](../01_wisconsin/README.md#reproduce). From the repository root:

```bash
.venv/bin/python -m pip install -r requirements-learning.txt
.venv/bin/python -m scripts.wisconsin_models --output-dir outputs/02_wisconsin_reproduction --artifact-dir artifacts/02_wisconsin_reproduction
cmp experiments/02_wisconsin_models/results.json outputs/02_wisconsin_reproduction/results.json
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/ruff check scripts/wisconsin_models.py tests/test_wisconsin_models.py
```

Use a fresh output folder for another rerun; existing results are not overwritten. Exact JSON comparison expects the recorded code and environment. Model checkpoints live in ignored `artifacts/02_wisconsin_models/`; small results, protocol, and figures are tracked. PyTorch checkpoints contain weights, map, scaler parameters, provenance, and protocol. Baseline bundles include the model and scaler. Only load pickle/joblib checkpoints from a trusted source.

Validation performed: all 33 tests and new-code lint passed; complete rerun equality, recomputation of all 16 saved validation/test metric sets from stored probabilities, exact checkpoint-reload prediction checks for all three CNNs, and visual inspection of both figures passed. Fixed the previous dependency snapshot's accidentally captured warning line; requirements parsing, offline installation dry-run, and dependency consistency now pass.

## Limits and next milestone

This is one morphology dataset and one reused learning split, with no patient-independence guarantee, external clinical validation, matched tuning budget, SOM2IMG reproduction, or explanation-quality measurement. The results do not prove that CNNs always beat or lose to tabular methods. Earlier genomic work is preserved.

Next, freeze these models and study explanations on a small deterministic set of validation examples, beginning with the first listed CNN seed rather than choosing a seed by its test score. Learn Grad-CAM and input-level attribution, map outputs back to named features, then introduce perturbation and model-randomization checks. Do not call a heatmap a thesis finding before these checks. The second dataset and learned-layout comparison remain later milestones.
