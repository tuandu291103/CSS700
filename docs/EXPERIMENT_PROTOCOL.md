# Experiment and artifact protocol

## Run identifier

Use an append-only identifier:

```text
YYYYMMDD-HHMM__dataset__encoding__model__seed-N__short-purpose
```

Each run writes to `artifacts/<run-id>/` and must fail rather than overwrite an existing run.

## Required run bundle

```text
config.yaml                 exact resolved configuration
manifest.json               run ID, timestamps, device, commit SHA, package versions
splits/                     sample IDs or deterministic indices
preprocessing/              fitted training-only preprocessing metadata
encoding/feature_map.csv    original-feature-to-pixel mapping
encoding/padding_mask.npy   non-feature pixel mask
model/                      checkpoint reference and predictive metrics
attributions/               raw and traced attribution files
metrics/                    faithfulness, agreement, stability, and runtime tables
figures/                    generated diagnostic plots
notes.md                    interpretation, anomalies, and limitations
```

Large outputs stay outside Git. A final result table must be reproducible from a tracked config,
code commit, immutable source dataset version, and referenced run bundle.

## Before running

- State the question and comparison.
- Freeze the eligible samples and metric definitions.
- Confirm no preprocessing or layout is fit outside training data.
- Record expected outputs and a failure criterion.
- Commit code and configuration.

## After running

- Check for NaNs, degenerate predictions, target mismatch, and mapping errors.
- Verify Integrated Gradients convergence delta and attribution completeness diagnostics.
- Verify occlusion masks exactly the intended feature locations.
- Compare top-feature deletion with random and bottom-feature controls.
- Complete an experiment log and link every claim to a table or figure.
- Update the thesis outline immediately; do not postpone all writing until Spring.
