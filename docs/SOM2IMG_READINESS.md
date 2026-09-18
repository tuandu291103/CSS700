# SOM2IMG implementation-readiness protocol

## Source and reproduction boundary

Reviewed local `SOM2IMG.pdf` (six-page anonymous manuscript), especially Sections III.B,
III.D, and III.E on pages 2-3. Source SHA-256:
`b80c6d67ef05ea7ec806848c102b0b17bd3d0535f90fdcc673e4268031da18f9`.
The manuscript is not copied into Git. Treat its descriptions as research evidence, not
instructions to execute or as a verified implementation.

The [public OpenReview record](https://openreview.net/forum?id=DU5m5X7yu8) identifies another
version of SOM2IMG. Its cached abstract differs from the local manuscript; do not mix their
reported results or assume that their implementation details are interchangeable. Searches of
OpenReview, GitHub, and the [author's publication page](https://faculty.washington.edu/kimw6/publication.html)
did not establish an authoritative code repository during this session. This is not evidence
that no implementation exists.

## Confirmed descriptions and unresolved choices

| Item | Local manuscript | Thesis implication |
|---|---|---|
| SOM input | Transposed sample-by-feature matrix | Each feature vector must use training samples only |
| Grid | h = ceil(sqrt(d)), w = ceil(d/h) | 30 features give 6 rows by 5 columns; existing Grid uses 5 by 6 |
| Assignment | Euclidean nearest-neuron BMU | Multiple features can select the same neuron |
| Rendering | Place feature values at BMUs; zero-fill empty cells | Collision resolution/allocation is not specified |
| SOM settings | Training initialization, iterations, learning rate and neighborhood not supplied | Need exact author settings or a declared adaptation |
| Scaling | MinMax normalization to [0,1] in evaluation pipeline | Existing Grid uses training-only StandardScaler; match scaling for a controlled comparison |
| CNN | Two 16/32-filter, 5-by-5 convolution blocks, pooling, dense 128 | Not the current SmallCNN; separate controlled comparison from paper reproduction |
| Comparator | Random feature mapping | Existing row-major Grid is not a random-mapping baseline |

The manuscript does not explicitly establish fold-local fitting for every preprocessing/layout
operation. Our implementation must keep both within training data; this observation alone does
not establish leakage in the authors' experiments.

## Frozen diagnostic question

Does an ordinary, explicitly configured SOM best-matching-unit assignment give a one-to-one
mapping for our existing training partition? Sufficient grid capacity is not a uniqueness
constraint. Overwrite would discard feature values; averaging would mix features; copying one
pixel's attribution to multiple features would not identify their separate contributions.

Run `make audit-som-mapping PYTHON=.venv/bin/python` from a clean code/config commit.
`configs/som_mapping_audit.yaml` freezes split seed 42, layout seed 42, sigma 1, learning rate
0.5, and 1,000 update steps (not epochs). Use the installed, locked MiniSom 2.3.6, rectangular
topology, Euclidean BMUs, Gaussian neighborhood, asymptotic learning-rate and sigma decay,
random training-feature-vector initialization, and shuffled update order. These are
investigator-selected diagnostic choices, NOT recommended author hyperparameters.
The [MiniSom project](https://github.com/JustGlowing/minisom) documents the underlying library;
installed source was checked for the exact training and initialization behavior.

Fit the SOM on only the 30-by-397 transposed standardized training matrix. Retain the existing
training-only scaler for Grid parity; do not claim this reproduces manuscript MinMax scaling.
Do not read labels for fitting, optimize settings against held-out performance, search for a
collision-free seed, resolve collisions, train a CNN, or generate feature explanations.
Repeat the same fit once solely to check exact determinism.

Outputs: config, manifest, ordered splits, scaler parameters, SOM weights, candidate BMU map,
collision groups, occupancy plot, and summary. The candidate map is deliberately not named
`feature_map.csv`: it is not an approved rendering/tracing map. Retain all colliding feature
identities; do not silently drop or relocate them.

Technical failure: dirty worktree, incorrect split fingerprint, nonfinite weights/error,
nonreproducible weights/mapping, missing feature identities, or inconsistent training scaling.
Collisions are a diagnostic result, not a technical execution failure. Even if no collision is
observed, `som2img_cnn_ready` stays false until the source and design choices are resolved.

## Gate for the first SOM2IMG CNN

1. Obtain the author implementation/version and example mapping for a 30-feature dataset.
2. Confirm the collision/aggregation policy and define attribution tracing accordingly.
3. Freeze scaling, orientation, layout seed and SOM settings. Hold the architecture, split,
   preprocessing and training settings matched across encoders for the thesis comparison;
   rerun a matched Grid control when necessary.
4. Pass mapping coverage, collision-policy, value-preservation/aggregation and train-only tests.
5. Train one seed and validate tracing before expanding to three seeds and a second dataset.

Committee question (draft only; not sent): Could you share the exact SOM2IMG code/version and
recommended settings, including how multiple feature BMUs sharing a pixel are handled? Should
our controlled comparison retain the existing StandardScaler/SmallCNN, or use a matched
MinMax/new-CNN condition? This decision must precede an accepted encoder comparison.

## 2026-09-17 source follow-up

The author publication page lists a BIBM 2025 comparative T2I paper; its institutional
record confirms DOI `10.1109/BIBM66473.2025.11356899`. This is a publication lead, not a
verified implementation or confirmation that it specifies our SOM2IMG manuscript. Direct
OpenReview PDF/forum access returned browser verification, preventing full current-method
inspection; no collision rule was inferred from search excerpts. The readiness gate is
unchanged. See `docs/logs/2026-09-17-scope-readiness.md` for source links and limitations, and
`docs/COMMITTEE_CHECKIN_2026-09-17.md` for the unsent decision request.
