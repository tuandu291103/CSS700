# Spambase Grad-CAM v2: inference-preserving activation boundary

Draft begun October 3; prospective freeze: **2026-10-04**, before executing the real-data backend diagnostic or
v2 candidate. V1 remains held and immutable. This is a numerical execution amendment,
not a new model, attribution target/layer/sign, tolerance sweep or test-set opening.

## Fixed question and candidate

V1 replayed no-gradient inference exactly, but gradient-enabled convolution outputs
differed and 74/691 validation rows failed `rtol=1e-5, atol=1e-6` at the logits. We will
test whether the NNPACK-enabled CPU inference path explains that discrepancy, and whether
starting autograd at the target activation preserves the original inference decisions.

The **sole repair candidate**, chosen before this run, evaluates `features[:3]` under
`torch.no_grad()` with NNPACK enabled, retains its pre-ReLU second-convolution activation
`A`, and makes `A.detach().requires_grad_(True)` the differentiation leaf. It applies the
unchanged `features[3]` ReLU, global average pooling and linear classifier under grad mode.
Compute the selected frozen class-logit's gradient with respect to A, spatially average
each channel, form the weighted activation sum and ReLU. Same 8×8 raw Grad-CAM formula.

This boundary is specific to the validated sequential SmallTabularImageCNN: all paths
from A to its logits are in that head. It is **not** an input-gradient method and must
not replace Integrated Gradients or skip upstream dependencies in other architectures.
Autograd need not differentiate the convolutions to obtain the logit's derivative with
respect to their output. PyTorch permits ordinary no-grad outputs to be used in later
grad-mode computations; do not substitute `inference_mode`. See
[autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd) and
[NNPACK controls](https://docs.pytorch.org/docs/stable/backends.html#torch.backends.nnpack).
The installed Torch version/build is recorded; documentation is not evidence of a local
backend cause, which must be established by the controlled calculations below.

## Preserved scope and controlled calculations

Pin the accepted training review and all its upstream identities. Additionally pin the
negative v1 review, all eight immutable v1 files and their upstream input hashes. Check
that v1 uses the same checkpoint review, IDs, targets and feature map. No raw-data parsing,
scaler fitting, new encoding, training or test inference. Hash-only source reads remain
provenance checks. Use all 691 saved validation rows, ascending source order, batch 32,
CPU, one thread, deterministic algorithms. No random selection or warm-up.

For every batch, run these **four diagnostic forward passes**, always in this order:

1. NNPACK enabled, no gradients (must replay original inference exactly).
2. NNPACK enabled, gradients enabled (must replay the held v1 logits exactly).
3. NNPACK disabled, no gradients.
4. NNPACK disabled, gradients enabled.

Retain each mode's logits and summarize pairwise maximum differences/failing values
using v1's unchanged tolerances. Record per-layer maximum differences for the two
NNPACK-enabled modes. The disabled-backend paths are diagnostics only, never alternative
checkpoints or selectable replacement predictions. Their identity patterns may support
a backend explanation; do not declare that cause if the observed pattern does not.

Then execute the single activation-boundary candidate with NNPACK enabled. One candidate
stem/head forward plus one gradient calculation per batch; no full-network backward or
optimizer. Require its logits to be **bitwise exact** to original inference, a stronger
check than v1's gradient tolerance. Also retain the original tolerance check and exact
class/target identity. No enlarged tolerance, float64 model substitution or backend
selection based on accuracy, labels or feature ranks. If exactness fails, retain another
held result rather than choosing a different candidate or retrying.

Bound the combined computation to 120 seconds, checked between batches and at completion.
Restore thread, determinism/warn mode, CPU RNG and NNPACK state on success and exceptions.
No hooks are added. Verify unchanged weights and absent parameter gradients.

## Maps, gates and independent review

Keep v1's raw/display normalization, 57-feature mapping, padding exclusion, positive-only
top lists/tie ordering, row and group-equal aggregates, and fixed examples (00,11,01,10).
Report zero maps and padding mass without excluding difficult rows. Save per-row inputs'
IDs/targets/labels/weights, all raw/display maps and traced scores; save diagnostic logits
separately. Compare v2 maps and positive top-five membership with held v1 descriptively;
these differences are not a selection criterion or explanation-quality metric.

Automatic gates: clean prospective commit, all pins and validation identities verified,
exact inference/v1-mode replay, exact candidate inference logits, v1 tolerance unchanged,
exact predicted targets, finite nonnegative maps, bounded display maps, input-sized maps,
all features traced exactly, reversed-map padding probe, unchanged inputs/weights and no
parameter gradients/hooks. Backend-disabled equality is descriptive, not an acceptance gate.

Require a separate audit without project model/Grad-CAM imports: rebuild the original
CNN, recompute the four diagnostic modes and candidate, verify exact saved logits, and
calculate a float64 reference from **inference-path A** using
`W[target,k] * 1[A>0] / 64`. Verify raw maps at the unchanged `rtol=1e-5, atol=1e-7`, plus
normalization, every feature/record/rank, padding diagnostics, aggregate weightings and
example IDs. This tests numerical identity and the formula, **not faithfulness**.

Commit code/config/tests before execution. Save one new append-only v2 run, including
failures, then separately pin and review it. Never edit/relabel v1 as accepted. A passing
v2 can be accepted only for validation tracing and proceeding to a prospective validation
faithfulness/input-level comparison. All **690 test rows remain closed**, and the full
matrix gains no completed condition. Cross-version/backend timing is not an efficiency
claim. Future explanation methods must independently handle numerical consistency.
