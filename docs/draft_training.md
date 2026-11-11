# Canonical draft objectives

`algorithms/objectives` owns training computation. `build_objective_model` wraps a
canonical draft and an optional frozen target head in one module, so distributed
training executes the entire differentiable computation through its wrapper.
Objectives register independently from model construction and checkpoint formats.
A registration declares supported architectures, input/output names and parameter
prefixes it trains. A trainable parameter outside that coverage fails construction;
explicitly frozen parameters and target bindings are excluded.

The default objectives cover EAGLE3, P-EAGLE, DFlash, DFlash2, DSpark, Domino,
DSpine, MTP and DMTP. They reuse explicit-label CE/L1 loss primitives and functional
sampling/mask construction. EAGLE3 replays its recurrent tree with explicit dense
visibility; it does not add algorithm hooks or mutable KV caches to the backbone.
This is an eager reference path, including dense vocabulary projections; fused
losses, sparse attention and sequence parallel execution are not implemented here.
The legacy training path retains those optimizations and its optional loss variants.

Input batches use **unshifted** `input_ids`, `loss_mask` and teacher features at the
same original sequence positions. Teacher tensors are detached at the boundary.
Auxiliary features are `hidden_states`; final normalized target states are
`target_last_hidden_states`, or callers can supply preprojected `target_logits`.
DMTP additionally requires `mtp_hidden_states` from a frozen MTP provider: state at
anchor A fuses with token A+1, and the objective supervises A+2 onward. Objectives
never run the target or frozen MTP themselves. P-EAGLE additionally needs an
`PEagleSpec.mask_token_id` configuration and currently requires batch size one.

`ObjectiveResult` exposes an additive numerator, denominator and metric ratios.
The trainer must pool these across microbatches and data-parallel ranks. DSpine's
alignment multiplier uses the global backbone CE for the optimizer window; with
accumulation, a graph-free prepass followed by RNG-preserving replay supplies that
multiplier. Its transfer space must be initialized or restored before the first
forward. Domino and DSpine schedules require the resolved total optimizer steps.
