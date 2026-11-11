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
The legacy training path retains the optimized kernels. MTP/DMTP CE is evaluated
in FP32 for stable log-softmax; it need not match the legacy BF16 reduction bitwise.

Loss and position weighting can be selected independently of the architecture:

```python
runtime = build_objective_model(
    draft, loss={"kind": "kl"}, weighting={"kind": "exponential", "gamma": 7.0}
)
result = runtime(batch, step=optimizer_step, total_steps=total_optimizer_steps)
```

`loss.kind` accepts `default`, `ce` (explicit labels), `soft_ce`, `kl`, `reverse_kl`,
`js`, `tv`, `nla` and `lk`. Distillation needs aligned teacher logits or final states
plus a frozen head. `default` preserves each algorithm's base loss. Selector,
confidence, refinement and alignment supervision remain owned by the algorithm.
Weighting accepts `uniform` or `exponential`; omitting it preserves the objective's
`loss_decay_gamma`. EAGLE/P-EAGLE apply it to rollout depth. DFlash2 selector warmup
and ramp accept either absolute steps or ratios of total optimizer steps.

Input batches use **unshifted** `input_ids`, `loss_mask` and teacher features at the
same original sequence positions. Teacher tensors are detached at the boundary.
Auxiliary features are `hidden_states`; final normalized target states are
`target_last_hidden_states`, or callers can supply preprojected `target_logits`.
DMTP additionally requires `mtp_hidden_states` from a frozen MTP provider: state at
anchor A fuses with token A+1, and the objective supervises A+2 onward. Objectives
never run the target or frozen MTP themselves. P-EAGLE additionally needs an
`PEagleSpec.mask_token_id` configuration and currently requires batch size one.

`ObjectiveResult` exposes an additive numerator, denominator and metric ratios.
EAGLE/P-EAGLE return each microbatch mean with denominator one. DSpark likewise
returns a microbatch mean, using a global token denominator and DDP gradient
averaging during training. Other objectives return token-weighted sums. Pooling
these pairs preserves each method's existing accumulation semantics. Evaluation
performs no objective-level collectives, so ranks may have unequal batch counts.
All objectives expose accuracy under the `acc` ratio key; EAGLE uses rollout depth
zero and compares mapped draft predictions against the full target vocabulary.
DSpine's
alignment multiplier uses the global backbone CE for the optimizer window; with
accumulation, a graph-free prepass followed by RNG-preserving replay supplies that
multiplier. Its transfer space must be initialized or restored before the first
forward. Callers pass `step` and `total_steps` explicitly (`None` when no total-step
schedule is used). EAGLE and block objectives reject packed sequences. P-EAGLE
supports packed documents and excludes shifted labels crossing document ends.
