# MoE FFN Design (`specforge.modeling.draft.moe`)

Design note for the sparse-MoE FFN that a draft model can opt into. The
training plane's picture is in
[`../../../training/DESIGN.md`](../../../training/DESIGN.md).

## Responsibility

Owns the FFN of a decoder layer when the draft JSON names a `moe_preset`, and
nothing else: attention, heads, losses and the trainer loop are unchanged.
A draft uses it in one of two ways:

- **DFlash-family drafters** (DFlash, DFlash2, DSpark, Domino) build it through
  `build_ffn`. They all inherit the Qwen3-based `DFlashDraftModel`, so the FFN
  is the only part a preset changes; a new target family adds one module under
  `families/` and no model code.
- **Drafters with their own block**, such as a target's official drafter, build
  `MoELayer` directly. Changing `DFlashDraftModel` for another block would
  change all four DFlash-family drafters and their checkpoints, so such a
  drafter is its own class; it still reuses this package's layer, hooks and
  checkpoint layouts.

The package is **one configurable layer**, not one block per target family:
one `MoELayer` whose router, balancing and experts are picked by config (the
Megatron-Core split), chosen over the transformers pattern of a copied
`XxxSparseMoeBlock` per model because the family differences are a handful
of small functions while the expensive parts are shared:

| differs per target family | shared by every family |
| --- | --- |
| score function (softmax, sigmoid, sqrtsoftplus) | token dispatch (sorted segments, grouped GEMM) |
| balancing (aux loss, aux-loss-free bias, none) | deferred update timing vs activation checkpointing |
| combine-weight renorm and scale | FSDP-friendly stacked expert layout |
| shared-expert gate (none, sigmoid), SwiGLU clamp | fp32 accumulation of routed and shared output |
| checkpoint file naming, serving config | load metrics, router centering and its export fold |

## Why match the target's MoE

The drafter's MoE should be the *target's* MoE, expressed as a preset:

- **Warm start.** Same expert shape means the draft experts can be seeded from
  a subset of the target's (`select_target_experts`), which a dense drafter
  cannot do.
- **Serving.** The drafter runs inside SGLang's draft model; matching the
  target's routing reuses its fused MoE kernels and weight naming.
- **Latency.** At small batch, top-k of narrow experts reads about the same
  bytes as the dense MLP, so MoE buys parameters at roughly constant step
  cost. That is the hypothesis the dense-vs-MoE ablation tests.

## Layout

```
base.py        Registry, RoutingResult, TrainingState, apply_keeping_fp32
config.py      MoEConfig + preset registry; resolves a draft JSON
router.py      score functions, Router ABC, TopKRouter (router bias and
               centering), group-limited mask, the centering export fold
balance.py     BalanceController: none, noaux_tc, aux_loss
experts.py     RoutedExperts ABC, GroupedExperts (stacked [E,out,in]), swiglu
shared.py      SharedExpert ABC, SwiGLUSharedExpert
layer.py       MoELayer, build_ffn, and the model-level hooks
checkpoint.py  checkpoint layouts, serving config, select_target_experts
families/      one module per target family: preset and layout
```

Every module except `families/` is family-agnostic.

## Rules

1. **Core never imports families.** Only `__init__.py` imports `families`, at
   the bottom, to trigger registration.
2. **Every choice is explicit.** The preset gives a default for each
   architecture field, including the checkpoint layout; the draft JSON may
   override any of them (`moe_layout` for the layout). Nothing is guessed from
   which keys a file happens to contain.
3. **A new family only adds files.** Its module under `families/` registers a
   preset and a layout; `families/__init__.py` gets one import line. A new
   routing mechanism or expert implementation is a family-agnostic component:
   a score function, router, balance controller, experts backend or shared
   expert registered in its core module.

## Contracts that matter

**Detection keys on the preset.** A draft is MoE iff it names `moe_preset`.
Configs copied from an MoE target (an MTP draft, say) carry
`n_routed_experts` without asking this package for anything, so the count
alone does not switch a layer to MoE; but a DFlash-family layer that sees
`n_routed_experts` without a preset refuses to build, and expert counts must
be positive integers. Neither case falls back to a dense FFN silently.

**One training-state loop.** Router, balance controller, routed experts and
shared expert all implement `TrainingState` (`observe`,
`apply_pending_update`, `aux_loss`, `metrics`, `reset_parameters`), with
no-op defaults. `MoELayer` drives them in a fixed order
(`gate, gate.balance, experts, shared_experts`). The balance controller stays
a child of the router, so its buffers keep the `gate.balance.*` keys.

**Updates are deferred.** `observe` only stashes (overwrite, never
accumulate) so an activation-checkpoint recompute leaves identical state.
`apply_pending_update` runs from the *model* forward before any routing,
outside checkpoint regions, and may run collectives (the noaux bias and the
router's centering mean move there). Mutating selection state inside a layer
forward would make the recompute route differently and raise
`CheckpointError`.

**Official accumulation order.** `RoutedExperts.forward` returns the weighted
routed sum as a fresh fp32 tensor; `MoELayer` adds the shared expert into it
and casts to the input dtype once, as DeepSeek's reference `MoE.forward`
does. Casting the routed sum to bf16 before adding the shared expert differs
by ~3e-3 relative, enough to flip top-k choices downstream; the fp32 add costs
under 2% of the layer's fwd+bwd on H200.

**Attribute names are the module-side contract.** `MoELayer` exposes `gate`
(with its balance controller at `gate.balance`), `experts` and
`shared_experts`, the DeepSeek-family names SGLang loads; a family's layout
maps them to its file naming.

**Naming is converted at the boundary, not in `state_dict()`.** FSDP's
full-state-dict hooks index the gathered dict by the module's own parameter
FQNs. Every file read/write calls `to_checkpoint_state_dict` /
`from_checkpoint_state_dict` with the draft model (any module tree) or its
config: `FSDPTrainingBackend` save/load, resume, warm start,
`AutoDraftModel.from_pretrained`, `materialize_draft` and the HF exporter.
Layouts act on one MoE layer at a time (the layer prefix is the text before
its `experts` component) and only the layout the config names runs.

**Router centering.** From-scratch DFlash-family drafters feed every router a
token-independent common mode (mask-token embeddings plus near-uniform
attention over the target features), so every token picks the same top-k and
an aux loss alone does not recover. `moe_router_center: "sample"` centers the
router input on each sample's token mean in training and on a running mean in
eval. Exports fold the running mean into a fixed fp32 router logit bias
(`gate.bias = -W @ mu`) and set `moe_router_bias` in `config.json`, so an
export reloads as the draft it describes and a serving engine that reads a
router bias reproduces the trained routing. `exported_config_updates` adds
the `config.json` fields the layout's serving engine reads, and
`serving_architecture` maps the class a dense draft is served with to the
engine's class for the same draft with the family's MoE FFN (the export
normalizer uses it).

**Routing state stays fp32 and exact.** The selection bias, the centering
mean and the router bias are fp32 buffers that no optimizer owns. Trainers
cast the draft to bf16 after warm-starting it, so the components' `_apply`
keeps those values exact (`apply_keeping_fp32`) instead of rounding them and
restoring the dtype: a warm-started DeepSeek selection bias differs across
experts by less than bf16's step at its magnitude.

**Model-level terms ride every strategy.** `DraftTrainStrategy.finalize_step_output`
(called by `TrainerCore` after `forward_loss`) adds the layers' summed
auxiliary loss to whichever term backpropagates (`loss`, or the `loss_terms`
numerator scaled by the micro-batch denominator) and the layer-averaged
`moe/*` metrics to `StepOutput.metrics`: `load_max_ratio`, `load_min_ratio`,
`experts_unused_frac`, plus the balance controller's (`aux_loss`,
`load_frac_*`, or `bias_*` and `global_load_*`). Strategies hold no MoE code;
dense drafts log no `moe/*` keys; evaluation calls `forward_loss` alone, so
eval losses exclude the auxiliary term.
The trainer DP-averages the `moe/*` metrics and logs them like any other
scalar.

## Extension points

- New target family: a module under `families/` with `register_moe_preset`
  (recipe + default layout) and `register_layout` (file naming, serving
  config, serving class).
- New routing mechanism: `register_score_function`, `register_router` or
  `register_balance_controller`.
- New expert implementation: a `RoutedExperts` subclass under
  `register_experts_backend` (or a new `MoEConfig.dispatch` value inside an
  existing backend), or a `SharedExpert` subclass under
  `register_shared_expert`.
- Ablations: any architecture key at the draft JSON top level overrides its
  preset default; `dflash_config.moe_*` sets training knobs.
