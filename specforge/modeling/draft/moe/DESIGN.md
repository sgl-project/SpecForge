# MoE FFN Design (`specforge.modeling.draft.moe`)

Design note for the sparse-MoE FFN that a draft model can opt into. The
training plane's picture is in
[`../../../training/DESIGN.md`](../../../training/DESIGN.md).

## Responsibility

Owns the FFN of a decoder layer when the draft JSON names a `moe_preset`, and
nothing else: attention, heads, losses and the trainer loop are unchanged.
DFlash-family drafters (DFlash, DFlash2, DSpark, Domino) build it through
`build_ffn`; they all inherit the Qwen3-based `DFlashDraftModel`, so the FFN
is the only part a preset changes, and a new target family adds one module
under `families/` and no model code.

The package is **one configurable layer**, not one block per target family:
one `MoELayer` whose router, balancing and experts are picked by config (the
Megatron-Core split), chosen over the transformers pattern of a copied
`XxxSparseMoeBlock` per model because the family differences are a handful
of small functions while the expensive parts are shared:

| differs per target family | shared by every family |
| --- | --- |
| score function | token dispatch (sorted segments, grouped GEMM) |
| balancing (aux loss, none) | deferred update timing vs activation checkpointing |
| combine-weight renorm and scale | FSDP-friendly stacked expert layout |
| shared expert (none or one) | fp32 accumulation of routed and shared output |
| checkpoint file naming | load metrics |

## Why a preset per target family

The drafter's MoE follows a target family's recipe, expressed as a preset:

- **Serving.** The drafter runs inside SGLang's draft model; the family's
  routing and weight naming let that family's MoE code path serve it.
- **Latency.** At small batch, top-k of narrow experts reads about the same
  bytes as the dense MLP, so MoE buys parameters at roughly constant step
  cost. That is the hypothesis the dense-vs-MoE ablation tests.

## Layout

```
base.py        Registry, RoutingResult, TrainingState
config.py      MoEConfig + preset registry; resolves a draft JSON
router.py      score functions, Router ABC, TopKRouter
balance.py     BalanceController: none, aux_loss
experts.py     RoutedExperts ABC, GroupedExperts (stacked [E,out,in]), swiglu
shared.py      SharedExpert ABC, SwiGLUSharedExpert
layer.py       MoELayer, build_ffn, and the model-level hooks
checkpoint.py  checkpoint layouts
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

**Updates are deferred.** `observe` runs inside the layer forward (training
only) and may only stash, overwriting and never accumulating: an
activation-checkpoint recompute re-runs the forward and must leave identical
state behind, and a loss built there is consumed right after the same
forward. `apply_pending_update` runs from the *model* forward before any
routing, outside checkpoint regions, and may mutate selection state and run
collectives. Mutating selection state inside a layer forward would make the
recompute route differently and raise `CheckpointError`.

**Accumulation order.** `RoutedExperts.forward` returns the weighted routed
sum as a fresh fp32 tensor; `MoELayer` adds the shared expert into it and
casts to the input dtype once. Casting the routed sum to bf16 before adding
the shared expert differs by ~3e-3 relative, enough to flip top-k choices
downstream; the fp32 add costs under 2% of the layer's fwd+bwd on H200.

**Attribute names are the module-side contract.** `MoELayer` exposes `gate`
(with its balance controller at `gate.balance`), `experts` and
`shared_experts`; a family's layout maps them to its file naming.

**Naming is converted at the boundary, not in `state_dict()`.** FSDP's
full-state-dict hooks index the gathered dict by the module's own parameter
FQNs. Every file read/write calls `to_checkpoint_state_dict` /
`from_checkpoint_state_dict` with the draft model (any module tree) or its
config: `FSDPTrainingBackend` save/load, resume, warm start,
`AutoDraftModel.from_pretrained`, `materialize_draft` and the HF exporter.
Layouts act on one MoE layer at a time (the layer prefix is the text before
its `experts` component) and only the layout the config names runs.

**Model-level terms ride every strategy.** `DraftTrainStrategy.finalize_step_output`
(called by `TrainerCore` after `forward_loss`) adds the layers' summed
auxiliary loss to whichever term backpropagates (`loss`, or the `loss_terms`
numerator scaled by the micro-batch denominator) and the layer-averaged
`moe/*` metrics to `StepOutput.metrics`: `load_max_ratio`, `load_min_ratio`,
`experts_unused_frac`, plus the balance controller's (`aux_loss` and
`load_frac_*`). Strategies hold no MoE code; dense drafts log no `moe/*` keys;
evaluation calls `forward_loss` alone, so eval losses exclude the auxiliary
term.
The trainer DP-averages the `moe/*` metrics and logs them like any other
scalar.

## Extension points

- New target family: a module under `families/` with `register_moe_preset`
  (recipe + default layout) and `register_layout` (file naming).
- New routing mechanism: `register_score_function`, `register_router` or
  `register_balance_controller`.
- New expert implementation: a `RoutedExperts` subclass under
  `register_experts_backend` (or a new `MoEConfig.dispatch` value inside an
  existing backend), or a `SharedExpert` subclass under
  `register_shared_expert`.
- Ablations: any architecture key at the draft JSON top level overrides its
  preset default; `dflash_config.moe_*` sets training knobs.
