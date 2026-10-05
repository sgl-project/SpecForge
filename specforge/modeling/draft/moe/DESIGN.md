# MoE FFN Design (`specforge.modeling.draft.moe`)

Design note for the sparse-MoE FFN that any DFlash-family draft (DFlash,
DFlash2, DSpark) can opt into. The training plane's picture is in
[`../../../training/DESIGN.md`](../../../training/DESIGN.md).

## Responsibility

Owns the FFN of a decoder layer when the draft JSON sets
`n_routed_experts > 0`, and nothing else: attention, heads, losses and the
trainer loop are unchanged. The package is **one configurable layer**, not one
block per target family. This is the Megatron-Core split (one `MoELayer`,
router/balancing/experts/dispatcher/shared-expert as orthogonal components
picked by config), chosen over the transformers pattern of a copied
`XxxSparseMoeBlock` per model because the family differences are a handful of
small functions while the expensive parts are shared:

| differs per target family        | shared by every family                    |
| -------------------------------- | ----------------------------------------- |
| score function (softmax, sigmoid, sqrtsoftplus) | token dispatch (sorted segments, grouped GEMM) |
| balancing policy (aux loss, aux-loss-free bias, none) | deferred balance-update timing vs activation checkpointing |
| combine-weight renorm and scale  | FSDP-friendly stacked expert layout       |
| shared-expert gate (none, sigmoid) | checkpoint naming boundary               |
| SwiGLU clamp                     | load metrics, warm-start plans            |

## Why match the target's MoE

The drafter's MoE should be the *target's* MoE, expressed as a preset:

- **Warm start.** Same expert shape means the draft experts can be seeded from
  a subset of the target's, which a dense drafter cannot do.
- **Serving.** The drafter runs inside SGLang's draft model; matching the
  target's routing reuses its fused MoE kernels and weight naming.
- **Latency.** At small batch, top-k of narrow experts reads about the same
  bytes as the dense MLP, so MoE buys parameters at roughly constant step
  cost. That is the hypothesis the dense-vs-MoE ablation tests.

## Layout

```
config.py      MoEConfig + preset registry. Architecture keys use the target
               checkpoints' native HF names at the draft JSON top level;
               training-only knobs live under dflash_config as moe_*.
router.py      Router contract (x -> RoutingResult) + score-function registry.
balance.py     BalanceController contract; owns selection-bias buffers, stashes
               counts in forward, applies updates from the model forward.
experts.py     RoutedExperts contract (weights + dispatch), MoEConfig.dispatch knob.
shared.py      SharedExpert contract, gate variant via MoEConfig.shared_expert_gate.
layer.py       MoELayer = gate + experts + shared_experts; build_ffn() is the
               dense/MoE switch used by the DFlash decoder layer.
hooks.py       apply_pending_balance_updates / collect_moe_aux_loss /
               collect_moe_metrics over any module tree.
state_dict.py  to/from_checkpoint_state_dict: module layout <-> official names.
init.py        WarmStartPlan: which target experts seed which draft experts.
```

Implementations register into these registries at import time (imported at
the bottom of `__init__.py`):

```
topk_router.py     "topk" router; score functions softmax / sigmoid / sqrtsoftplus;
                   optional group-limited selection (DeepSeek top-2 group scores).
noaux_tc.py        "noaux_tc" controller: fp32 selection bias + sign controller on
                   all-reduced loads; converter gate.balance.bias <-> gate.bias.
grouped_experts.py "grouped" experts: stacked [E, out, in] w1/w2/w3, sorted-segment
                   loop or torch._grouped_mm dispatch; converter experts.w1 <->
                   experts.{i}.w1.weight.
swiglu_shared.py   "swiglu" ungated shared expert (shared_experts.w1/w2/w3).
presets.py         "deepseek_v4": sqrtsoftplus + noaux_tc + renorm x1.5 + one
                   ungated shared expert + SwiGLU clamp 10.
```

## Contracts that matter

**Attribute names are the checkpoint contract.** `MoELayer` exposes `gate`,
`experts`, `shared_experts` (the DeepSeek-family names SGLang loads). A
component whose native parameter layout differs from the official file naming
registers a `state_dict` converter pair; both directions are idempotent and
no-ops on dense models.

**Naming is converted at the boundary, not in `state_dict()`.** FSDP's
full-state-dict hooks index the gathered dict by the module's own parameter
FQNs, so a rename inside `state_dict()` breaks under `use_orig_params`. Every
file read/write goes through `to_checkpoint_state_dict` /
`from_checkpoint_state_dict`: `FSDPTrainingBackend` save/load, warm start,
`materialize_draft`, and the HF and SGLang exporters.

**Balance updates are deferred.** `BalanceController.observe` only stashes
(overwrite, never accumulate) so an activation-checkpoint recompute leaves
identical state. `apply_pending_update` runs from the *model* forward before
any routing, outside checkpoint regions, and may run collectives. Mutating
selection state inside a layer forward would make the recompute route
differently and raise `CheckpointError`.

**Metrics ride the existing scalar channel.** `collect_moe_metrics` yields
`moe/load_max_ratio`, `moe/load_min_ratio`, `moe/experts_unused_frac` plus
controller metrics; the DFlash/DSpark strategies add them to `StepOutput.metrics`,
and the trainer DP-averages and logs them like any other scalar.

**Aux losses are collected, not yet consumed.** `collect_moe_aux_loss` sums
scaled layer losses; wiring it into an objective is done with the first preset
whose balancing policy emits one (aux-loss-free policies do not).

## Extension points

- New target family: `register_moe_preset("<family>", scoring_func=..., ...)`
  plus any missing component registrations. The draft JSON then sets
  `moe_preset` and the per-run sizes.
- New dispatch: a `RoutedExperts` subclass under `register_experts_backend`,
  or a new `MoEConfig.dispatch` value handled inside an existing backend.
- Ablation knobs: any `ARCHITECTURE_KEYS` entry at the draft JSON top level
  overrides its preset default; `dflash_config.moe_*` overrides training knobs.
