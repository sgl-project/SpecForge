# Canonical draft objectives

`algorithms/objectives` owns training computation. `build_objective_model` wraps a
canonical draft and an optional frozen target head in one module, so distributed
training executes the entire differentiable computation through its wrapper.
Objectives register independently from model construction and checkpoint formats.
A registration declares supported architectures, input/output names and parameter
prefixes it trains. A trainable parameter outside that coverage fails construction;
explicitly frozen parameters and target bindings are excluded.

The default objectives cover EAGLE3, P-EAGLE, DFlash, DFlash2, DSpark, Domino,
DSpine, MTP and DMTP. Shared sampling and mask construction are functions;
objectives do not inherit from or wrap another algorithm's training model.
`algorithms/losses/` holds chunked reductions and selects the numerical kernels
in `core/`; `algorithms/metrics/` holds diagnostics. Teacher and draft projections
receive separate heads.

Attention backends live in `modeling/draft/layers/attention/backends/`. Eager and
SDPA materialize visibility; Flex keeps callable visibility sparse. EAGLE owns
its recurrent execution and caller-scoped KV cache in `models/eagle_rollout.py`,
while its objective owns rollout depth and supervision. `training/sequence_parallel.py`
supplies USP process groups, and `training/canonical_data.py` prepares owned rows,
teacher overlap and global positions. Shared backbone execution has no algorithm hooks.

The composed components support optional Liger RMSNorm/SwiGLU and fused grouped
convolution without replacing modules or changing checkpoint keys. Runtime kernel
selection belongs to training; structural configuration nodes remain unchanged.

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

## Training entry point

The existing `specforge train -c run.yaml` command accepts an expanded draft
mapping or an ordinary YAML/JSON file. Presets use this same file path; there is
no separate preset selector. The architecture chooses the default objective.
`training.objective` can select another compatible registration, and its settings
are passed through `training.objective_options` (unknown options are errors).
Retired `model.draft_*` and top-level algorithm switches are rejected. Model
structure belongs to the draft YAML; loss and optimization settings belong to
`training.objective_options`.

Optimization options use `training.objective_options`:

| Objectives | Options |
|---|---|
| DFlash, DFlash2, Domino, DSpark, DSpine | `objective_chunk_blocks`, `objective_checkpointing`, `fused_logits_loss` |
| DFlash, DFlash2 | `loss_type` (including D-PACE), `lk_loss_type`, `kl_scale`, `kl_decay`, `teacher_metrics` |
| DFlash2 | `selector_loss_alpha`, `selector_stop_gradient`, warmup/ramp steps or ratios |
| EAGLE3 | `ttt_length`, `lk_loss_type`, `compact_teacher`, `compact_teacher_chunk_size`, `trim_loss_positions` |

`training.attention_backend` selects `eager`, `sdpa` or `flex_attention`.
EAGLE3 also supports cached `fa` and offline `usp` with causal full GQA/MHA.
USP uses `training.sp_ulysses_size` and `training.sp_ring_size`; trainer TP remains
unsupported. `model.use_liger_kernel` selects runtime Liger kernels. Setting
`SPECFORGE_DFLASH2_FUSED_CONV=0` selects the eager grouped convolution.

```yaml
model:
  target_model_path: /models/target
  torch_dtype: bfloat16

draft: /configs/dflash.yaml

data:
  hidden_states_path: /data/canonical-features

training:
  attention_backend: eager
  objective_options:
    num_anchors: 128
    loss_decay_gamma: 7.0
  max_steps: 1000
  save_interval: 100

output_dir: /output/draft
```

Only one loading path may be selected:

- `draft`: construct a new model with its declared target bindings.
- `initialization.checkpoint`: restore a complete draft through `load_draft`.
- `draft` plus `initialization.weights_from`: pass `checkpoint`, `mapping` and
  `allow_new` to the initialization planner. Optional `initialization.from_target`
  supplies disjoint strict target copies.
- `resume`: restore a training checkpoint through the existing trainer, including
  optimizer, scheduler, RNG and data position. No draft YAML is required. An
  optional draft must equal the saved structure exactly. Training options and
  the original schedule horizon must still match the recorded run contract.

`initialization.adapter` and `initialization.legacy_config` handle explicit
external checkpoint import. `resume` and initialization are mutually exclusive;
retired `model.draft_*` source fields are not accepted.

Checkpoints contain the canonical `config.json` alongside the existing
`training_state.pt` and per-rank optimizer/RNG files. The public loader reads only
the draft tensors, while the trainer alone restores training state. Exact resume
checks structure, objective/options, frozen prefixes, target provenance and dtype.
An existing vocabulary mapping is preserved on resume; it is not regenerated from
the dataset. `training.frozen` lists module or parameter paths excluded from
training, and is part of the resume contract.

Canonical records use the tensor names listed above. The offline reader also
accepts existing EAGLE `hidden_state` / `aux_hidden_state` fields and maps them
to canonical names without shifting their tokens. Packed P-EAGLE records retain
their optional `lengths` tensor. The generic offline preparation and
server capture adapters now supply unshifted canonical records. DMTP currently
accepts offline records with `mtp_hidden_states`; the generic capture service
cannot produce frozen-MTP features and DMTP streaming fails validation.

The training route supports the attention backends above and data parallel
training, including FSDP and DDP. Frozen target bindings and the teacher head remain replicated outside
FSDP, matching the existing training backend's frozen-table policy. The canonical
model's requested dtype also controls FSDP computation. Online wiring is
validated through capture contracts; a live SGLang capture run is not part of the
current integration tests.
