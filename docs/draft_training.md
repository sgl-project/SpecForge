# Canonical draft objectives

`algorithms/objectives` owns training computation. `build_objective_model` wraps a
canonical draft and an optional frozen target head in one module, so distributed
training executes the entire differentiable computation through its wrapper.
Objectives register independently from model construction and checkpoint formats.
A registration declares supported architectures, input names, capabilities and parameter
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
`js`, `tv`, `nla`, `lk`, `lk_alpha`, `lk_tv`, `lk_lambda` and `ce_l1`. Distillation needs aligned teacher logits or final states
plus a frozen head. `default` preserves each algorithm's base loss. Selector,
confidence, refinement and alignment supervision remain owned by the algorithm.
Weighting accepts `uniform`, `exponential` (gamma), `geometric` (factor),
`dpace`, `dpace-cumulative-confidence-only` or `dpace-continuation-value-only`
(alpha, default 0.5).
Omitting it preserves block loss decay and EAGLE's geometric factor of 0.8.
An explicit weighting replaces that default; EAGLE does not apply another depth
factor. EAGLE/P-EAGLE apply weighting to rollout depth. DFlash2 selector warmup
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

Each protocol owns sampling, model execution and auxiliary supervision. It returns
`Predictions` containing hidden states, aligned labels/masks, a deferred teacher
source and model-owned `project()`. The shared executor checks capabilities,
projects logits, applies the selected loss and weighting, then reduces results.
Auxiliary selector, confidence and alignment losses remain protocol callbacks.
All differentiable work runs inside `ObjectiveModel.forward`, including projection.

`register_protocol` declares the capabilities a protocol provides;
`register_loss(name, requires=...)` declares what a loss consumes. Compatibility
is checked at construction and against the actual predictions before projection.
Teacher-dependent losses extend capture inputs; teacher projection runs only when
needed. Architecture checks select execution protocols, not permitted loss lists.

`Predictions.select()` slices complete projection units. DSpark's Markov head and
Domino's GRU keep a full block together; independent token projections use
single-token units. Projection does not rerun the backbone, and Domino computes
base and corrected logits together. This boundary supports later chunked projection
and fused-loss executors; this PR still uses the eager executor. Chunked DSpine
training must reuse the optimizer-window alignment multiplier across chunks.
D-PACE groups predictions by stable `chain_ids` and ordered `positions`, including
chains spanning EAGLE depth sets. `sequence_ids` balance valid anchors within each
sequence. Credit weights are detached; the denominator counts valid sequences,
not credit mass. This applies to explicit D-PACE selection, including protocols
whose default loss uses a microbatch mean. The eager executor materializes complete
chains before weighting; future chunked executors must preserve this normalization.

Legacy LK modes use `lk_alpha`, `lk_tv` and `lk_lambda`. Their `options.target` is
`labels` by default (one-hot acceptance), or `teacher` for distribution overlap.
`kl_scale` and `kl_decay` default to 1.0; lambda uses detached mean acceptance over
supervised positions. Teacher LK preserves the target probability mass outside a
reduced draft vocabulary, and uses soft cross-entropy for its legacy KL term.
The existing `lk` remains the per-token KL/TV hybrid, with configurable `eta`.

```python
runtime = build_objective_model(
    draft,
    loss={"kind": "lk_lambda", "options": {"kl_scale": 1.0, "kl_decay": 1.0}},
    weighting={"kind": "dpace", "alpha": 0.5},
)
```

For legacy EAGLE LK semantics, select `options.target="teacher"` and retain the
default geometric weighting. The training entry resolves loss and weighting
before capture, and records both in the exact-resume contract.

## Training entry point

The existing `specforge train -c run.yaml` command accepts an expanded draft
mapping or an ordinary YAML/JSON file. Presets use this same file path; there is
no separate preset selector. The architecture chooses the default objective.
`training.objective` can select another compatible registration, and its settings
are passed through `training.objective_options` (unknown options are errors).
`training.loss` and `training.weighting` select the common token objective and
position weights; omitting them preserves the protocol defaults. Capture inputs
combine protocol and loss requirements. A teacher head is loaded only when the
loss or protocol requires teacher logits; final-state inputs alone do not imply
a head is needed.
Legacy algorithm switches remain on the legacy route. On the canonical route,
use loss/weighting for the common objective and objective_options for sampling
and auxiliary training. DFlash2 selector_warmup_ratio/selector_ramp_ratio belong
in objective_options. Nondefault legacy loss/sampling settings are rejected.

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
  loss:
    kind: kl
  weighting:
    kind: exponential
    gamma: 7.0
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
none of these fields may mix with legacy `model.draft_*` source fields.

Checkpoints contain the canonical `config.json` alongside the existing
`training_state.pt` and per-rank optimizer/RNG files. The public loader reads only
the draft tensors, while the trainer alone restores training state. Exact resume
checks structure, resolved loss/weighting, objective/options, frozen prefixes,
target provenance and dtype. A changed or missing loss/weighting contract is
rejected rather than guessed.
An existing vocabulary mapping is preserved on resume; it is not regenerated from
the dataset. `training.frozen` lists module or parameter paths excluded from
training, and is part of the resume contract.

Canonical records use the tensor names listed above; EAGLE records using legacy
`hidden_state` / `aux_hidden_state` names must be regenerated with the canonical
capture layout or explicitly converted. The generic offline preparation and
server capture adapters now supply unshifted canonical records. DMTP currently
accepts offline records with `mtp_hidden_states`; the generic capture service
cannot produce frozen-MTP features and DMTP streaming fails validation.

The new route supports eager attention and data parallel training, including FSDP
and DDP. Frozen target bindings and the teacher head remain replicated outside
FSDP, matching the existing training backend's frozen-table policy. The canonical
model's requested dtype also controls FSDP computation. Optimized backends and
sequence-parallel execution remain on the legacy route. Online wiring is
validated through capture contracts; a live SGLang capture run is not part of the
current integration tests.

DSpine training uses a graph-free prepass and RNG-preserving replay for every
optimizer window, including a single microbatch. The strategy reduces backbone
CE once per window and passes the alignment coefficient into the objective.
Objective evaluation performs no collectives; the objective itself never reduces
its alignment statistics across ranks. Transfer-space initialization and
restoration remain owned by the public model loader.

Legacy checkpoints keep their original full-resume path (`model.draft_*` plus
`training.resume_from`). Importing one through `initialization.checkpoint` or
`weights_from` reads weights and effective settings only; it does not translate
optimizer shards, scheduler state or RNG. For legacy imports, explicit
`model.mask_token_id`, `model.aux_hidden_state_layer_ids` and
`training.norm_before_residual` must agree with recorded settings. Canonical YAML
keeps these settings in its architecture nodes.

For scratch training, `model.load_target_embedding: true` copies the target table
into an owned embedding through `from_target`, with shape/dtype validation.
Explicit target mappings take precedence; checkpoint and warm-start embeddings
are preserved. Target-bound tables always follow their bindings. Teacher heads
and draft bindings use the same target key resolver and cached source tensors;
a compatible bound head is shared directly.

Canonical training uses PyTorch kernels by default. `model.use_liger_kernel: true`
selects Liger RMSNorm and SiLU/SwiGLU on CUDA/ROCm; other activations retain their
PyTorch implementation. This option does not change parameter names or canonical
configuration and is recorded in the training resume contract.
