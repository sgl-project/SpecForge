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
sampling/mask construction. EAGLE3 runs each TTT depth once over a differentiable
KV prefix with explicit dense visibility; it does not add algorithm hooks or mutable
KV caches to the backbone.
Reference and optimized loss implementations share the same executor.
Training wrappers with capabilities not yet migrated, such as sequence parallelism,
remain available and read the same named loss configuration. MTP/DMTP CE is evaluated
in FP32 for stable log-softmax; it need not match the legacy BF16 reduction bitwise.

Loss and position weighting can be selected independently of the architecture:

```python
runtime = build_objective_model(
    draft, loss={"terms": {"token": {"kind": "kl"}}}, weighting={"kind": "exponential", "gamma": 7.0}
)
result = runtime(batch, step=optimizer_step, total_steps=total_optimizer_steps)
```

`loss.terms.<name>.kind` accepts `ce` (the protocol's labels: data tokens for block and
MTP protocols, the teacher's top-1 within the draft vocabulary for EAGLE3/P-EAGLE),
`soft_ce`, `kl`, `reverse_kl`,
`js`, `tv`, `nla`, `lk`, `lk_alpha`, `lk_tv`, `lk_lambda` and `ce_l1`. Distillation needs aligned teacher logits or final states
plus a frozen head. Omitting a term preserves its algorithm default. Selector,
confidence, refinement and alignment supervision remain owned by the algorithm.
Weighting accepts `uniform`, `exponential` (gamma), `geometric` (factor),
`dpace`, `dpace-cumulative-confidence-only` or `dpace-continuation-value-only`
(alpha, default 0.5).
Omitting it preserves block loss decay and EAGLE's geometric factor of 0.8.
An explicit weighting replaces that default; EAGLE does not apply another depth
factor, and `loss_decay_gamma` or `depth_decay` given with it is rejected, rather than silently ignored. EAGLE/P-EAGLE apply weighting to rollout depth; MTP proposes one token per
position and accepts only uniform weighting. DFlash2 selector warmup
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
On a reduced-vocabulary draft, a supervised depth whose teacher top-1 is outside the
draft vocabulary carries no loss and ends its chain, since the draft cannot propose it.

Legacy LK modes use `lk_alpha`, `lk_tv` and `lk_lambda`. Their `options.target` is
`labels` by default (one-hot acceptance), or `teacher` for distribution overlap.
`lk_lambda` also takes `kl_scale` and `kl_decay` (default 1.0) and uses detached mean
acceptance over supervised positions; the other modes reject them. Unknown loss
options fail when the loss is built. Teacher LK preserves the target probability mass outside a
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

The new route supports eager attention (EAGLE3 also SDPA) and data parallel
training, including FSDP and DDP. Frozen target bindings and the teacher head remain replicated outside
FSDP, matching the existing training backend's frozen-table policy. The canonical
model's requested dtype also controls FSDP computation. Other optimized backends and
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
into an owned embedding with shape validation. Like legacy `load_embedding`, this
default copy casts to `model.torch_dtype`; explicit `from_target` copies stay
bit-preserving and only widen. Explicit target mappings take precedence;
checkpoint and warm-start embeddings are preserved. Target-bound tables always
follow their bindings. Teacher heads
and draft bindings use the same target key resolver and cached source tensors;
a compatible bound head is shared directly.

Canonical training uses PyTorch kernels by default. `model.use_liger_kernel: true`
selects Liger RMSNorm and SiLU/SwiGLU on CUDA/ROCm; other activations retain their
PyTorch implementation. This option does not change parameter names or canonical
configuration and is recorded in the training resume contract.

### Loss execution

The preferred configuration names mathematical terms and uses one shared fusion
switch. Omitting fused means reference calculation.

```yaml
training:
  objective: dspark
  objective_options:
    num_anchors: 512
  loss:
    fused: true
    terms:
      token:
        kind: ce_l1
        options:
          ce_weight: 0.1
          l1_weight: 0.9
      confidence:
        kind: bce
        coefficient: 1.0
    scheduling:
      chunk_size: 128
      checkpoint: true
```

Algorithms declare stable input names and defaults through `LossDefinition`.
The resolver rejects unknown names or incompatible mathematical input contracts.
DSpark confidence uses binary scores and soft targets; DSpine alignment uses
feature groups. DSpine refinement additionally requires independently masked
CE/L1 components. Omitted terms retain algorithm defaults.

| Algorithm | Terms |
| --- | --- |
| DFlash, EAGLE3, P-EAGLE, MTP, DMTP | token |
| DSpark | token, confidence |
| DFlash2 | token, selector |
| DSpine | token, refinement, alignment |
| Domino | base, final |

A term's coefficient multiplies its algorithm-supplied schedule factor. Sampling
and input construction stay in objective_options. Mathematical parameters
and coefficients belong to loss.terms; loss.fused controls optimization. Dependent auxiliary
calculations call `calc.evaluate_term()` using those resolved settings.

Domino uses the same configuration shape; its algorithm declares joint inputs internally:

```yaml
training:
  objective: domino
  loss:
    fused: true
    terms:
      base: {kind: ce}
      final: {kind: ce}
    scheduling:
      chunk_size: 128
      checkpoint: true
```

With fused enabled, the executor selects the highest-priority compatible
implementation, including joint optimizations. Unsupported groups split into
individual terms; unsupported terms use reference formulas. Selection and fallback
reasons are logged. Missing dependencies and execution failures propagate.

Scheduling accepts chunk_size, checkpoint, trim and teacher_chunk_size.
Chunking and checkpointing are independent of fusion; checkpointing also works
without splitting. Losses needing whole-set statistics stay unsplit with a warning
when a requested split cannot be honored. Unsafe replay disables checkpointing
with a warning. Unchunked D-PACE also reports that its materialized
chain-weighting path cannot use loss checkpointing.

LossConfiguration.bind() installs formulas and coefficients while preserving
algorithm-owned joint ranges. The effective settings are stored in the resume
contract. Users do not configure groups, implementation names or term execution.

Only the named configuration above is accepted. Flat loss dictionaries,
separate `loss_execution`, old top-level loss flags, and loss coefficients in
`objective_options` are rejected. Retained training wrappers consume the same
configuration and reject formulas or scheduling they cannot execute.

- `chunk_size` counts complete prediction units (blocks for block protocols,
  independent rows for EAGLE/P-EAGLE/MTP). Omission preserves the protocol's
  default; zero disables chunking. Recurrent heads always keep whole blocks.
- `checkpoint` controls recomputation of chunk projections and losses during
  backward, including an unsplit prediction set. It defaults to true and is
  independent of fused execution.
- `fused` defaults to false. Registered CUDA FP16/BF16/FP32 kernels implement CE,
  soft CE, CE/L1 and supported divergences. DFlash2 selector and DSpine refinement
  use the same CE/L1 kernel; Domino's CE path consumes base and compact GRU
  corrections without constructing full corrected logits. CPU, FP64 and other
  unsupported generic contracts use a logged reference. Missing required kernel
  dependencies and execution failures raise.
  Empty inputs use the reference only to preserve zero gradients.
  The first selection for each prediction group is logged at INFO, checked across
  ranks, and recorded for resume. Rejection reasons explain lower-priority choices.
- `trim` defaults to false and skips prediction units with no supervised positions.
  It preserves explicit denominators and does not truncate recurrent blocks.
  D-PACE keeps complete chains; prediction-scope losses and scalar terms keep all rows.
  Trimming also respects both prediction and projection chunk-safety declarations.
- `teacher_chunk_size` enables compact teacher projection for offline
  vocabulary-aware protocols with a reduced vocabulary and a frozen linear
  teacher. It preserves full-vocabulary argmax and retained probability mass
  while projecting vocabulary tiles; no full-vocabulary logits are retained.

For plain DFlash/MTP linear heads, `fused: true` can use Liger linear CE without
retaining full logits. It requires a compatible Liger kernel that also returns
predicted tokens. Its installed version is checked against the project's Liger
requirement before loading the kernel; an out-of-range version raises, including
after a no-deps installation. An unavailable kernel also raises for an eligible
linear head.
Wrapped/transformed heads retain ordinary projection and may use a token kernel.

Chunked D-PACE first collects detached chain statistics, then recomputes projection
chunks with the complete-chain weights. Contextual LK losses retain whole-set
execution, and DSpine retains it when no window alignment statistic is supplied.
Thus requesting chunking does not promise bounded loss memory for those paths.

`model.use_fused_conv` opts canonical grouped convolution into its existing Triton
kernel. Enabling it without grouped convolution, or with unsupported weights or
group sizes, fails during setup. Incompatible runtime input dtypes and compiled
execution also raise. The selected path is logged during setup. Ordinary loading
through specforge.loading and exported models use eager convolution by default;
this training option is not stored in the model architecture.
The fused kernel accumulates in FP32 and rounds once, so
BF16 results need not be bit-identical to eager per-tap rounding. Parameter names
and checkpoint tensors do not change. Liger selection remains independent.

Execution settings are saved and checked on full resume. Checkpoints must record
the current named loss settings and execution policy; old loss metadata is not
converted or assigned defaults. Fused runs must also record selected implementation
identities. Import model weights and start a new run when this contract is absent.

MTP uses `loss.scheduling.chunk_size` to control loss projection chunks.
Its objective's default chunk size remains 4096.

`resume` and `training.resume_from` are alternative inputs for one resolved resume
path. Neither is copied into the other. A canonical checkpoint supplied through
`training.resume_from` also selects the canonical training route. Legacy DSpine
imports through `initialization.checkpoint` must explicitly match saved training
sampling settings in `training.objective_options`. Loss settings always come from
the new training configuration; `weights_from` copies weights only.


### Extending loss execution

Algorithm implementations live in `specforge/algorithms/objectives/protocols/`:
one module per algorithm, imported by that package to register the built-in
protocols. The parent `objectives/` package holds shared contracts, registration,
loss definitions and execution. New algorithms belong in `protocols/`; they reuse
the public registration and execution interfaces without adding algorithm branches
to shared modules.


Objectives describe supervision and return `Predictions` carrying a `Projection`
and an immutable `LossPlan`. The projection exposes ordinary named outputs and,
optionally, a typed optimization contract. It references existing model parameters
and must not bypass wrapping, sharding or head transforms.

`LossPlan.terms` declares outputs, coefficients and optional per-term loss, mask,
weighting and fixed denominator. Omitted settings inherit the prediction/runtime
defaults. `joint_groups` contains non-overlapping tuples of term indices that
**may** execute together; it does not select a kernel or combine their denominators.
Without a matching joint implementation, the executor selects an implementation
for each term. Removing a joint range preserves the mathematical objective.

Domino declares its existing `lambda * base_CE + (1 - lambda) * final_CE` schedule
and a `DominoSuffixProjection` contract. Its `domino_suffix_ce` implementation
uses compact corrections; the reference path computes both outputs normally.
There is no algorithm-name branch in the executor.

Execution adapters live in `objectives/implementations/`: `registry.py` owns
registration and selection, `linear_ce.py` implements linear fusion, and `ce.py`
contains shared CE applicability checks. `token_kernels.py` registers token-loss
adapters. Shared numerical formulas and kernels live in
`specforge/losses/fused.py`, with CE/L1 Triton details in
`specforge/losses/_fused_logits_loss_triton.py`. Both the mathematical loss layer
and execution adapters depend on this library; it imports neither protocols nor
registries. Import numerical losses from `specforge.losses`; the old loss
module paths are no longer supported.
`DominoSuffixProjection` lives with the other contracts in `projection.py`;
`implementations/domino_ce.py` contains only its execution adapter.
Fusion requires one complete block per
projection unit, additive corrections on a contiguous suffix, and base/final CE
terms with shared supervision weights. Other layouts fall back to compatible per-term implementations or reference calculation.

All final and intermediate calculations use `compute_loss(request, group, ...)`.
The algorithm chooses the mathematical terms and their coefficients. The loss
entry chooses how to evaluate them:

| Mode | Configuration | Selection |
| --- | --- | --- |
| Reference | fused: false | Registered mathematical formulas |
| Optimized | fused: true | Compatible joint or per-term optimization, otherwise reference |

Both generic kernels and projection-specific kernels participate in automatic
selection. Matching checks the mathematical loss, projection, requested outputs,
device and dtype. Runtime errors and missing required dependencies propagate.
The execution API has no explicit implementation argument.

Register formulas with `register_loss`; its optional `outputs` callback provides
extra reference results without changing the formula. CE's L1/top-k outputs use
the same reference and token CE paths, not separate statistics registrations.
Register implementations with `register_implementation(category=..., losses=...)`,
or use `register_token_kernel` for generic per-token optimization.
`implementation_registrations()` exposes the names, categories, loss mapping and
applicability callbacks for inspection without a second routing table.

Domino declares the base/final joint range; the shared selector matches
domino_suffix_ce automatically when fused is enabled. Explicit implementation
names are not accepted.

Implementations return weighted numerators by term index, plus predictions or
requested statistics; they do not apply coefficients or final normalization.
Selection uses priority and stable names. INFO logs report the first selection
and fallback reasons, and distributed/resume validation retains the actual name.

Term statistics are combined across chunks before normalization. Independent
masks/denominators require `sum_means` (normalize each prediction set) or
`global_microbatch` (normalize each term across ranks during training). The existing
`tokens` contract and D-PACE require shared term reduction. Differentiable
denominators and arbitrary accumulation-window normalization are not implemented.
Existing trainer accumulation and protocol semantics are unchanged.

Projection outputs and intermediate results are scoped to one chunk. No logits
cache is stored on the model. Loss registrations distinguish context inputs from statistics scope: prediction-scope
losses stay unsplit; per-position contextual losses can share fixed whole-set counts.
Algorithms must set `chunk_safe=False` when their computation couples units.

All objective terms live in `LossPlan.terms`. By default the executor runs its
groups directly. Algorithms with dependent computations supply
`Predictions.compute(calc)`, executed inside each chunk and checkpoint replay:

- `calc.run((0,))` executes a declared term (or a joint group).
- `calc.evaluate(key, loss, logits, labels, teacher=...)` requests per-position
  values and statistics through the same implementation registry. Request extra
  CE outputs explicitly with `return_teacher_l1=True` or `top_k=k`; these
  do not change the mathematical loss or its coefficient.
- `calc.add(index, values)` supplies a computed tensor to a declared
  `TokenLoss("value")` term. It uses the same weighting and denominator rules.
- `calc.ratios` holds additive metric pairs.

Every declared term executes exactly once. Intermediate keys must be unique and
stable across ranks, steps and checkpoint replays. Reference and optimized
implementations share the result contract: weighted numerators for final terms,
or differentiable per-position values for intermediate requests. Projection-fused
implementations can decline requests requiring intermediate values.
Statistics retain gradients; algorithms explicitly detach supervision targets.

For example, DSpark reuses `teacher_l1` from the main loss, constructs a detached
acceptance target, and requests BCE through the registry. DFlash2 requests top-k
CE statistics, runs its selector, then requests candidate CE. DSpine similarly
keeps refinement and alignment logic in its protocol. There is no separate
primitive registry or auxiliary-loss execution channel.

A MoE model can expose per-token router z-loss through `context_inputs` and add
it as a value term with its own coefficient and mask. This preserves gradients
to router parameters and supports additive chunk reduction. Load-balancing losses
that depend on whole-batch expert statistics require whole-set execution and
explicit normalization; do not recompute them independently per chunk. Router
statistics, cross-rank aggregation, and optimizer-step updates remain model/training
responsibilities, not loss-kernel responsibilities. Scalar model losses use `LossTerm(..., loss=TokenLoss("value"), reduction="scalar")`
and `calc.add(index, scalar)`. Such a plan runs unsplit and untrimmed, counts the scalar
once per prediction set, and applies no token mask or denominator. It requires
sum_means or global_microbatch reduction. The latter averages local scalars across
ranks; cross-rank expert statistics must be computed explicitly by the model.
No router state updates run inside loss callbacks or checkpoint recomputation.

Fused resume contracts include stable implementation identities for intermediate
calls as well as final terms. Checkpoints using the previous auxiliary primitive
identities fail the existing resume check rather than silently changing execution.
The equivalent CE statistics names ce_statistics and fused_ce_statistics map to
reference and token_ce on restore; other identity changes still fail.

The fused soft CE kernel is new in this PR, rather than migrated from legacy.
Its BF16 loss and gradients are checked against the legacy `LogSoftmaxLoss`
on a 128,256-token vocabulary.

Functionality and optimization support are separate: a method can use its ordinary
projection without supporting linear/Domino suffix fusion. TP, LoRA and quantized heads
need explicit compatible projection implementations before using these fused paths.

### Canonical training requirements

`register_protocol(..., training=TrainingRequirements(...))` declares capture
method, vocabulary mapping, batch-size constraints, `attention_backends` and an
optional `loss_mask_filter` callable. EAGLE3 also accepts `sdpa`; the other objectives keep eager attention;
block objectives register the shared consecutive-supervision filter. Canonical
composition builds providers directly from these requirements; it does not resolve
a legacy algorithm registration. An optional `window_statistic=WindowStatistic(ratio_name=..., argument_name=...)`
requests accumulation-window reduction and replay, passing the result to the
objective by the declared argument name.

`legacy_strategy` is separate registration metadata that preserves existing data
and rendezvous identifiers; it does not select a legacy provider.
`supports_vocab_mapping` declares draft-vocabulary support.
Worker counts belong to `data.dataloader_num_workers`, not objective requirements.
Canonical runs use a shared default of 4; set this explicitly to 8 to retain the
previous DFlash-family/MTP default. Legacy provider defaults are unchanged.

Canonical parsing, training and capture read target metadata from the same raw
`config.json` reader, including nested `text_config`. Transformers defaults are
not added to the canonical configuration.

Parameters frozen by `training.frozen` or an objective must have loaded weights:
a complete checkpoint, `weights_from`, `from_target`, or a target binding. An
objective's frozen head that the draft disables (DSpark's confidence head at
`loss.terms.confidence.coefficient: 0`) needs none.
Freezing a newly initialized parameter fails before model allocation; partial
imports must cover every frozen parameter, including tied aliases.

Prepare canonical records from the run YAML used for training, so the records
carry what its objective and loss read (for example `target_last_hidden_states`
for `loss: {terms: {token: {kind: kl}}}`):

```bash
python scripts/prepare_hidden_states.py -c /configs/run.yaml \
  --server-urls http://localhost:30000 \
  --data-path /data/train.jsonl --output-path /data/canonical-features
```

`--target-model-path` defaults to the run's `model.target_model_path`.
`--draft /configs/eagle3.yaml` captures for the default objective and loss only.
`-c`, `--draft` and the legacy `--draft-model-config` are mutually exclusive. The
canonical YAML selects the architecture, capture layers and output schema.


### Compatibility boundary

Old canonical checkpoints normalize their loss formulas, coefficients and
scheduling before resume comparison. Known equivalent CE-statistics identities
are translated; DSpine's old CE-plus-L1 refinement identity maps to CE/L1, and
its unchanged cosine calculation gains a reference identity. Unknown execution
changes still fail the resume check.

Old training wrappers remain for capabilities not yet migrated, including their
full resume and parallel execution paths. They share specforge/losses numerical
functions; this PR does not remove those training paths.

Use `loss_registrations()` and `implementation_registrations()` to inspect the
formula-to-implementation mapping. Each execution registration names its category,
mathematical loss names, and a supports callback for dynamic constraints.


Generate the implementation inventory directly from the registry:

```bash
python -m specforge.algorithms.objectives.implementations
```


### Diagnostic metrics

The trainer computes detailed block diagnostics only on logging steps. These
reports run without gradients, in bounded projection chunks, and preserve
additive counts and numerator/denominator pairs across accumulation and ranks.
They do not participate in loss checkpoint recomputation.

DFlash/DFlash2 retain hard-label, selector, prefix and sampled-anchor walk
metrics. Set training.objective_options.teacher_metrics to true to request their
teacher-distribution metrics and the required target features; the default is
false. DFlash also accepts metric_top_k (default 16). Domino and DSpark retain
sampled-anchor walk reports. DSpine reports backbone, refinement, alignment and
candidate-coverage statistics from the objective's existing calculations.
