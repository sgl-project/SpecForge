# Training backends: program, step runner, trainer (`specforge.training`)

**Status: proposal, sixth revision.** Nothing in this note is implemented and nothing in it was
run. The design that runs today is [`DESIGN.md`](DESIGN.md); the cross-plane picture is
[`../runtime/ARCHITECTURE.md`](../runtime/ARCHITECTURE.md). Symbols cited as "today" were read on
`main` at `d4391211`, after the canonical training migration removed the legacy step paths. The
section [Decisions for maintainers](#decisions-for-maintainers) lists the choices this note does
not make.

## Goals

1. **Attach an external training engine** (first: [torchtitan](https://github.com/pytorch/torchtitan))
   without changing an objective, a draft definition or the data path.
2. **Allow a trainer dedicated to draft training**, accepted on evidence rather than on review of
   its internals.
3. **Keep the API small and closed**: a few value types and protocols, each with one owner and a
   conformance test.

Out of scope: fault tolerance, a run controller, changes to capture or transport, and how a loss
is executed inside a program (chunking, which kernel computes a term). The note fixes only how
that choice is requested, recorded and agreed.

**Not yet shown.** No measurement in the repository shows that the step mechanics, rather than the
data path, bound training throughput, and no gain has been attributed to torchtitan for a small
trainable draft beside large frozen tables. The first deliverable of this work is that
measurement; see [Delivery](#delivery).

Throughput is one of two motives for the first goal. The other is coverage: the hardware and model
features an external engine maintains. The measurement settles the first motive only. Whether
coverage alone justifies the work, and whether such an engine is meant to become the default, is
[decision 9](#decisions-for-maintainers).

## What the design starts from

| # | Statement | Kind |
|---|---|---|
| 1 | The target is frozen and only prefills. A captured feature depends on target weights and tokens, not on the draft. There is no weight sync and no staleness, and recorded features can be replayed into any trainer | workload |
| 2 | Target execution belongs to the inference plane ([`../inference/DESIGN.md`](../inference/DESIGN.md#ownership-of-target-execution)). A trainer never runs target layers | decision |
| 3 | The loss denominator is not known before the forward: anchors are sampled inside the objective and some weightings depend on the draft's own predictions. An update is therefore normalised after the backward | consequence of how today's objectives sample; a backend cannot assume otherwise |
| 4 | The unit a backend can treat uniformly is a program: `program(tensors, ctx) -> ObjectiveResult`. An objective may drive the draft several times per batch (the EAGLE3 rollout, the DSpine two-pass window) and may or may not materialize full logits. A backend cannot assume logits exist | workload |
| 5 | The frozen target tables are large and are never sharded or optimized; the trainable draft is a few layers. Today the default wraps the draft with FSDP `SHARD_GRAD_OP` (`training.fsdp_sharding`); `NO_SHARD` selects DDP | fact about today |
| 6 | The artifact is the product: a canonical draft directory goes to export and to SGLang. Which trainer produced it must not show | decision |
| 7 | Canonical training has one model-parallel axis: sequence parallelism (ring and Ulysses attention) for an objective that declares `supports_sequence_parallel`, which is EAGLE3, under `attention_backend: usp`. The trainer's tensor-parallel size must be one (`training/canonical.py`). Four things happen in four places: the data plane cuts each sequence into per-rank shards; attention exchanges data on groups attached to the model as attributes (`training/sequence_parallel.py`); the LK losses sum per-sequence statistics over the sequence-parallel group inside the objective; and the accumulation count is multiplied by the sequence-parallel size (`training/assembly.py`) while the wrapper synchronises gradients over the whole world. No bundled recipe sets a sequence-parallel size above one | fact about today |

## Process model

This is fixed outside the training plane and a backend must fit inside it.

- **Ranks are launched before any trainer code runs**, by the launcher (`torchrun`). Training is
  SPMD: every protocol method below runs on every rank, in the same order, with one exception for
  `evaluate` (see [Rank symmetry](#rank-symmetry)).
- **The world is initialized once**, in `specforge.cli`: `init_distributed` creates the process
  groups, including the tensor- and sequence-parallel ones, and keeps them as module state in
  `specforge.distributed`; `set_seed(training.seed)` seeds every rank with the same seed. The data
  plane then creates its own groups in a fixed order (`runtime/control_plane/dp_ack.py`).
- **A runner derives what it needs from that world.** It does not initialize or destroy it, does
  not create groups in a rank-dependent order, does not re-seed, and does not change process-wide
  torch state without declaring it. It is the only part that hands a process group to a component
  (see [Two axes of communication](#two-axes-of-communication)).
- **Groups are created in one order, before the first batch.** Today that is the world and its
  meshes in `specforge.cli`, then the data plane's acknowledgement group where a run uses one
  (`build_disagg_online_consumer`), then the trainer. `runner.prepare` takes the trainer's place:
  a runner whose engine creates groups does so there, on every rank in the same order, and nothing
  creates a group after `prepare` returns. The composition root owns the order.
- **The launcher owns process lifetime.** A trainer that starts helper processes owns their death.

torchtitan's `TrainingEngine` constructor binds the device, applies NUMA binding, seeds RNG, sets
process-group timeouts and calls its own `init_distributed`, which returns early when the world is
already initialized. An adapter has to make each of those either a no-op or the single owner; see
[Attaching torchtitan](#attaching-torchtitan).

## Today

`Trainer -> TrainerController -> TrainerCore -> CanonicalTrainStrategy -> FSDPTrainingBackend`
([`DESIGN.md`](DESIGN.md)). There is one strategy and one producer of models,
`training.canonical.build_bundle`. A `TrainingBackend` ABC exists and has one implementation. What
keeps a second one out:

| Coupling | Where |
|---|---|
| The model learns and tells facts through attributes set after construction | `build_bundle` sets `frozen_target_modules` and `fsdp_param_dtype`, and under sequence parallelism `ring_group`, `ulysses_group` and a reduction callable; `FSDPTrainingBackend` reads the first two and `_no_split_modules` with `getattr`, and the EAGLE3 objective reads the groups the same way |
| Collectives are issued outside the backend | `objectives.base.microbatch_mean`, the window replay in `CanonicalTrainStrategy`, `ExecutionSelection.agree`, `TrainerCore._normalize_gradients`, `BF16Optimizer`, the evaluator, `CheckpointManager` |
| The step core rescales gradients assuming the wrapper averages across ranks | `TrainerCore._normalize_gradients` multiplies by world size |
| Structure, weights, target tables and kernels are built in one call | `training.canonical.build_bundle` |
| A checkpoint mixes the draft weights with the backend's state | `training_state.pt` holds the draft weights, the loop counters and, under DDP, the optimizer; `training_state_rank{r}.pt` holds FSDP optimizer state and every rank's RNG |
| Process groups are process-global | `specforge.distributed` holds the meshes and groups; layers, data utilities, the batch partition and the trainer read them through getters |
| Precision is decided in three places | `model.torch_dtype` becomes the `fsdp_param_dtype` attribute; the wrapper sets `MixedPrecision(param_dtype=..., buffer_dtype=float32)` and no `reduce_dtype`, so gradients are reduced in the parameter dtype; `BF16Optimizer` keeps fp32 master copies |

Each side has one fixed producer as well: `Trainer.__init__` constructs `FSDPTrainingBackend`, and
`training.assembly` constructs `BF16Optimizer`.

## The contract

Two levels. The first is specified here; the second is introduced only together with its first
real implementation (see [A dedicated trainer](#a-dedicated-trainer)).

```mermaid
flowchart TB
  SPEC["Specs<br/>program, optim, precision, topology, run"]
  subgraph REF["Reference trainer"]
    direction TB
    LOOP["Run loop<br/>windows, cadence, commits, checkpoint sequence"]
    FRAME["Step frame<br/>window walk, replay order"]
    LOOP --> FRAME
    FRAME --> FSDP["DDP / FSDP runner<br/>today's backend"]
    FRAME --> PLAIN["Plain runner<br/>one process"]
    FRAME --> TITAN["torchtitan runner<br/>later"]
  end
  ART["Canonical draft directory, metric records, sample commits"]
  KIT["Conformance kit"]
  SPEC --> LOOP
  FSDP --> ART
  PLAIN --> ART
  TITAN --> ART
  ART --> KIT
```

```python
# training/contracts.py: value types and protocols only

@dataclass(frozen=True)
class ProgramSpec:                               # pure data; recorded; available without building a model
    draft: DraftConfig
    objective: ObjectiveSelection                # objective, loss, weighting, options
    kernels: Kernels                             # the requested policy for every implementation choice
    requires: DataRequirements                   # features, capture layers, batch constraint, host inputs, sample cost
    reduction: Literal["window", "microbatch"]   # how the loss is normalised; see "Reductions"
    statistics: tuple[StatisticSpec, ...]        # name and delivery point of each declared statistic
    partition: PartitionHints                    # indivisible blocks; tensors pinned to their own dtype
    axes: Mapping[str, Literal["copy", "part", "own"]]   # model-parallel axes it communicates on

class TrainingProgram(Protocol):                 # today's ObjectiveModel
    draft: nn.Module                             # trainable; `draft_state` exports it under canonical keys
    frozen: tuple[nn.Module, ...]                # target tables, teacher head: never sharded or optimized
    def forward(self, tensors: Mapping[str, Tensor], ctx: StepContext) -> ObjectiveResult: ...
    def apply_statistic(self, name: str, value: Tensor) -> None: ...   # no collective; same on every rank
    def kernel_record(self) -> Mapping[str, str]: ...                  # realised kernels so far
    def expect_kernels(self, record: Mapping[str, str]) -> None: ...   # on resume; checked when inputs arrive

class ProgramFactory(Protocol):
    spec: ProgramSpec
    def build(self, *, device, precision: PrecisionSpec,
              groups: Mapping[str, ProcessGroup]) -> TrainingProgram   # construction; reads no weights
    def load(self, program: TrainingProgram) -> None   # weights, target bindings, teacher head

@dataclass(frozen=True)
class StepContext:                               # values only; no callables, no process groups
    step: int
    total_steps: int
    accumulation_steps: int                      # micro-batches per update
    collect_detailed_metrics: bool               # true on logging steps; in evaluation, what the objective declares
    replay_values: Mapping[str, Tensor]          # pooled `replay` statistics by name; empty outside a replay

class StepRunner(Protocol):
    @classmethod
    def supports(cls, spec: ProgramSpec, optim: OptimSpec, precision: PrecisionSpec,
                 topology: Topology, run: RunSpec) -> Supported | Unsupported: ...
    def prepare(self, factory: ProgramFactory, optim: OptimSpec, precision: PrecisionSpec,
                topology: Topology, run: RunSpec) -> None: ...
    def forward_backward(self, batch: TrainBatch, ctx: StepContext, *,
                         is_boundary: bool) -> PendingStep | None: ...   # one micro-batch
    def apply_step(self, pending: PendingStep) -> StepMetrics: ...    # consumes the handle
    def prepass(self, batch: TrainBatch, ctx: StepContext) -> ObjectiveResult: ...   # training mode, no gradients
    def evaluate(self, batch: TrainBatch, ctx: StepContext) -> ObjectiveResult: ...  # eval mode; see "Reductions"
    def reduce_sum(self, values: Tensor) -> Tensor: ...
    def reduce_max(self, values: Tensor) -> Tensor: ...               # sizes a buffer when a rank's shard is empty
    def agree(self, error: str | None, *, value: object = None) -> None: ...   # see "Rank symmetry"
    def draft_state(self) -> dict[str, Tensor]: ...                   # full on rank 0, empty elsewhere
    def save_resume(self, out: Path) -> ResumeRecord: ...             # this rank's record; the next `agree` compares
    def load_resume(self, src: Path, record: ResumeRecord) -> None: ...
    def close(self) -> None: ...
```

### Resolved values

The composition root (`application`) resolves these once, before any model is built, and compares
them across ranks. Every checkpoint's run record carries them. A runner honours every field or
refuses in `supports()`. It never substitutes a value. Everything `supports()` decides on is in
these values and in `ProgramSpec`, so the verdict needs no model.

| Value | Holds | Today |
|---|---|---|
| `Kernels` | the requested policy for every implementation choice that leaves the math alone but may move rounding or memory: attention backend, loss chunking and checkpointing, and for each kernel family on, off or auto | `training.attention_backend`, `training.loss.execution`, and three tri-state switches where unset means auto: `training.loss.fused`, `model.use_fused_conv`, `model.use_liger_kernel` |
| `PrecisionSpec` | five dtypes: `parameter` (as stored), `compute` (the forward), `reduce` (gradient communication), `master` (optimizer copies, or none), `buffer` (a cast applied by the wrapper, or none) | bf16, bf16, bf16, fp32 for the default `model.torch_dtype`; buffers are cast to fp32 under FSDP and left as built under DDP. See the last row of the table in [Today](#today) |
| `Topology` | `world_size`; for the data-parallel axis and each model-parallel axis, its size and the group index of every rank | the arguments of `init_distributed` and the two meshes it builds in rank order, read back through getters |
| `OptimSpec` | learning rate, schedule and warmup, total steps, clip norm, master-weight placement | `training.*`; `BF16Optimizer` |
| `RunSpec` | accumulation window, horizon, seed, cadences | `training.*` |

- **Requested at startup, realised when the facts exist.** `Kernels` is the request, compared across
  ranks at startup. An explicit request that cannot run is an error, never a fallback. Build-time
  kernels are realised in `build_bundle`. Liger under auto follows facts that are equal on every
  rank and raises if its package is missing. The fused convolution under auto falls back to the
  eager kernel where it cannot run, so two ranks with the same unset request can differ when one of
  them lacks Triton; `prepare` therefore ends by comparing the realised build-time kernels across
  ranks. The loss kernel is resolved for each prediction and term group when the first real input
  arrives, as `ExecutionSelection` does today, and may depend only on facts that are equal on every
  rank. `save_resume` returns this rank's realised kernels in its `ResumeRecord`, and the `agree`
  that follows compares the records. No forward issues a collective to agree on a kernel.
- **An implementation variant of a component is a kernel.** It is registered next to the component
  and selected through `Kernels`, and it carries a parity test against the reference implementation.
  If it stores parameters differently, each canonical entry of its state dict still has to be a
  writable view of the stored parameter, or the component provides an explicit import and export. A
  conversion hidden inside `state_dict` is not enough, because the weight plan writes through the
  tensors that call returns. In this design a runner does not bring its own implementation of a
  component; whether that can hold for an engine with its own module protocol is decision 11.
- **No dtype is inferred.** A parameter or buffer that keeps a dtype of its own (an fp32 buffer
  inside a bf16 draft) is pinned by its component through `PartitionHints`, and a runner that
  cannot keep a pin refuses.
- **`Topology` replaces reads of process-global state** by the data plane (which ranks share a
  batch, who acknowledges), the frame and the runner. A runner asserts in `prepare` that the groups
  it uses, whether created at startup or by its engine, match the value.

The remaining names: `ProgramSpec` is derived from the objective registry (`ResolvedObjective` and
`TrainingRequirements`) wherever the registry holds the fact. Two facts are not there yet and move
in step 3: the reduction, a literal on the `PredictionBatch` a protocol returns on every call, and
the indivisible blocks, which the backend reads from `_no_split_modules`. `DataRequirements` is the
union of what the draft, the objective and the loss need from the data plane. `StatisticSpec` is a
name and a delivery point. `PendingStep` is the single-use handle that the boundary call of
`forward_backward` returns: the window's reduced denominators, metric sums and statistic sums.
`StepMetrics` is the reduced loss, the reduced metric pairs, the pre-clip gradient norm, the
learning rate applied and the pooled `after_step` statistics; metrics keep both the ratio pairs and
the raw sums a program reports. `Supported` carries the numerics class; `Unsupported` carries the
reason. `ResumeRecord` is what a runner returns from `save_resume` and takes back in `load_resume`:
the runner's name and version and the realised kernels, build-time and loss. The run loop passes it
as the `value` of the next `agree`, which is where a difference between ranks raises, and stores it
in the run record.

### The window

One optimizer update consumes a window: a lazy, non-empty stream of at most `ctx.accumulation_steps`
micro-batches that the run loop forms across epoch boundaries, so accumulation does not restart at
an epoch. Three batches per epoch, two epochs and an accumulation of two give three updates. **The
frame is the only part that walks a window.** It takes micro-batches one at a time and hands each to
the runner as soon as it arrives:

```python
def run_window(runner, window: Iterator[TrainBatch], ctx) -> StepMetrics:   # the frame, shared by every runner
    pending = None
    for index, batch in enumerate(window, start=1):
        pending = runner.forward_backward(batch, ctx, is_boundary=index == ctx.accumulation_steps)
    if pending is None:
        raise IncompleteWindow     # the stream ended inside a window: no step, no commit
    return runner.apply_step(pending)
```

This is the path without a `replay` statistic; [Step statistics](#step-statistics) has the other.

- **Nothing is collected first.** A batch already holds tensors, and an online loader may deliver
  them on the training device, so holding a window would raise peak memory and delay the first
  backward. Five bundled recipes accumulate 512 micro-batches.
- **One call covers one micro-batch's forward and backward**, so a runner's synchronisation scope
  (`no_sync` under DDP) encloses both, and under `microbatch` the reduction sits between them.
- **Only a program that declares a `replay` statistic pays for buffering.** The frame then keeps
  host copies of the window's inputs, as `CanonicalTrainStrategy` and
  `TrainerCore._prepare_and_replay` do today.
- **A short window is an error.** A window is short only when the whole training stream ends inside
  it, never at an epoch boundary. The frame then raises and the run loop stops, as
  `TrainerController` does today: no partial step is applied and nothing is committed.
- **The boundary call returns the handle.** `forward_backward` returns `None` until `is_boundary`,
  then the `PendingStep` that `apply_step` consumes. The run loop commits the window's samples after
  `apply_step` returns.

This is today's shape: `TrainerCore.train_step` computes one micro-batch inside
`backend.accumulation_context(is_boundary=...)`. The consequence is that a runner executes
micro-batches and does not schedule them. An engine that wants to schedule a whole window itself
belongs to the [dedicated-trainer](#a-dedicated-trainer) level, not to this one.

### Reductions

`ObjectiveResult` carries the loss, metric pairs, raw metric sums and the local sums of the
program's declared statistics. `ProgramSpec.reduction` declares one **reduction** for the program:

| Reduction | The program returns | The update's objective over a window of `A` micro-batches and the data-parallel ranks | Who reduces |
|---|---|---|---|
| `window` (default) | one numerator, the coefficient-weighted sum of its terms, and **one** denominator | `sum(numerator) / sum(denominator)`, both sums over every micro-batch and rank | the runner, once, in the boundary call |
| `microbatch` | terms `(coefficient, numerator, denominator)`, each with its own denominator | the mean over the `A` micro-batches of `sum_t c_t * (sum over ranks of numerator_t) / (sum over ranks of denominator_t)` | the runner, inside every call, between forward and backward, in one reduction for all terms |

**One denominator per window.** Under `window` the gradient is accumulated first and rescaled once,
so it can carry one denominator only. An objective whose terms need different denominators has
three choices, made when it is resolved and never at run time: declare `microbatch`; normalise each
term locally and return a denominator of one (the mean of local losses); or be refused, with the
terms named. Several terms that share one denominator need none of this and stay under `window`:
that is every multi-term canonical objective today (DFlash2, Domino, DSpine) except DSpark's, which
is `microbatch`. The three reductions in `objectives/execution.py` map onto these: `tokens` is
`window`; `sum_means` is `window` with locally normalised terms and a denominator of one, except
under D-PACE weighting, where it keeps a pooled denominator like `tokens`; `global_microbatch` is
`microbatch`.

The program never issues a collective across samples and never needs the data-parallel size. The
two reductions are what `TrainerCore` and `microbatch_mean` compute today in exact arithmetic for
every canonical objective. Under `window`, each numerator is divided by the accumulation count, the
wrapper averages over ranks, and the gradients are rescaled by world size x accumulation / global
denominator. Under `microbatch`, the denominator is reduced before the backward and the later
rescale is by one. How a runner gets there (averaging or summing across ranks, where it divides) is
its own business. The formula does not fix rounding: see [Numerics classes](#numerics-classes).

**Evaluation is reduced differently.** `evaluate` returns the result as the program computes it
outside training: a `microbatch` program normalises each term on its own rank and returns a
denominator of one, and no reduction happens inside the call. The evaluator sums numerators and
denominators over the pass and pools them once at the end through `reduce_sum`, with one
`reduce_max` over the per-position length so that a rank with an empty shard can size its buffer, as
`Evaluator.run` does today. Applying a training reduction would change results: with two ranks
holding 1/1 and 0/9, today's DSpark evaluation loss is 0.5 and the training rule gives 0.1, which
moves `eval/avg_loss`. A rank may hold fewer evaluation batches than another, or none, whenever the
wrapper's forward is not itself a collective.

### Step statistics

A program may declare statistics in `ProgramSpec.statistics`: sums that must be pooled over the
window and the data-parallel ranks before the program can use them. The program returns its local
sums in `ObjectiveResult`, the runner reduces them, and the pooled sums come back unchanged, as
values. Any arithmetic on them, such as a ratio and its zero guard, is the program's. No process
group reaches the program.

| Delivery | The pooled sums come back | Today |
|---|---|---|
| `replay` | through `ctx.replay_values`, to a second pass over the same window. The frame calls `prepass` for each micro-batch and keeps a host copy of its inputs, pools the sums with `reduce_sum`, then calls `forward_backward` for each kept micro-batch, in the same order, with the values. Prepass and replay are paired by position in the window: the runner keeps the RNG state each `prepass` started from and restores it before the replayed forward at the same position | the DSpine window statistic (`WindowStatistic`): `CanonicalTrainStrategy.forward_loss` makes the host copy and forms the ratio, and `TrainerCore._prepare_and_replay` buffers the window |
| `after_step` | through `program.apply_statistic(name, value)`, which the runner calls on every rank inside `apply_step`, after the optimizer update | nothing on `main`; the intended use is updating a draft buffer from pooled counts |

`apply_statistic` issues no collective and computes the same result on every rank, so the draft
stays identical across ranks. A buffer it writes belongs to the draft: it is in the artifact and in
the resume state, and `draft_state()` sees it because the call sits inside `apply_step`.

Not covered: a loss term that needs a value pooled from the same forward before that forward's
backward. It can be written as `replay`, at the cost of a second forward. A cheaper form, finishing
a deferred term between forward and backward, would be a new member of the contract and is added
with the first objective on `main` that needs it (see [API rules](#api-rules)).

### Two axes of communication

A rank belongs to one data-parallel group and, when a job declares them, to model-parallel groups.

| Axis | Ranks hold | Communication | Issued by |
|---|---|---|---|
| Data-parallel | different samples | gradient synchronisation, loss denominators, metrics, step statistics, agreement, checkpoint collectives | the runner, and nothing else in the training plane. The data plane keeps its acknowledgement collectives on its own groups |
| Model-parallel (tensor, sequence, expert) | parts of one sample, or parts of one parameter | inside a component's forward: the partial sums of a tensor-parallel layer, the key and value exchange of ring attention, the all-to-all of Ulysses attention, the token dispatch of expert parallelism | the component, on a group the runner passed to `build` |

Model-parallel communication is part of a component's math, so it stays inside the program. It is
legal under four conditions:

- **Declared.** `ProgramSpec.axes` names every axis the program communicates on, and `Topology`
  gives its size and layout. An axis larger than one that the program does not declare, or a
  declared axis the runner cannot provide, is refused in `supports()`.
- **Bound, not fetched.** The group arrives through `ProgramFactory.build(groups=...)`, which has a
  key for every declared axis, a group of one rank included. A missing key is an error: in
  `torch.distributed` an absent group silently means the whole world. A component never reads a
  process-global and never finds a group in an attribute set after construction.
- **Counted once.** `ProgramSpec.axes` gives each axis one of three kinds, and the runner's
  reduction and the data plane's dispatch follow from it:

  | Kind | Ranks of a group | A rank's returned sums | Example |
  |---|---|---|---|
  | `copy` | receive the same batch | are the whole; count one rank | tensor parallelism |
  | `part` | receive their own shard of the same samples, cut by the data plane | are a shard; add them | sequence parallelism |
  | `own` | receive different batches | are their own samples; add them | expert parallelism, where only the parameters are split |

- **Same sequence.** Every rank of a group runs the same forwards in the same order.

A kind says how to count a rank's sums. It does not say where each quantity is handled, and for a
`part` axis four things happen in four places. Sequence parallelism, the one axis canonical
training has today, shows them:

| Quantity | Handled over | By |
|---|---|---|
| The sequence | cut into one shard per rank, with lookahead overlap | the data plane |
| Per-sequence loss statistics, for the LK losses | the sequence-parallel group | the program, inside the forward |
| Gradients | every rank of the world | the runner's wrapper |
| The accumulation window | whole sequences: the configured count times the axis size | the composition root, in `RunSpec` |

Sequence parallelism uses three groups today: ring, Ulysses, and their product for statistics and
dispatch. `groups` carries each under its own key, and `Topology` holds the ring and Ulysses sizes.
Of the four rows, the runner's own are the gradient group and honouring the accumulation count it
is given; a job on this axis needs all four in place. [Conformance](#conformance) names the case
that checks them.

A reduction over a model-parallel group that completes one sample's value, such as a per-sequence
statistic whose sequence is split across ranks, belongs to the component. A reduction across
samples never does.

Today's sequence-parallel code meets the counting rules and not the binding rule: the groups are
attached to the model as attributes read from process-global getters, together with a callable that
performs the per-sequence reduction (`training/sequence_parallel.py`). Passing them through
`build(groups=...)` is part of step 3. Which further axes are in scope is
[decision 10](#decisions-for-maintainers).

### Rank symmetry

Every method runs on every rank in the same order; `evaluate` is the one exception, in its row.
Collectives and failure handling:

| Method | Collective | A rank-local failure |
|---|---|---|
| `supports` | no; pure, and the verdicts are compared across ranks at startup together with the resolved values | a disagreement stops the run before `prepare` |
| `prepare` | yes; it ends with one `agree` on the realised build-time kernels | a build failure raises on that rank and the launcher tears the world down; a kernel difference raises on every rank through the closing `agree` |
| `forward_backward` | yes in every multi-rank run: gradient synchronisation and the reductions; every rank receives the same number of batches | raises on that rank and the launcher tears the world down; a failure inside a step is not agreed, as today |
| `prepass` | yes under sharding or with a model-parallel axis; every rank receives the same number of batches | as for `forward_backward` |
| `evaluate` | only when the wrapper's forward is one (sharding, or a model-parallel axis); otherwise ranks may hold different numbers of evaluation batches, or none | as for `forward_backward` |
| `reduce_sum`, `reduce_max`, `apply_step` | yes | a check on a reduced value reaches the same verdict on every rank, so all ranks raise together; any other failure raises on that rank and the launcher tears the world down |
| `draft_state`, `save_resume`, `load_resume` | yes | reported through `agree`, never raised between two collectives |
| `agree` | yes: one gather of an error string and an optional value per rank | raises on every rank if any rank reported an error or the values differ |
| `close` | no | — |

`draft_state()` is legal only between `apply_step` and the next `forward_backward`.

**`agree` is never called inside a step.** It runs at the end of `prepare`, in the checkpoint
sequence and after `load_resume`; today `CheckpointManager._all_ok` is the call in the checkpoint
sequence, and `read_resume_state` gathers once on resume. A step adds no gather and no host read:
its checks (the loss denominator, a finite gradient norm) use values that are already reduced, in
the single host read `BF16Optimizer` makes today.

### What each part owns

| Part | Owns | Must not know |
|---|---|---|
| Composition root (`application`) | resolving the specs and the [resolved values](#resolved-values) once; comparing them across ranks; choosing the runner | anything a runner does after `prepare` |
| Program | supervision, the draft forward, loss terms, the declared reduction and statistics; model-parallel communication inside its components | the data-parallel size, the optimizer, how it is wrapped |
| Step frame | walking the window one micro-batch at a time and marking its boundary; for a `replay` statistic, buffering the window's inputs and ordering prepass, pooling and replay; pooling evaluation results through the runner | gradients, process groups, the optimizer, RNG |
| Step runner | wrapping and placing the program, passing it its model-parallel groups, and deciding when `factory.load` runs; one micro-batch's forward and backward per call; every collective across samples, the training metrics included; normalisation; clipping; the optimizer and LR schedule; `after_step` statistics; RNG, with the restore for a replay; its resume state | the loss math; where data comes from |
| Run loop | the horizon; windows and the incomplete tail; repositioning the stream on resume; logging and checkpoint cadence; evaluation data, cadence and best-checkpoint tracking; the checkpoint sequence; sample commits | `torch.distributed`; model internals |

`OptimSpec` is a frozen value with today's fields (learning rate, schedule and warmup, total steps,
clip norm, master-weight placement). A runner honours it or refuses in `supports()`. It grows a
field when a second optimizer needs one. Where master copies are placed is in `OptimSpec`; their
dtype is in `PrecisionSpec`.

Weights are handed over as a plan, not as tensors. `factory.load(program)` does what `load_draft`,
`initialize_draft` and `build_bundle` do today after construction. Through `WeightPlan`
(`checkpoint/weight_plan.py`) it copies the draft's weights under canonical keys and installs the
bound target tables, which are deliberately absent from the draft's state dict. It then binds the
teacher head, which is either the draft's own LM head or a separate `BoundLMHead` that is loaded
today by a direct full read outside the plan. It ends with `finalize_initialization`, which may
derive draft tensors from a bound table (DSpine's transfer codes) and so has to come after the
bindings. The factory owns what all of this means and the runner owns when it runs. Today's DDP /
FSDP path loads before wrapping, which `apply_weight_plan` requires of an FSDP1 model, and keeps the
bound modules out of the wrapper so the bindings survive. An engine that shards first calls `load`
after materialising: the plan already writes rank-local slices into sharded and replicated
placements and refuses meta tensors, so no layout passes from the runner to the factory. Whatever a
runner does to storage after `load` has to keep the bindings, or the first forward finds an unbound
head. The separate teacher head is outside the plan's rank-local path; it is a frozen table and
never sharded (statement 5), so a full read stays correct. The plan copies into the tensors the
draft's `state_dict` returns, so each canonical entry has to be a writable view of the stored
parameter. A component that rearranges data for its canonical form would return a temporary copy,
and the load would silently change nothing; such a layout needs an explicit import, and the plan has
to refuse it until it has one. `load` and `draft_state` both speak canonical keys. Random
initialization happens at construction; an engine that constructs on a meta device has to run it
after materialising, and the prototype in step 2 has to show that.

### Checkpoints

A checkpoint has three parts and one commit point.

| Part | Content | Written by | Read by |
|---|---|---|---|
| Resume state | everything needed to continue: every draft parameter and buffer as the runner stores them, optimizer and scheduler state, RNG | `runner.save_resume`, every rank | the same runner under the same `Topology` |
| Artifact | the canonical draft directory, under canonical keys: a view derived from the resume state at the draft's declared dtype | rank 0, from `runner.draft_state()` | export, SGLang conversion, `initialization.checkpoint`, the conformance kit |
| Run record | step, epoch, samples consumed, the requested `Kernels`, the runner's `ResumeRecord` with the realised kernels, `PrecisionSpec` and `Topology`, the per-key resume contract and model provenance compared today, without the kernel keys, which move into the record | rank 0, **last** | the run loop on resume |

Sequence, on every rank: flush sample commits up to this step → `draft_state()` → rank 0 writes the
artifact → `save_resume()`, which returns this rank's record → `agree(errors, value=record)` → rank
0 writes the run record atomically, with the record inside → `agree` again, so every rank sees a
failed write → rank 0 moves the `latest` link. A checkpoint exists if and only if its run record
exists. A checkpoint is never ahead of its durable commit, as today
(`TrainerController.save_checkpoint`), and online resume keeps requiring the ledger marker to equal
the checkpoint step. The run loop never reaches into the program: the runner reads the realised
kernels and returns them in its record, and the first `agree` is where a difference between ranks
raises, as `ExecutionSelection.agree` does from `CanonicalTrainStrategy.before_checkpoint` today. A
rank-local failure of `draft_state` or `save_resume` is caught and passed to that `agree` as its
error. Today's save agrees once and then moves the link (`CheckpointManager.save`); the second
agreement is new.

The resume state is the truth for continuing; the artifact is derived. The step counter has one
owner, the run loop; a runner whose engine keeps its own counter takes it from the run record.
Resuming with a different runner, `Topology` or `PrecisionSpec`, or with an explicit kernel request
that contradicts the record, is refused, naming both values. An unset request adopts the recorded
request, as `adopt_recorded_execution` does today; the runner receives the record in `load_resume`
and owns both checks: the build-time kernels it realised in `prepare` against the record at once,
and the loss kernels when real inputs arrive (`program.expect_kernels`). A mismatch raises, as the
per-key comparison in `Trainer` does today. Starting from the artifact always works and restarts the
optimizer.

### Numerics classes

`supports()` returns the class for this job, not a constant of the runner: the kernels a runner
executes depend on configuration.

- **`exact`**: bit-identical to the stored trajectory of the reference runner **for the same
  topology** (same runner family, `Topology`, realised kernels, `PrecisionSpec`, device type and
  torch build). Bit-equality across
  wrappers or world sizes is not defined: DDP and FSDP divide gradients at different points, and a
  single process cannot reproduce several ranks drawing from the same seed.
- **`equivalent`**: a different numerical program for the same objective. It is a numerics change.
  It is held to two checks, both set from the reference and never by the candidate:
  1. *first update*: from the same weights and the same recorded window, the pre-clip gradient norm
     and the parameter update agree with the reference within a stated relative tolerance;
  2. *end to end*: the comparison in [Conformance](#conformance), with a band taken from repeats of
     the reference.
  An `equivalent` runner is never selected by default until both exist for its class of jobs.

The choice is made once at startup, logged, compared across ranks and carried in every checkpoint's
run record. Selecting a runner whose dependency is missing is an error; it never falls back to
another runner.

**The reference runner is a role.** The DDP / FSDP runner holds it today. It moves to another
runner only by a maintainer decision, after that runner has passed both `equivalent` checks and the
serving metric for the recipes concerned. The stored trajectories are then re-recorded under the new
reference runner, and `exact` is measured against them from that point. The "reference trainer"
below is a different thing: the repository's own run loop and frame, over whichever runner is
selected.

## The reference trainer

| Today | Becomes |
|---|---|
| `TrainerController` | the run loop |
| `TrainerCore` and `CanonicalTrainStrategy` | the step frame; `train_step` already computes one micro-batch per call |
| `FSDPTrainingBackend`, with `BF16Optimizer` behind it | the DDP / FSDP runner; the reference |
| (none) | the plain runner: one process, no wrapper. New code; it takes the reference-runner role for CPU only after it is shown bit-identical to the DDP / FSDP runner at one rank |
| `build_bundle` | `ProgramFactory.build` and `load`; the artifact keeps its writer in `specforge.checkpoint` |
| attributes set on the model after construction | declarations on `ProgramSpec` and the fields of `PrecisionSpec` |
| getters of `specforge.distributed` read by layers, data utilities, the trainer and `sequence_parallel.py` | `Topology` as a value; groups passed by the runner to `build` |
| `microbatch_mean` and the DSpine replay calling `torch.distributed` | the declared reductions and the frame's prepass |
| `CheckpointManager._all_ok`, and `ExecutionSelection.agree` called from `before_checkpoint` | `runner.agree`, with the runner's `ResumeRecord` as its value |

What the current backend provides implicitly, and a new runner must provide or refuse: `no_sync`
around forward and backward for DDP, backward only for FSDP; the loss-denominator check folded into
the optimizer's single host read; frozen target modules kept out of sharding and out of the state
dict; per-rank RNG and FSDP optimizer state; gradient reduction in the parameter dtype, and fp32
buffers under sharding.

## Attaching torchtitan

At the step-runner level, by wrapping its `TrainingEngine`, as
[miles](https://github.com/radixark/miles) does (`miles/backends/torchtitan_utils/`). This section
is a list of what an adapter has to solve, not a mapping that is known to work. Read from
torchtitan `main` at `a1f6dcd2`; miles pins the `v0.4.0` release, where several names differ.

**Choose first: the torchtitan revision and the torch build.** SpecForge pins `torch==2.14.1`.
miles runs torchtitan `v0.4.0` on torch 2.14 by patching `torch.distributed.fsdp` at import time,
and notes that the release targets torch 2.15. A comparison across torch builds is confounded, so
the reference has to be re-recorded on whichever build is chosen.

| Contract | torchtitan | What the adapter has to solve |
|---|---|---|
| `prepare` | `TrainingEngine.initialize`: `model_config.build()` on the meta device → `model.parallelize(...)` → `to_empty` → `init_weights` → optimizer → checkpointer | Every stateful child must implement torchtitan's `Module` protocol; `_parallelize` raises on a plain `nn.Module` that owns parameters or buffers. The adapter has to meet that protocol for any component generically; a second implementation per component is not part of this design unless decision 11 allows it. Parameters are sharded before `init_weights`; `factory.load` then writes rank-local slices, which `apply_weight_plan` already does for sharded and replicated placements. torchtitan's own route for initial weights is its checkpointer with a state-dict adapter. `Config` must provide `max_context_length`, `get_parameter_counts` and `build_flops_estimator` |
| Process model | the constructor binds the device, applies NUMA binding, seeds RNG and sets group timeouts | make each a no-op or the single owner |
| `forward_backward` | `engine.forward_backward` expects its micro-batch type, `global_loss_token_counts` before the forward, and `preprocess_inputs`; accelerator-graph capture is on by default and forbids host syncs and varying shapes | the contract calls the runner once per micro-batch and the denominator is unknown before the forward, so the adapter drives `model_parts` and the loss itself, as miles does with a stateful loss object. The engine's own whole-window `forward_backward` is not usable here. The route depends on engine internals |
| Reduction | stock FSDP2 divides; torchtitan turns the division off in each model's own FSDP setup (`apply_fsdp_to_decoder`, and the flux and Transformers-backend equivalents), not in the engine; deferring the reduction to the last micro-batch is opt-in | the adapter's own `_apply_fsdp` decides, and `apply_step` scales accordingly |
| `apply_step` | `engine.optim_step()`: clip, finite check by `torch._assert_async`, step, scheduler | scale the reduced gradients before it |
| `save_resume`, `load_resume` | its checkpointer saves on its own interval, model-only at the last step by default, rotates, and loads the newest step in its folder. It saves no RNG | drive it explicitly; add RNG; reconcile its step counter with the run record |
| `draft_state` | no engine call gathers a full state | gather; map names to canonical keys; cast to the artifact dtype |
| Precision | fp32 parameters with bf16 compute; under replication through autocast, under sharding through the FSDP cast; frozen parameters stored in bf16 | it cannot honour today's `PrecisionSpec` (bf16 parameters, fp32 masters) and refuses it; a job selects fp32 parameters with bf16 compute explicitly. That selection is a numerics change, which is why the runner is `equivalent` for today's recipes |
| Step statistics | an engine may update state of its own around the optimizer step | one owner per statistic: the adapter turns the engine's update off, or the runner refuses the program |
| `close` | `engine.close()` | call it |

miles's adapter is about 1,000 lines for the core and, beyond the table, needed shape buckets with
compile warm-up and a custom checkpoint manager.

## A dedicated trainer

A second level, `Trainer.fit(job)`, lets a trainer replace the run loop, the frame and the runner
together. Its types (`Trainer`, `TrainJob`) are **not** specified here: they are introduced with the
first trainer that needs them, so that the abstraction is drawn from two working implementations.

What is fixed now, because it comes from the other planes:

- It consumes the same per-rank stream of `TrainBatch` and never executes target layers.
- It runs inside the process model above.
- **It keeps the commit contract.** A sample is committed only after the update that consumed it
  has been applied on every rank; commits are issued by every rank in the same order; an incomplete
  window is never committed; commits are flushed before a checkpoint. Features may be deleted at
  commit, and each rank receives exactly its dispatched references per window.
- It produces the same three checkpoint parts with the same commit point.

## Conformance

- **Numerics, by replay.** Recorded feature streams in the offline feature format: small ones for
  CPU, one at recipe scale. Today's fixtures hold losses and gradient norms per objective
  (`tests/fixtures/objective_goldens.json`) and an end-to-end resume test; they do not yet hold
  parameter updates, optimizer and RNG state or the data position. Features do not depend on the
  draft, so one recording serves every runner. An `exact` runner reproduces the stored trajectory
  for its topology (loss sequence, final weights, optimizer state). An `equivalent` one passes the
  first-update check and the end-to-end band.
- **Protocol, by a fake stream.** A consume-once stream with a recording ledger checks what a
  recording cannot: commit order across ranks, no commit for an incomplete window, commits flushed
  before a checkpoint, the run record written last. It also pins the window: three batches per
  epoch, two epochs and an accumulation of two must give three updates.
- **The kit must be able to fail.** Three mutations of the reference are part of it and each must be
  caught: gradients scaled by the world size, clipping skipped, the LR schedule shifted by one step.
- **One case per declaration, not per recipe.** A runner is exercised with one job for each thing it
  claims in `supports()`: each reduction, each statistic delivery, each model-parallel axis. A
  recipe is a combination of these, so new drafts and losses need no new runner tests.
- **Sequence parallelism is in the first version.** An EAGLE3 job with data-parallel and
  sequence-parallel sizes above one, accumulation above one and an LK loss, so that the per-sequence
  reduction runs, must reproduce its stored trajectory: loss, gradients and the step at which the
  optimizer moves. It needs `flash_attn`, so it is a GPU recording.
- **Evaluation is pinned separately.** A two-rank case with unequal denominators and one with an
  empty evaluation shard must reproduce today's `eval/avg_loss`.
- **End metric.** The repository's evaluator reports a teacher-forced proxy for acceptance length.
  Serving acceptance length, measured through SGLang, is the metric for a default change.

## API rules

| Rule | Enforced by |
|---|---|
| Parts meet only through `training/contracts.py`, `runtime/contracts.py` and `config.draft` | an import scan |
| Values cross, handles do not: no `getattr` on another part's object, no callable in a context, no process-global topology, no environment read below `application`. One handle crosses, in one direction: the process group of a declared model-parallel axis, from the runner to `build` | a lint with an allowlist that may only shrink |
| In the training plane, a collective across samples appears only in a runner | the import scan: a `torch.distributed` collective in program, frame or run-loop code needs an allowlist entry naming its model-parallel axis |
| A runner decides from declarations, never from an architecture or objective name | the import scan: a backend package imports neither the model registry nor the objective registry |
| Honour or refuse: `supports()` names the reason; verdicts are compared across ranks | a test per refusal; a per-field mutation test on the first update for each `OptimSpec` field a runner claims |
| A step cannot be applied twice or without a backward | `PendingStep` is single-use |
| A backend is one package under `training/backends/`, registered by name; core never imports it; a missing dependency is an error | the import scan; the CPU suite without the dependency |
| A contract gains a member only with a second implementation that needs it | review |

## Delivery

Prerequisite, met on `main` at `d4391211`: one step path. The legacy strategies are gone, and
`CanonicalTrainStrategy` over `FSDPTrainingBackend` is the only route.

| # | Change | Kind | Gate |
|---|---|---|---|
| 0 | Measure where step time goes on one recipe per family: data wait, forward, loss, backward, optimizer | measurement | a table in this note; it settles the throughput argument for the remaining steps, and decision 9 settles the coverage argument |
| 1 | Record the reference: trajectories of the DDP / FSDP path through the CLI at every world size the bundled recipes use, holding parameter updates, optimizer and RNG state and the data position as well as losses, with the sequence-parallel case included | tests | a second recording reproduces the first bitwise |
| 2 | A prototype: one canonical draft under torchtitan at the chosen revision and torch build, through initialization, one update, export and resume | measurement | a go or no-go with the adapter's size and throughput, whether one generic adapter meets its `Module` protocol, a load of asymmetric weights followed by a forward, and what the prototype makes the contract change |
| 3 | The minimal runner: `run_window` as the frame and `StepRunner` over today's backend; `ProgramSpec` derived from the objective registry; declarations replace attributes set after construction; `factory.load`; groups passed to `build` | behaviour-preserving | stored trajectories, bitwise, per topology |
| 4 | `Kernels`, `PrecisionSpec` and `Topology` resolved once and compared across ranks; kernel agreement through `runner.agree`; `application` as the only composition root for training | behaviour-preserving | stored trajectories; the import scan |
| 5 | Conformance kit: replay, fake stream, mutations | tests | the three mutations fail |
| 6 | torchtitan runner | new, `equivalent` | the kit; serving acceptance length; throughput and memory |
| 7 | Plain runner | new | bit-identical to the DDP / FSDP runner at one rank on CPU, or the difference explained |
| 8 | Checkpoint parts and commit point | on-disk change | a census of every reader of `training_state.pt`; a reader for existing run directories |
| 9 | `Trainer` and `TrainJob`, with the first dedicated trainer | new | the kit; serving acceptance length |

Order: 0, 1 and 2 come first, in any order, and the contract is fixed only after 2. Then
3 → 4 → 5 → 6. Steps 7 and 8 need only step 3 and can be scheduled separately; step 9 needs the
kit (5). The kit's check that the run record is written last arrives with step 8. Decision 6 has
to be taken before a second runner (step 6 or 7) can be selected.

The first change a user can see is step 8. Step 3 changes the shape of the public
`specforge.training.Trainer` only if maintainers choose so.

## Decisions for maintainers

| # | Decision | Why it is not made here |
|---|---|---|
| 1 | Whether the measured step cost justifies this work, and for which recipes | step 0 has not been run |
| 2 | The torchtitan revision and the torch build | a dependency and numerics choice; SpecForge and torchtitan target different torch versions |
| 3 | Accepting torchtitan as `equivalent`, the first-update tolerance, and who sets the end-to-end band | a numerics trade |
| 4 | The artifact's dtype rule: written at the draft's declared dtype, cast by the same rounding the reference uses when it copies masters to parameters | a precision decision; a runner with fp32 parameters would otherwise produce a different artifact |
| 5 | The run-directory layout in step 8, and whether existing directories get a reader or a documented break | a user-visible on-disk contract |
| 6 | The config field that selects a runner, its default, and its relation to `training.fsdp_sharding` | a user-visible key |
| 7 | Whether `specforge.training.Trainer` keeps its public shape | a public API |
| 8 | Whether a dedicated trainer is in scope now | no implementation exists |
| 9 | Whether an external engine is meant to become the default runner and take the reference role, with the DDP / FSDP runner retired | a strategy choice. It decides whether coverage alone justifies steps 2 and 6, whether the plain runner (step 7) is worth building, and how early decision 2 has to be taken |
| 10 | Which model-parallel axes canonical training supports beyond sequence parallelism for the objectives that declare it | each further axis needs a declaration, a bound group, a data-plane rule and a regression case of its own |
| 11 | Whether a backend may carry its own implementation of a component when no generic adapter satisfies its engine | it multiplies implementations by backends, and each one needs its own parity evidence; step 2 supplies the facts |

## Prior art

- miles: one step frame over a four-method runner
  (`miles/backends/training_utils/torch_native/`), and torchtitan `v0.4.0` attached by wrapping its
  `TrainingEngine` (`miles/backends/torchtitan_utils/`). Taken from it: the optimizer behind the
  runner, one adapter package per backend, refusals asserted at init. Not taken: a loss defined on
  full logits, a normaliser computed before the backward, process-global parallel state, and a
  backend that owns the model definition.
- torchtitan: `trainer.py` (the loop) over `training_engine.py` (model build, parallelization,
  optimizer, checkpoint, forward-backward).
