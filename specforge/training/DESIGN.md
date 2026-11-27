# Training Plane Design (`specforge.training`)

This is the design note for the **training**, scoped to this plane.
The cross-plane picture (whole-system map, endpoint reference, autonomy) lives in
[`../runtime/ARCHITECTURE.md`](../runtime/ARCHITECTURE.md); the shared records
every plane exchanges are in
[`../runtime/contracts.py`](../runtime/contracts.py).
The backend architecture (program, shared step layer, parallel runtime and
update registries) is in [`BACKENDS.md`](BACKENDS.md); this note describes what
runs on `main` today.

## Responsibility

Owns the one caller-facing `Trainer` lifecycle that turns a normalized,
tensor-carrying `TrainBatch` stream into optimizer steps and checkpoints.
`Trainer.fit()` is the only package-level training call: it enters any
topology-owned stream context, invokes the internal `TrainerController` over its
private loader, exits the context, and writes the final checkpoint when the
last optimizer step was not already saved by the periodic checkpoint path.
For the disaggregated online consumer, that same lifecycle then publishes its
terminal done/failed signal and always stops the rank-0 ref distributor. Direct
Python builder callers therefore receive the same cleanup guarantees as the
CLI; there is no second wrapper-owned training lifecycle.

Below that boundary, `TrainerController` is the run loop: it owns the epochs,
the optimizer windows it cuts from the training stream, optimizer-step
counting, interval checkpoints, and durable acknowledgements. The step frame
(`frame.run_window`) walks one window and decides only the order in which the
runner sees its micro-batches. `DataParallelRunner` executes each micro-batch
and applies the update: it issues the step's collectives across samples (the
loss denominator, the window's metrics, the replay statistics), normalizes the
gradients and owns the RNG. That includes the `microbatch` reduction (DSpark,
whose protocol batches are `global_microbatch`): the objective returns its loss
terms and the runner pools each term's denominator over the ranks between the
forward and the backward. No objective issues a collective across samples. The
program the runner calls
is the objective model itself (`algorithms.objectives.base.ObjectiveModel`): it
owns the forward and the loss, the target projection, the host-side inputs it
derives from a batch, and the mapping from a training state dict to the draft's
own; `training/program.py` reads what it declares (`ProgramSpec`) from its
registration. `FSDPTrainingBackend` owns wrapping, backward, optimizer steps,
distributed gradient norms, and full training state. What the frame, a runner
and a program exchange is declared in `contracts.py` (`StepRunner`,
`PendingStep`, `StepMetrics`, `StepContext`, `ProgramSpec`,
`TrainingProgram`).
Checkpoint rotation and the latest pointer live in
`specforge.training.checkpoint`. Resume restores each rank's optimizer/RNG
state and repositions fixed offline refs through `FeatureDataLoader.seek()`.

### Boundaries

The parts meet through `contracts.py`. `tests/test_runtime/test_training_boundaries.py`
reads six import rules from the source:

| Rule | Modules | May not reach |
|---|---|---|
| A collective across samples appears only in a runner | objectives, drafts, losses, layers; the frame; the run loop and the evaluator; the checkpoint manager, the profiler and `trainer.py` | `torch.distributed`, `specforge.distributed` |
| The frame imports the shared contracts only | `contracts.py`, `frame.py`, `program.py` | anything else in `specforge` |
| The run loop imports no runner, backend, optimizer, objective or model | `controller.py`, `specforge/eval/` | `training.runner`, `training.backend`, `training.backends`, `optimizer`, `algorithms`, `modeling`, `layers`, `losses`, `torch.optim` |
| A runner decides from declarations and imports no registry | `runner.py`, `backend.py`, `backends/`, `optimizer.py` | `algorithms`, `modeling`, `config.draft.registry` |
| A program imports nothing that trains it | objectives, drafts, losses, layers | `training`, `runtime`, `launch`, `eval`, `optimizer`, `torch.optim`, `torch.nn.parallel` |
| Only the trainer imports a runtime package | everything in `specforge` outside `training/backends/` and `trainer.py` | `specforge.training.backends` |

The shared contracts are `training.contracts`, `runtime.contracts` and the
value types of `config.draft`, which the test lists by name (`DraftConfig`):
another name the package exports is followed to where it comes from.

A module reaches a name by importing it, at module level or inside a function;
by using the imported name, through its alias and with the attributes that
follow (`import torch as t; t.distributed.barrier()`); by a literal
`importlib.import_module`; or through the package: a name another module
re-exports is followed to its origin with the rest of the chain, and a helper
function brings what its own body uses. Names are resolved in the scope they
are used in: a comprehension is a scope of its own, a function's defaults,
decorators and annotations belong to the scope around it, and a class body's
names are not in scope for its methods.

The scan follows names. It does not see which process group a collective runs
on, what a method called on a value does (the run loop holds a runner and the
scan does not stop it from reaching through it), a decision taken from
`spec.name` in place of a declaration, or a name computed at run time.

Every module that breaks a rule today is listed in the test with the reason
and with each forbidden name it reaches. A name that communicates or reads
the process-global topology is listed with how many times; one use more or
one fewer fails. A type, a constant or a query of the process's own rank or
of a group's size is listed by name only. Two are the sequence-parallel axis
(the rollout, ring attention), whose groups share a sequence and never
samples. Three are the run loop's own uses of the process group
(`controller.py`, `checkpoint.py`, `profiling.py`). `controller.py` and
`trainer.py` take ranks and the world size from the job's `Topology`; the
design moves the remaining reads to that value and the error gathers to
`runner.agree`; rank 0's two broadcast verdicts (whether an evaluation runs,
whether a checkpoint is the best) need either a way to distribute a decision
or every rank computing it, which the contract does not have yet.

### Conformance

`tests/test_runtime/conformance.py` is what a step runner is held to:

- `declaration_cases()`: one recorded training case for each thing a program
  declares to a runner (each reduction, a replay statistic, evaluation
  diagnostics), at two ranks. They train the runner `specforge train`
  assembles, which is the reference: no configuration key selects another.
  A candidate is recorded by substituting it around every rank's run
  (`record(cases, around_training=...)`), and a runner of the `exact` class
  then reproduces the reference's recording on the same machine.
- `MUTATIONS`: three broken references (gradients scaled by the world size,
  clipping skipped, the learning-rate schedule shifted by one step). Each has
  to change the recording, or the replay proves nothing.
- `ProtocolChecks`: the run loop over the runner with a stream that is read
  once and a ledger of what was read, handed to the runner, applied,
  committed, snapshotted and checkpointed. The runner is handed the stream's
  batches, each once and in order; an update consumes one whole window; a
  commit names every sample its step's update consumed and follows it;
  nothing is committed for an incomplete window; a checkpoint's state is
  collected and written with every update committed and no acknowledgement
  in flight; a complete run does not raise.
- `RunnerChecks`: one job per declaration in one process, compared with the
  reference runner bit for bit: every update's metrics, gradient norm,
  learning rate and weights, and the exported draft and the generator at the
  end. The optimizer state is not compared across runners: what a runner
  resumes from is its own to lay out, and the resume job holds it to the
  runner's own uninterrupted run. The program's loss reads the step, the
  run's length and the noise it draws, detailed metrics are asked for on
  every second update, its replay statistic depends on the mode and the
  generator of the prepass, and one job runs in bfloat16, where the
  optimizer's fp32 masters hold more than the weights. Plus the single-use
  handle, a refused batch, evaluation, and resume in bfloat16 through two
  further updates.

A new runner adds two test classes that set `make_runner`; those hold it in
one process, on the host. Not in the kit: a recorded case with sequence
parallelism (it needs `flash_attn` and accelerators); the order of commits
across ranks; a prepass in the protocol's ledger; evaluation with unequal
denominators or an empty shard across ranks; frozen modules and partition
blocks, which only a sharding runner acts on.

## Internal mechanics

```mermaid
flowchart TD
  classDef compute fill:#e6f6ea,stroke:#3bb061,color:#0b4a22;
  classDef control fill:#e8f0fe,stroke:#3b6fd6,color:#0b2e6b;

  PUB[Trainer fit public lifecycle] -->|private loader| FIT[TrainerController run loop]
  PUB -->|final unsaved step| CKPT[save_checkpoint collective]
  FIT -->|one window of batches| FR[frame run_window]
  FIT -->|append sample_ids| PA[pending_ack accumulator]
  FR -->|forward_backward per micro-batch| RN[DataParallelRunner]
  FR -->|apply_step at the boundary| RN
  RN -->|program call| ST[ObjectiveModel projection plus loss]
  RN -->|backward and optimizer step| BK[FSDPTrainingBackend]
  FR -->|StepMetrics| FIT
  FIT -->|global_step plus 1| BND[boundary actions]
  BND -->|ack_fn ids step| ACK[ack_train_refs durable]
  BND -->|save_interval| CKPT
  CKPT -->|checkpoint_state and agree| RN

  class PUB,FIT,FR,RN,ST,BK,BND,PA,CKPT compute;
  class ACK control;
```

The training plane is the only tensor-carrying side besides the data plane; it
consumes `TrainBatch.tensors`. One optimizer update consumes one window: at
most `accumulation_steps` micro-batches, cut from the stream across epoch
boundaries and handed to the runner one at a time, so no window is held in
memory. `DataParallelRunner.forward_backward` divides each loss numerator by
`accumulation_steps` and uses `no_sync()` only for non-boundary micro-batches;
its boundary call returns the `PendingStep` that `apply_step` consumes, so an
update cannot be applied twice or without its backward. The update's objective
is `sum(numerator) / sum(denominator)` over the window and the ranks.
For DDP, the backend accumulation context encloses both forward and backward;
wrapping backward alone does not disable DDP gradient synchronization. FSDP
retains its backward-only deferral.
`TrainerController` increments `global_step` only when a window's update has
been applied, commits the pending sample acknowledgements, emits metrics, and
performs configured interval saves. The outer `Trainer` owns topology cleanup and the guarded final
save, so CLI, builders, and Python callers cannot select a second loader-based
training entry. For a save the run loop asks the runner for its
`checkpoint_state()`: the draft's weights under canonical keys (complete on
rank 0), this rank's own optimizer and RNG state, the optimizer state that is
the same on every rank (DDP), and the loss kernels this rank has realised. The
ranks compare those kernels through `runner.agree` before anything is written,
so what rank 0 records for resume holds for every rank; a rank that failed to
collect its state reports that through the same agreement. All saves then
delegate to `CheckpointManager`. Rank 0 writes the shared file: the draft
state and, under DDP, the one optimizer state every rank holds. Every rank
writes its own file: its RNG state and, under FSDP, its shard of the optimizer
state. On resume the trainer hands each rank's saved state back through
`runner.load_resume` and the ranks agree that it restored.

DSpine declares a replay statistic: its alignment CE is normalized over the
whole optimizer window and every rank. For such a window the frame first calls
`runner.prepass` for each micro-batch (training mode, no gradients) and keeps a
host copy of its inputs, pools the statistic's sums once with
`runner.reduce_sum`, then calls `forward_backward` for each kept micro-batch
with the pooled sums in `StepContext.replay_values`. The runner restores, for
each replayed forward, the RNG state its prepass started from. Replay adds a
forward pass without retaining the window's activations. Each replayed
forward/backward pair uses the backend accumulation context; only the last
synchronizes DDP gradients. Evaluation does not replay.

Evaluation goes through the same runner. `Evaluator.run` opens one
`runner.evaluation()` pass on every rank (evaluation mode from entry to exit,
the mode it found restored afterwards), calls `runner.evaluate` for each batch
without gradients, and pools the pass once at the end through
`runner.reduce_sum` and `runner.reduce_max`, on a schedule that is the same on
every rank. `evaluate` reduces nothing across samples, so a rank may hold fewer
evaluation batches than another, or none, unless the forward itself
communicates (a sharded wrapper, or sequence-parallel attention). An objective that normalizes each rank's micro-batch over
all ranks in training (DSpark) is normalized on its own rank in evaluation; the
evaluator pools numerators and denominators of the whole pass instead.

Natural end-of-stream is accepted only at an optimizer boundary. If the stream
ends inside a window the frame raises `IncompleteWindow` and `fit` fails
instead of stepping unreduced gradients or reporting a checkpoint as
successful. Queue-mode loaders
terminally settle and clean a short `drop_last` batch without emitting it;
fixed offline refs keep normal `drop_last` semantics.

## Consumer dispatch

`runtime.consumer_dispatch` controls sample assignment to consumer ranks:

- `round_robin` (default): assign samples in source order.
- `cost_balanced`: balance estimated anchor/context costs within each optimizer window.
- `two_window_bucketed`: sort samples across two windows, then balance each window.
- `two_window_joint_balanced`: balance two windows jointly, falling back to independent
  balancing unless the estimated total critical-path cost decreases.

Two-window policies require both high and low ref watermarks to cover two global
optimizer windows and can change optimizer-step sample membership and the training trajectory.
Producer byte accounting follows the acknowledged refs, including reordered windows;
resume rebuilds release totals from the durable ledger and publication stream.
All policies preserve sample coverage, per-rank accumulation counts, deduplication,
durable acknowledgements, and incomplete-tail handling. Cost estimates are heuristics,
not guarantees of throughput improvement.

## Endpoints

### What this plane calls into

| From | Endpoint | Plane |
|---|---|---|
| `TrainerController` | `FeatureDataLoader.__iter__` | compute |
| `TrainerController` | `frame.run_window` | compute |
| `frame.run_window` | `DataParallelRunner.forward_backward` / `apply_step` | compute |
| `TrainerController` | `Evaluator.run` | compute |
| `Evaluator` | `DataParallelRunner.evaluate` / `reduce_sum` / `reduce_max` | compute |
| `TrainerController` | `DataFlowController.ack_train_refs` | control |
| `DataParallelRunner` | `ObjectiveModel.forward` | compute |
| `DataParallelRunner` | `FSDPTrainingBackend.backward` | compute |
| `DataParallelRunner` | `FSDPTrainingBackend.step` | compute |
