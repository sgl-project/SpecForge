# TorchTitan training runtime

`training.backend=torchtitan` runs SpecForge draft objectives inside the
TorchTitan v0.3.0 `Trainer`. TorchTitan owns distributed initialization, meshes,
gradient accumulation, backward, clipping, AdamW, the scheduler and
distributed checkpoints. SpecForge supplies the draft architecture, feature
loader, objective, parallelization plan and pipeline stages.

The default `training.backend=fsdp` continues using the existing SpecForge
trainer. Installing or importing TorchTitan is unnecessary for that path.

## Environment

Use a separate Linux CUDA environment with Python 3.11/3.12, PyTorch 2.14.0 and
TorchTitan 0.3.0. SpecForge's default environment pins PyTorch 2.13.0 for its
existing stack; do not overwrite a running capture-server environment.

For an offline trainer environment:

```bash
uv venv --python 3.12 .venv-titan
uv pip install --python .venv-titan/bin/python \
  torch==2.14.0 torchvision==0.29.0 torchtitan==0.3.0 \
  transformers==5.12.1 accelerate click pydantic pyyaml safetensors \
  openai-harmony yunchang psutil
uv pip install --python .venv-titan/bin/python --no-deps -e .
```

SGLang feature capture runs separately. Offline features use the same algorithm
reader, normalizer and collator as SpecForge's FSDP runtime. Online consumers
reuse SpecForge's retained Mooncake feature stream and durable ACK ledger; the
capture producer continues using its existing runtime and environment.

## Recipe

Start with a DFlash, DFlash2 or DSpark offline recipe and add:

```yaml
training:
  backend: torchtitan
  tp_size: 2
  torchtitan:
    dp_replicate: 1
    dp_shard: -1
    cp_size: 1
    pp_size: 1
    activation_checkpoint: full
    compile: false
    disable_cuda_graphs: true
```

`dp_shard=-1` uses the remaining ranks after TP, CP, PP and replicated DP.
All degrees must multiply to `WORLD_SIZE`. `batch_size` is the batch per DP
replica; TP/CP/PP peers receive the same logical examples. `accumulation_steps`
retains its existing meaning.

Launch with the same entry point, using the Titan environment's Python:

```bash
.venv-titan/bin/python -m torch.distributed.run --standalone \
  --nproc-per-node=4 -m specforge.cli train --config run.yaml
```

## Parallelization

The model adapter uses TorchTitan's supported `partial_dtensor` backend with a
model-specific TP plan. Q/K/V and MLP input projections are column-sharded;
output projections are row-sharded. The teacher feature projector and auxiliary
heads remain replicated over TP. Frozen teacher embeddings/head are replicated.
This adapter does not implement v0.3's default `spmd_types` module protocol or
`full_dtensor`; selecting either fails explicitly.

CP divides complete sampled draft blocks among ranks, keeping the teacher
context replicated. This reduces draft query activations but does not divide
teacher-context KV memory. Blocks remain intact for DFlash2 convolution and
DSpark's Markov heads. Loss normalization uses the full logical objective,
including unequal valid counts and ranks with no valid blocks.

PP partitions decoder layers and carries both draft activations and projected
teacher context between stages. Context gradients return to the first stage's
projector. The final stage owns the output norm and auxiliary heads. Use `1F1B`
or `GPipe`, set `pp_microbatch_size` to a divisor of `batch_size`, and provide at
least one decoder layer per stage. PP pads features and anchor capacity to fixed
shapes; this padding can cost more than PP saves for small draft models.
PP with `lk_loss_type=lambda` is rejected: its nonlinear coefficient depends on
the full logical batch and cannot be recomputed independently per microbatch.

Activation checkpointing and per-block `torch.compile` use TorchTitan's
implementations. The feature projector is compiled as well. Expert parallelism,
FP8, async TP, USP and non-GQA tensor-parallel plans are unsupported.

## CUDA Graph and GraphTrainer

For offline features, the ordinary native Trainer can capture its forward and
backward with `training.torchtitan.disable_cuda_graphs=false`. Pair it with
`compile=true` to retain per-block compilation. DP, TP and block CP are
supported; PP with capture is rejected. Sequence length and anchor capacity
are padded to fixed shapes. This can waste work on short examples, so measure
with the actual length distribution. Detailed training diagnostics stay outside
this graph path; native weighted loss, gradient norm and throughput remain.

The separate experimental upstream GraphTrainer is selected explicitly:

```yaml
training:
  backend: torchtitan
  tp_size: 1
  torchtitan:
    engine: graph
    compile: true
    graph_inductor: regional
    disable_cuda_graphs: false
    activation_checkpoint: none
```

This executes TorchTitan's joint forward/loss/backward tracing, graph passes,
SimpleFSDP communication, and CUDA Graph pass. It currently supports DP only.
`graph_inductor=full` compiles the full traced graph; `regional` compiles tagged
regions such as FlexAttention and leaves the rest interpreted. GraphTrainer uses
its default selective activation memory policy and honors the configured FSDP
resharding policy. It owns activation-memory planning, so the ordinary
activation-checkpoint setting must remain `none`. Compilation and capture costs
are separate from steady training speed.

The adapter functionalizes traced Triton buffer writes before graph elimination
and retains the DFlash2 fused convolution/head. It also gives accumulated
gradients independent storage so CUDA replay cannot overwrite the previous
microbatch's gradient. Repeated reads of a SimpleFSDP parameter share one BF16
materialization through the joint forward/backward call, including objective
chunk recomputation. This matches eager FSDP2's accumulation into one unsharded
parameter instead of separately casting each use's gradient to FP32.

Both native block compilation and GraphTrainer preserve eager BF16 rounding
boundaries and division rounding. The policy covers initial tracing, compilation,
replay and evaluation; setting it only during the final Inductor pass would miss
the rounding barriers. DFlash2 uses the same fused convolution under Dynamo and
joint tracing, rather than changing to an ATen decomposition only under Dynamo.
Compiled checkpoints record the numerical policy revision and reject resumes
from the earlier policy. Pure eager native checkpoints retain their old contract.

These corrections do not make different compilation boundaries bitwise equivalent.
FP32 reductions inside RMSNorm can round differently before the BF16 output cast;
block compilation can also group shared-context gradients differently from a
joint backward graph. In the focused DFlash2 FP32 eager/raw-joint comparison,
first-step gradient relative L2 error was below `5e-8`, whereas BF16 amplified
the accumulation difference. Full compilation additionally uses nondeterministic
indexed reductions for candidate-selector gradients: independent fresh runs can
differ even with identical seeds. Regional compilation passed exact checkpoint
continuation and evaluation-isolation checks in the corrected tiny DP2 fixture;
default full compilation did not pass that bitwise check. Repeating the full
test with native TorchTitan deterministic debug mode restored exact agreement,
isolating this difference from checkpoint or evaluation-state corruption.

Treat GraphTrainer as experimental and validate convergence before adopting it
for a training run. The corrected full-graph DFlash2 synthetic trajectory still
does not meet a strict native-Trainer loss-agreement gate, so its timing results
must remain diagnostic rather than evidence of equivalent training quality.

## Evaluation and online features

Set `data.eval_hidden_states_path` and `training.eval_interval` together to run
periodic offline evaluation from the native training loop. Evaluation visits
each example once; tail batches are padded with zero-weight rows so every rank
executes the same number of forwards. It reduces objective numerators and
denominators over DP/CP, without counting TP replicas or normalizing the weighted
objective again by token count. It preserves training RNG, data cursor, gradients
and model mode. PP evaluation is rejected; the sequence-walk acceptance metric
is omitted under CP because separate local walks cannot be summed correctly.

Online training uses the existing disaggregated Mooncake recipe and the same
producer/consumer commands, with `training.backend=torchtitan`. This first
adapter supports the native Trainer with DP only, synchronous ACKs and native
DCP checkpoints. TP, CP, PP and either graph-capture path are rejected for online
features. It currently receives features on the host before native H2D transfer.

The online loader opens only after native DCP restore. It never seeks a live
queue. Each successful optimizer window commits its sample IDs to the existing
ledger and inbox before checkpoint/evaluation; a failed or partial window is
not ACKed. Resume requires the retained producer/feature store and a ledger
boundary matching the selected checkpoint. An ACK ledger ahead of an older
checkpoint is rejected, matching the existing recovery contract. Enabling
online features or evaluation is a workflow capability, not a promised speedup.

## Checkpoints and precision

TorchTitan saves full DCP checkpoints under
`output_dir/run_id/checkpoint/step-N`. Resume with `training.resume_from` pointing
to a native step directory or checkpoint root. The checkpoint includes native
optimizer/scheduler state, feature cursor, anchor RNG and rank RNG state. The
resume contract checks dataset identity, model/teacher provenance, objective,
batch size and parallel layout. A SpecForge FSDP checkpoint is not a Titan
optimizer checkpoint; use `model.draft_checkpoint_path` for a weights-only warm
start instead.

Final draft export writes an ordinary Hugging Face model directory under
`output_dir/run_id/draft`, including sharded safetensors for PP. Frozen teacher
weights are excluded from that export.

Native Titan uses FP32 parameters/Adam state, BF16 forward materialization and
FP32 gradient reduction. The existing SpecForge FSDP optimizer stores BF16 model
weights with FP32 masters. Report this difference in performance comparisons.
MFU is an analytical dense-work estimate; use measured step time and peak GPU
memory when comparing these custom sparse objectives.

Current frontend constraints include text features, BF16 compute, tracking
`none` or `tensorboard`, and no SpecForge profiler configuration or optimizer
CPU offload. Offline disaggregated ingestion is not connected to this adapter.
The runtime emits native weighted loss,
gradient norm, learning-rate and throughput metrics; selector, teacher and
acceptance diagnostics are not connected to its logger yet. Titan's
token-normalized maximum-local-loss diagnostic is omitted because it does not
represent these weighted draft objectives. Unsupported combinations fail before
training rather than silently using the FSDP runtime.
