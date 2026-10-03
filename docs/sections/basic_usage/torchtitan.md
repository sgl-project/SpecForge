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

SGLang feature capture runs separately. This runtime currently reads existing
offline feature files through the same algorithm reader, normalizer and
collator as SpecForge's FSDP runtime. Live queue acknowledgements and online
consumer recovery need their own stateful data adapter.

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
implementations. CUDA graphs remain disabled because the custom objective has
dynamic work and host synchronization. Enabling the flag fails explicitly.
The separate upstream GraphTrainer, expert parallelism, FP8, async TP, USP and
non-GQA tensor-parallel plans are outside this adapter's supported contract.

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

Current frontend constraints: offline colocated text features, BF16 compute,
tracking `none` or `tensorboard`, and no periodic validation, SpecForge profiler
configuration or optimizer CPU offload. The runtime emits native weighted loss,
gradient norm, learning-rate and throughput metrics; selector, teacher and
acceptance diagnostics are not connected to its logger yet. Titan's
token-normalized maximum-local-loss diagnostic is omitted because it does not
represent these weighted draft objectives. Unsupported combinations fail before
training rather than silently using the FSDP runtime.
