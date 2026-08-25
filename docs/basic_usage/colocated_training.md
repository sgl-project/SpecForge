# Colocated online training

Colocated mode loads one SGLang target shard and one FSDP draft shard in every
trainer process. Target capture and draft training alternate on the same GPU,
so hidden states never cross a process or node. Use it when the combined target,
KV/cache reservation, draft, optimizer, activations, and collective workspaces
fit with enough headroom for the longest configured sequence.

Start the Qwen3-8B H200 recipe from the repository root:

```bash
specforge train --config examples/configs/online/colocated/qwen3-8b-dspark-colocated.yaml
```

Every rank runs as role `all`; the CLI starts torchrun itself when
`deployment.trainer.nproc_per_node` is greater than one.

## Pull-through capture

The runtime is pull-through: when the trainer asks for its next microbatch, the
rank ingests exactly one target batch of prompts, SGLang captures it
synchronously, the rank trains its local slice, and the features are released
before the next capture. Nothing is captured ahead of demand, so the
rank-private in-memory store never stages more than one local batch. The loader
skips its defensive clone for that store (`CLONE_ON_FETCH` has no effect here)
and stays synchronous regardless of `data.dataloader_num_workers`, keeping all
CUDA work on the trainer thread.

At each `training.log_interval`, colocated runs add capture time/throughput,
peak staged features, and accelerator allocated/reserved peaks to the existing
`perf/*` metrics. Compare `perf/data_wait_time_s` with
`perf/train_compute_time_s` to decide whether target TP or draft DP is the next
scaling bottleneck.

## Prompt plan

Colocated runs use the same deterministic prompt plan as the disaggregated
producer: every epoch is the `training.prompt_seed + epoch` permutation of the
prepared prompts, and passes after the first mint epoch-specific task ids. Each
target-DP island (below) takes a strided shard of that permutation, truncated so
every island receives the same target-batch-aligned count. A colocated and a
disaggregated run over the same prompts and seed therefore see the same sample
identities and can be compared at matched samples.

Checkpoints record the plan (seed, epochs, island count, dataset size, batch
and TP sizes). Resuming under a different plan raises instead of silently
training the wrong slice; changing `training.num_epochs` on resume is one such
change.

## Memory sizing

`model.sglang_mem_fraction_static` is SGLang's static target budget, not an
exclusive partition of the GPU. It covers target weights plus SGLang-managed
KV/cache pools. The remainder is shared by the draft weights/FSDP shards,
optimizer state, activations, temporary kernels, NCCL buffers, and allocator
fragmentation.

Measure the post-warm-up peak at the production `max_length`, local batch, and
accumulation settings. Fixed shapes make the peak repeatable, but lazy kernel
loading, compilation, communication buffers, and allocator fragmentation still
make the first steps different from steady state.

## Target islands

`training.tp_size` is the target TP width in colocated mode. The world is split
into contiguous target-TP islands; islands are the target-DP replicas and
receive disjoint prompt shards. All TP peers of an island capture the same
TP-wide batch, and each peer trains only its contiguous local slice. The full
packed output exists transiently during capture; each peer detaches its slice
and releases the other slices' allocation before draft training begins.

`model.sglang_context_length` defaults to `data.max_length + 7`, and
`sglang_max_running_requests` / `sglang_max_total_tokens` default to
`training.tp_size * training.batch_size` requests of that length per island.
Explicit values override the defaults; the schema rejects values below them.

## HSDP across islands

For multi-node targets, `training.fsdp_sharding: HYBRID_SHARD` uses the target
TP group as the FSDP shard group and the target-DP group for replication:

| Target shape | Suggested topology | Draft sharding |
| --- | --- | --- |
| Qwen3-8B on 8 H200 | 8 islands of TP1 | `SHARD_GRAD_OP` across all 8 ranks |
| One-node target | one TP island per node | `HYBRID_SHARD` |
| K3-class target on 4x8 B300 | 4 islands of TP8 | `HYBRID_SHARD` |

For a TP8 target on four eight-GPU nodes:

```yaml
training:
  batch_size: 1
  tp_size: 8
  fsdp_sharding: HYBRID_SHARD

deployment:
  mode: local_colocated
  trainer:
    nnodes: 4
    nproc_per_node: 8
    master_addr: trainer-0
```

The complete K3 starting recipe is
[`kimi-k3-dspark-colocated.yaml`](../../examples/configs/online/colocated/kimi-k3-dspark-colocated.yaml).

HSDP shards the draft inside each island and replicates corresponding shards
across islands, so parameter all-gathers stay inside the island and only replica
synchronization crosses islands. That traffic is node-local only when an island
does not span nodes, so `training.tp_size` must divide
`deployment.trainer.nproc_per_node` (validated at config load). Loss and metric
reductions remain WORLD-wide, and the gradient norm counts each replicated shard
once.

Prompt-cache preparation is coordinated: rank zero builds the tokenized Arrow
cache first (node-local caches are then built once per remaining node) and all
other ranks take the cache-hit path. The coordination collective runs inside
`training.dist_timeout`, so raise that timeout or pre-build the cache for very
large raw datasets.

Colocation maximizes fixed-GPU throughput when it fits because it removes
feature serialization and transport. Disaggregation remains preferable when a
target cannot share memory safely, producer/consumer elastic scaling matters,
or independent fault domains are more important than per-GPU throughput.
