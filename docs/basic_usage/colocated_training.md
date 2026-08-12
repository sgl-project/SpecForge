# Colocated online training

Colocated mode loads one SGLang target shard and one FSDP draft shard in every
trainer process. Target capture and draft training alternate on the same GPU,
so hidden states never cross a process or node. Use it when the combined target,
KV/cache reservation, draft, optimizer, activations, and collective workspaces
fit with enough headroom for the longest configured sequence.

Start the Qwen3-8B H200 recipe from the repository root:

```bash
specforge train --config examples/configs/qwen3-8b-dspark-colocated.yaml
```

The runtime is pull-through: the trainer requests one batch, SGLang captures
it synchronously, the rank-local feature slice is trained, and its local store
entry is released before the next capture. `zero_copy_features: true` avoids a
defensive loader clone. With target TP greater than one, the capture adapter
still copies this rank's slice once to detach it from the full packed output;
otherwise a small rank-local view would pin every peer's hidden-state storage.
`synchronize_after_capture: true` is the stable default because it prevents
SGLang stream tails from overlapping a following FSDP collective; disable it
only after profiling the exact model/SGLang/PyTorch stack.

FlexAttention uses the stable Triton kernel on Hopper and older GPUs. On
Blackwell with PyTorch 2.11 or newer it selects the faster FLASH kernel; set
`SPECFORGE_FLEX_ATTENTION_BACKEND=TRITON` to force the portable path or
`SPECFORGE_FLEX_ATTENTION_BACKEND=FLASH` after validating a new compiler stack.
The Inductor configuration retains an ATen fallback for dynamic shapes.

At each `training.log_interval`, colocated runs add capture time/throughput,
rank-local feature-residency peaks, and accelerator allocated/reserved peaks to
the existing `perf/*` metrics. Use `perf/data_wait_time_s` versus
`perf/train_compute_time_s` to decide whether target TP or draft DP is the next
scaling bottleneck.

## Memory sizing

`model.sglang_mem_fraction_static` is SGLang's static target budget, not an
exclusive partition of the GPU. It covers target weights plus SGLang-managed
KV/cache pools. The remainder is shared by the draft weights/FSDP shards,
optimizer state, activations, temporary kernels, NCCL buffers, and allocator
fragmentation. For example, `0.76` on a 280 GiB B300 asks SGLang to budget about
212.8 GiB; it does not reserve the remaining 67.2 GiB exclusively for FSDP.

Measure the post-warm-up peak at the production `max_length`, local batch, and
accumulation settings. Fixed shapes make the peak repeatable, but lazy kernel
loading, compilation, communication buffers, and allocator fragmentation still
make the first steps different from steady state.

## Target islands and HSDP

`training.tp_size` is the target TP width in colocated mode. The world is split
into contiguous target-TP islands; every island receives a deterministic,
disjoint prompt shard. All TP peers capture the same TP-wide batch, and each
peer trains only its contiguous local slice. The full packed output exists
transiently during target capture; before draft training begins, each peer
detaches its slice and releases the other slices' backing allocation.

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
  colocated:
    synchronize_after_capture: true
    zero_copy_features: true
```

The complete K3 starting recipe is
[`kimi-k3-dspark-colocated.yaml`](../../examples/configs/kimi-k3-dspark-colocated.yaml).

HSDP shards the draft inside each TP8 island and replicates corresponding
shards across islands. This keeps parameter all-gathers node-local while only
the replica synchronization crosses nodes. Set `model.sglang_context_length`
to at least `data.max_length + 7`; size `sglang_max_running_requests` and
`sglang_max_total_tokens` for `training.tp_size * training.batch_size` requests
per island. Explicit values override the safe derived defaults.

Colocation maximizes fixed-GPU throughput when it fits because it removes
feature serialization and transport. Disaggregation remains preferable when a
target cannot share memory safely, producer/consumer elastic scaling matters,
or independent fault domains are more important than per-GPU throughput.
