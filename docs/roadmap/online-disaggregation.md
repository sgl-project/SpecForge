# Online Disaggregation Roadmap

Live disaggregated training: a rollout pool runs the **frozen target** model autoregressively
over a fixed prompt set and streams its hidden states across the network into a separate trainer
pool. This folds PR #618's online roadmap into the consolidated tree and applies the shared
decisions: **train-with-decode only** (the online target is frozen, so the streamed data is
independent of draft weights — there is no staleness and nothing to re-sync, and the whole
weight-sync / hot-update / on-policy axis is dropped here, kept only as the explicit out-of-scope
note below); and **Ray is OPEN/undecided** for the O2 scale-out layer — presented as a candidate
with a decision gate against a lighter torchrun/`rcli` alternative, not as a commitment.

Substrate is canonical and shared with the sibling docs: `SampleRef` (control plane, tensor-free)
+ `FeatureStore` (tensors: Local / SharedDir / Mooncake) + `FeatureDataLoader` → `TrainBatch`.
There is no separate hidden-state stream source of truth — `FeatureDataLoader` over
`SampleRef` + `FeatureStore` *is* the stream; online/offline/disagg vary only in the ref source +
the `FeatureStore` backend, shielded from training. See [domain-refactor](./domain-refactor.md)
for the domain `Trainer`/managers that wrap the runtime training seam and for Phase B's
`TargetEngine` that O1.3 depends on, [eval-and-breadth](./eval-and-breadth.md) for the orthogonal
eval/algorithm track, and the top-level [plan](../../plan.md).

---

## What we keep (do not rebuild)

The data plane is competitive — arguably ahead of the comparables on rigor (three backends behind
one contract, contract-tested, with a durability seam). The gap is the **orchestration +
live-engine + async control loop**. Reuse, do not rewrite:

- **Data plane** — `MooncakeFeatureStore` zero-copy (`data_plane/mooncake_store.py`) with hard-pin
  replication and per-generation key sets; `LocalFeatureStore` / `SharedDirFeatureStore` behind the
  same `FeatureStore` contract (`data_plane/feature_store.py`).
- **Control-plane vocabulary** — `SampleRef` (tensor-free; `assert_no_tensors` in `contracts.py`);
  `SampleRefQueue` with lease/ack and the reserved partition seam (`data_plane/sample_ref_queue.py`);
  `DataFlowController` commit/lease/ack + `reconcile_on_restart` (`control_plane/controller.py`);
  the `MetadataStore` ABC with `InMemoryMetadataStore` / `SQLiteMetadataStore`
  (`control_plane/metadata_store.py`).
- **Streaming seam** — `StreamingRefChannel` / `StreamingRefQueue`
  (`data_plane/streaming_ref_channel.py`) already carry committed refs cross-process with
  in-flight accounting and a close-and-drain protocol.
- **Trainer seam** — `TrainerCore` / `DraftTrainStrategy` / `TrainingBackend` + `StepContext`
  (`training/`), wrapped (not replaced) by the domain `Trainer`.

## Comparables (from #618)

| Axis | SpecForge (today) | TorchSpec | DeepSpec |
|---|---|---|---|
| Data plane (Mooncake zero-copy, tensor-free metadata) | **At parity / cleaner** — 3 backends, one contract, contract-tested, durability seam | `EagleMooncakeStore` (per-tensor keys) | disk cache, no streaming |
| Orchestration / scale-out | **GAP** — library + composable builders, controller in-process | Ray actors + placement groups + async controllers + `EnginePool` | single 8-GPU node, no Ray |
| Live generation (train-with-decode) | **GAP** — in-process generator stub today; no server | live autoregressive rollout streaming | offline regen-to-disk |
| Cross-pool backpressure | policy exists, off by default | pool-byte feedback + capacity await | n/a |
| Online weight sync / staleness | n/a — out of scope (frozen target) | **also absent** (not needed) | absent |
| Fault tolerance | `reconcile_on_restart` + checkpoint resume | checkpoint resume (≈ parity) | n/a |

**Read:** the data plane is at or ahead of parity; the competitive gap is orchestration + the
live engine + the async control loop. DeepSpec's separate edge (breadth + eval) is orthogonal and
lives in [eval-and-breadth](./eval-and-breadth.md).

---

### O1.1 — Shared cross-process control plane · size M · CPU · status: in-review
- **Goal** Producer and consumer attach to **one** durable control plane instead of each building
  its own `InMemoryMetadataStore`, so commit/lease/ack/dedup and restart-reconcile work across
  processes (~PR #624).
- **Target state** A separate producer and consumer share commit/lease/ack via a durable store;
  the "single-host in-process index" caveat is gone for the shared-store path; `reconcile_on_restart`
  is wired into the launcher and `resume=True` re-streams the committed-but-unacked tail.
- **Implementation**
  - `control_plane/metadata_store.py` — `SQLiteMetadataStore` is the end-to-end dev/single-host
    durable tier behind the existing `MetadataStore` ABC; a thin `RedisMetadataStore` is a later
    subclass behind the same interface (chosen at O2 if multi-node demands it).
  - `data_plane/sample_ref_queue.py` — `SampleRefQueue` operates over the shared store rather than
    purely in-process.
  - `control_plane/controller.py` — `DataFlowController(metadata_store=...)`; the producer's
    `commit_samples` and the consumer's `lease_train_refs` / `ack_train_refs` land in the same
    durable store; `reconcile_on_restart` resolves committed-but-unacked refs on attach.
  - `launch.py` — `build_disagg_online_producer` / `build_disagg_online_consumer` already accept
    `metadata_store` / `metadata_db_path` (O1.1 wiring); `_resolve_metadata_store` points both
    halves at the same durable path, and the consumer's `resume=True` reconciles before training.
- **Tests / gates** Two controllers over one store: producer commits, consumer leases/acks across
  processes; restart reconcile hands already-trained ids back as `skip_ids` and re-streams the tail;
  at-least-once dedup. `tests/test_runtime/test_disagg_online_launch.py`,
  `tests/test_runtime/test_disagg_online.py`.
- **Done when** A separate producer and consumer share commit/lease/ack via the durable store, and
  the single-host-in-process-index caveat is removed for the shared-store path.

### O1.2 — Async streaming loop + one-process builder · size M · CPU / small-GPU · status: in-review
- **Goal** Replace synchronous drain-then-fit with an async streaming loop, using the **existing
  in-process generator** as a stub so this milestone carries no engine risk (~PR #625).
- **Target state** End-to-end live 1+1 run with in-process generation: producer pulls prompts →
  generates → `put` + commit ref; consumer leases → `get` → trains → acks at the optimizer step;
  a bounded sample pool throttles via the channel's in-flight watermark.
- **Implementation**
  - `launch.py` — the composable builders already exist today:
    `build_disagg_online_producer` returns `(workers, drive_producer)`;
    `build_disagg_online_consumer` returns `(trainer, loader)`;
    `run_disagg_online_interleaved` runs the producer on a background thread while `trainer.fit`
    consumes on the main thread, with symmetric hang-free shutdown (trainer-done sets `should_stop`;
    producer-done closes the channel; producer-raise closes the channel then re-raises).
    `build_disagg_online_runtime` (aliased `build_disagg_online_eagle3_runtime`) composes the two
    halves and dispatches to `run_disagg_online_interleaved`.
  - The producer's `drive_producer(should_stop=...)` applies channel backpressure (pauses while
    `channel.in_flight_remote()` exceeds `in_flight_high_watermark`) and closes the channel on exit.
  - In-process generator stub: `SGLangAdapter.generate_features` over the SpecForge
    `Eagle3TargetModel.generate_eagle3_data` path (`inference/sglang_adapter.py`), driven by
    `RolloutWorker` (`inference/rollout_worker.py`).
- **Tests / gates** Tiny-model 1+1 online run: trainer consumes a **streamed** (not precomputed)
  ref set; metrics align with the offline baseline on the same prompts/seed. Interleaved
  shutdown is hang-free in all three orderings. `tests/test_runtime/test_disagg_online_launch.py`,
  `tests/test_runtime/test_streaming_ref_channel.py`.
- **Done when** An end-to-end live 1+1 run drives training from an in-process generator over the
  streaming channel, with no precomputed features.

### O1.3 — Live SGLang-server hidden-state capture · size L · GPU + engine · status: next
> 🔴 **GATING RISK — run the capture spike first.** The unknown is how far
> `Eagle3TargetModel.generate_eagle3_data` is from server-backed capture. The spike is the
> difference between O1.3 being days vs. a quarter; SpecForge is SGLang's own project so reuse is
> plausible (TorchSpec needed an SGLang codebase patch). Run it before/with O1.1.
- **Goal** Replace the in-process generator with a real SGLang **server** emitting aux + final
  hidden states straight into `MooncakeFeatureStore`.
- **Target state** A live SGLang server feeds training with **zero** precomputed features: server
  generates → hidden states land in the store → consumer trains; acceptance metrics match the
  offline baseline.
- **Implementation**
  - Depends on [domain-refactor](./domain-refactor.md) Phase B's `TargetEngine` abstraction
    (`SGLangServerEngine`) — O1.3 is the online consumer of that engine.
  - `inference/sglang_adapter.py` — a server-backed `FeatureSource` (launch/attach an SGLang
    server) alongside the existing in-process `SGLangAdapter`; the capture hook is engine-side,
    writing aux + final hidden states to the `FeatureStore`. Reuse the `CaptureConfig` /
    `verify_capture` contract (`inference/capture.py`) so an aux-layer-id / width / target-dim
    mismatch fails loudly. See `inference/sglang_patch_inventory.md` for the patch surface.
  - `inference/rollout_worker.py` — the producer uses the server-backed source; committed
    `SampleRef`s flow through the same `StreamingRefChannel`.
- **Tests / gates** (gated GPU) Server generates → hidden states resident in the store → consumer
  trains; capture verification passes; acceptance vs. offline. `tests/test_runtime/test_mooncake_store.py`
  for the store path.
- **Done when** A live SGLang server feeds training with zero precomputed features and acceptance
  metrics match the in-process / offline baseline.

### O2 — Scale-out orchestration (Ray = OPEN) · size L · infra · status: later
- **Goal** Move from 1+1 to N producers + M trainers with per-DP-rank leasing, cross-pool
  backpressure, and a managed streaming-pool lifetime — and **decide** the orchestration layer.
- **Target state** Independent N producers and M trainers scale without re-leasing or dropping
  samples; a slow trainer throttles the producer (bounded memory, not OOM); the online
  `MooncakeFeatureStore` bugs that become reachable at scale are closed; the orchestration layer is
  chosen against an explicit gate.

#### Orchestration: Ray candidate vs. lighter alternative (decision gate)

Ray is a **candidate** for the scale-out orchestration layer, not a commitment and not a non-goal.
It is likely necessary for multi-node N-producer/M-trainer scale-out and high-bandwidth
isolated-pool runs (its actor/placement model maps cleanly onto independent rollout and trainer
pools with separate GPU counts/sharding, matching the comparables' shape). The lighter alternative
is **torchrun + the existing `rcli` multi-node launch** driving the already-composable
`build_disagg_online_producer` / `build_disagg_online_consumer` builders across nodes, with the
shared durable `MetadataStore` (O1.1) as the only cross-process coordination point — no actor
framework.

- **Trade-off** Ray buys dynamic placement, actor restart/health, and a uniform multi-pool control
  surface, at the cost of a heavy runtime dependency, a new failure surface, and packaging/operability
  overhead on the GPU boxes. torchrun + `rcli` keeps the dependency surface minimal and reuses the
  existing launch path, but leaves placement, fan-out, and restart as bespoke launcher logic.
- **Decision gate — adopt Ray only if** the N+M / multi-node runs demonstrably need *dynamic*
  placement or per-actor restart that the static torchrun + `rcli` + shared-store path cannot
  deliver (e.g. heterogeneous pool sizes that must reshard at runtime, or actor-level fault recovery
  that bash-level relaunch cannot meet the availability target for). If a static launch over the
  shared store carries N+M with acceptable restart behaviour, stay on the lighter path. Re-evaluate
  at the first true multi-node scale-out run.

#### dp_partition leasing
- **Implementation** `data_plane/sample_ref_queue.py` — the partition seam exists today:
  `dp_partition(sample_id, num_partitions)` computes a stable DP-rank assignment, and
  `lease(..., partition=(index, num_partitions))` restricts a lease to a DP shard via
  `_lease_partition_locked`. The reserved `partition_key` producer-side routing hint is
  accepted-and-ignored today; turn on per-DP-rank leasing so every sample is leased exactly once
  across a reshard. `control_plane/controller.py` gains multi-trainer lease dispatch.
- **Tests / gates** N+M run; every sample leased exactly once across a reshard; no double-train;
  changing `num_partitions` re-distributes the same committed pool without re-leasing or dropping.

#### Cross-pool backpressure
- **Implementation** `control_plane/backpressure.py` exists and is **off by default**:
  `BackpressureController` with `should_pause_prompts` / `cap_prompt_grant` / `cap_train_lease` and a
  high/low watermark `BackpressureConfig`. Wire the controller into the dispatch loop so it returns
  pool-size feedback and blocks granting when the pool is full; the producer throttles instead of
  OOM-ing the store. `control_plane/controller.py` reports capacity.
- **Tests / gates** A slow trainer makes the producer throttle (bounded store memory) rather than
  OOM.

#### Streaming-pool lifetime + known online MooncakeFeatureStore bugs
- **Implementation** `data_plane/mooncake_store.py` — add consume-then-GC / TTL lifetime for the
  streaming pool, and close the two online-only bugs that become reachable at scale, both of which
  the shared metadata index (O1.1) unblocks:
  - **release-pending re-put gc-clobber** — a re-put bumps the per-sample generation and removes the
    prior key set; a parked failed-free in `_release_pending` plus `gc()` can clobber the fresh
    blob. Needs a tombstone-then-free protocol tied to the shared index.
  - **cross-process abort tombstone** — under a lease-deferred remove, `producer.abort` marks only
    its own `_freed`, so the bytes linger and a *separate* consumer (empty `_freed`) still resolves
    the aborted ref → stale. This is the `expectedFailure`
    `test_cross_process_abort_under_lease_defer_is_known_gap` in
    `tests/test_runtime/test_mooncake_store.py:445`; flip it to a hard assertion once the shared
    index lands.
- **Tests / gates** Re-put-while-pending no longer deletes the fresh blob; cross-process abort
  blocks a separate consumer (the `expectedFailure` becomes a real assertion); consume-once
  streaming is correct cross-process.
- **Done when** N producers and M trainers scale without re-leasing/dropping; trainer-behind-rollout
  is a bounded-memory backpressure signal; the streaming-pool bugs are closed; the orchestration
  decision gate has been evaluated against a real N+M run.

### O3 — Hardening · size L · infra · status: later
- **Goal** Make the live disaggregated path production-operable.
- **Target state** RDMA path runs without per-op buffer churn; actor/process restart and health are
  handled; the run is observable.
- **Implementation**
  - RDMA buffer-pool registration in `data_plane/mooncake_store.py` — replace the per-op
    `register_buffer` / `unregister_buffer` around `_store_put_tensor` / `_store_get_tensor` with a
    registered pool before flipping `DEFAULT` `protocol` from `tcp` to `rdma`.
  - Actor / process restart + health: restart the rollout/trainer/controller processes (under
    whichever orchestration O2 selects) with `reconcile_on_restart` driving recovery from the shared
    store; health endpoints on `RolloutWorker.health` / `SGLangAdapter.health`.
  - Observability: pool depth, in-flight, lease/ack rates, backpressure snapshots
    (`BackpressureController.snapshot`), capture-verification counts.
- **Tests / gates** Kill-and-restore a producer/trainer mid-run and confirm no lost or
  double-trained samples; RDMA path registers buffers once; metrics emitted.
- **Done when** The live disaggregated run survives a process restart with no data loss and runs the
  RDMA path with one-time buffer registration.

---

## Provenance, not weight sync

`draft_weight_version` is retained **only as provenance metadata** on each committed sample — it is
recorded by `RolloutWorker._put_metadata` / `_offline_metadata` (`inference/rollout_worker.py`),
threaded through `contracts.py`, the metadata store, the controller, and every `FeatureStore`
backend's metadata round-trip. It exists for debugging/audit. There is **no** staleness gate, no
version registry, and no consumer-side check keyed on it. Because the online target is frozen, the
streamed hidden states are independent of draft weights.

## Dependency graph

```
                          capture spike (gates O1.3)
                                   │
O1.1 ──▶ O1.2 ──▶ O1.3 ──▶ O2 ──▶ O3
 │                  │
 │                  └─ depends on domain Phase B TargetEngine (SGLangServerEngine)
 │
 └─ shared MetadataStore unblocks the O2 online-store bug fixes (re-put gc-clobber,
    cross-process abort tombstone)

eval-and-breadth track ── independent (orthogonal to disaggregation)
```

## Explicitly out of scope

- **Draft-weight sync, weight-version registry, hot draft-update, staleness gate,
  `WeightPublisher` / `WeightVersion` / serving accept-length gate / `ServingTrafficStream` /
  serving-traffic capture** — not needed for train-with-decode. The online target is frozen; the
  draft is not in the generation loop; the data is independent of draft weights, so there is no
  staleness and nothing to re-sync. `draft_weight_version` survives as provenance only (above).
- **True on-policy / draft-coupled training** (draft proposes → target verifies → train on realized
  acceptance, requiring live draft-weight sync) — a separate research bet, not addressed here. No
  public framework ships it.
- **Elastic / dynamic engine scaling at runtime** — engines are fixed at init; the comparables do
  the same.
