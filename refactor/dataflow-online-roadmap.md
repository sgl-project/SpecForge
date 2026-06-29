# DataFlow Runtime — Online Disaggregated Training Roadmap

**Status:** planning. Target = a *live* disaggregated run (rollout pool generates target
hidden states on the fly, trainer pool consumes them over the network), matching the
architecture of [TorchSpec](https://github.com/lightseekorg/TorchSpec) (the framework
behind the EAGLE 3.1 / Kimi K2.5 drafters).

This document records the decision, the gap vs. the SOTA comparables, and — most
importantly — a **decomposition of the work into a stacked series of small, reviewable
PRs** (no mega-PRs; merge bottom-up, same as the M1–M6 series).

---

## 1. Decision: "train-with-decode", *not* on-policy

The chosen regime is **train-with-decode** (TorchSpec-style): a live inference engine runs
the **frozen target** model autoregressively over a fixed prompt set and streams its
hidden states into training. The draft model being trained is **not** in the generation
loop.

**Consequence — weight sync / staleness is OUT of scope.** Because the streamed data is
target hidden states over a fixed prompt distribution, and the target is frozen, the data
is independent of the draft weights. There is therefore *no* staleness and *nothing to
re-sync*. This is confirmed by reading TorchSpec's controller source: it has **no**
draft-weight reload, versioning, or staleness logic — the inference manager is a stateless
dispatcher. The SOTA does not solve on-policy weight sync for this task because it does not
need to.

This **removes** the previously-floated milestone "online disaggregated training + weight
sync + two-axis staleness gate". Keep `draft_weight_version` as provenance metadata (cheap,
useful for debugging); delete the staleness *gate* and the hot draft-update path from the
near-term plan.

> True on-policy spec-decode training (draft proposes → target verifies → train on realized
> acceptance, requiring live weight sync) is a separate, research-grade direction. No public
> framework ships it. Out of scope here; revisit only as a deliberate research bet.

---

## 2. Where we are vs. the comparables

| Axis | SpecForge (M6) | TorchSpec | DeepSpec |
|---|---|---|---|
| Data plane (Mooncake zero-copy, tensor-free metadata) | **At parity / cleaner** — 3 backends behind one contract (Local/SharedDir/Mooncake), B5/B9 formalized, contract-tested, durability seam | `EagleMooncakeStore` (per-tensor keys + shapes/dtypes) | disk cache (~38 TB), no streaming |
| Orchestration / scale-out | **GAP** — library + bash `RCLI_NODE_RANK` + `.done/.consumed` sentinels; controller in-process, no run loop | Ray actors + `create_placement_groups` + `AsyncTraining/InferenceController` + `EnginePool` | single 8-GPU node (`mp.spawn`), no Ray |
| Live generation ("train with decode") | **GAP** — producer ingests precomputed `.ckpt`; `SGLangAdapter` is in-process, not a server; only online builder is colocated + synchronous | live autoregressive rollout streaming | offline regen-to-disk |
| Cross-pool backpressure | policy object exists, **off by default, unwired** | pool-byte feedback + `_await_pool_capacity` | n/a |
| Online weight sync / staleness | stubbed | **also absent** (not needed) | absent |
| Algorithm breadth + eval | EAGLE3-centric, no eval harness | EAGLE3/3.1 | **DSpark/DFlash/Eagle3 + 9-benchmark eval** |
| Fault tolerance | `reconcile_on_restart` + checkpoint resume | checkpoint resume + manual Ray debug (≈ parity) | n/a |

**Read:** our *data plane* is competitive (arguably ahead on rigor). The gap is the
**orchestration + live-engine + async control loop**. DeepSpec's separate edge (breadth +
eval) is orthogonal to disaggregation.

Sources: [PyTorch TorchSpec blog](https://pytorch.org/blog/torchspec-speculative-decoding-training-at-scale/) ·
[lightseekorg/TorchSpec](https://github.com/lightseekorg/TorchSpec) ·
[EAGLE 3.1 (vLLM)](https://vllm.ai/blog/2026-05-26-eagle-3-1) ·
[deepseek-ai/DeepSpec](https://github.com/deepseek-ai/DeepSpec).

---

## 3. What we keep (do not rebuild)

- **Data plane** — `MooncakeFeatureStore` zero-copy (PR #614) ≈ `EagleMooncakeStore`.
- **Control-plane vocabulary** — `SampleRef` (tensor-free) ≈ shapes/dtypes metadata;
  `SampleRefQueue` (lease/ack/`dp_partition`); `DataFlowController` commit/lease/ack +
  `reconcile_on_restart`.
- **Trainer** — `TrainerController/Core` + `Eagle3TrainStrategy` + FSDP/TP/Ulysses-SP.

## 4. Target architecture

```
            ┌──────────────── Ray cluster ────────────────┐
            │                                              │
  prompts → │  RolloutActor×N            TrainerActor×M    │
            │  (SGLang server,      ┌──▶ (FSDP+TP+SP,       │
            │   frozen target) ──put┤    leases refs,       │
            │       │   commit ref  │    get + train + ack) │
            │       ▼               │         │             │
            │  MooncakeFeatureStore │         │             │
            │  (per-tensor, hard-pin)         │             │
            │       ▲   keys+meta only        ▼             │
            │  ControllerActor (sample pool, lease/ack,     │
            │   per-DP-rank dispatch, backpressure)         │
            └──────────────────────────────────────────────┘
   shared metadata/lease index (SQLite→Redis); tensors never leave the store
```

This is functionally TorchSpec's shape (`AsyncInferenceManager` + `AsyncTrainingController`
+ `EnginePool` + Mooncake store), built on SpecForge's existing FeatureStore/SampleRef/
trainer abstractions.

---

## 5. Milestones

- **O1 — Live single-pair.** One rollout proc + one trainer proc, live generation,
  cross-process control plane. *No Ray yet.* Proves the data + control paths live.
- **O2 — Ray orchestration + scale-out.** Actor/placement layer; N producers + M trainers;
  cross-pool backpressure; DP-resharding; streaming-pool lifetime.
- **O3 — Hardening.** RDMA buffer-pool, actor restart/health, observability. (Last; TorchSpec
  is weak here too, so we are near parity.)
- **Eval track (parallel, orthogonal).** Acceptance-length eval harness + algorithm breadth
  (MTP/DFlash). DeepSpec's edge; can be owned independently of the above.

---

## 6. PR plan (the decomposition)

Stacked series, **merge bottom-up**. Sizes: **S** ≈ ½ day review, **M** ≈ 1 day, **L** ≈ 2 days.
"CPU" PRs run in fast CI; "GPU/engine" PRs need the H200 box.

### O1 — Live single-pair

**O1.1 — Shared, cross-process control plane** · **M** · CPU
- *Goal:* producer and consumer attach to **one** control plane instead of each building its
  own `InMemoryMetadataStore`.
- *In:* wire `SQLiteMetadataStore` end-to-end (and/or add a thin `RedisMetadataStore` behind
  the existing `MetadataStore` ABC) for cross-process commit/lease/ack/dedup; lift the
  FeatureStore generation/lease index behind a shared interface so liveness is not
  in-process-only for the single-host case; make `SampleRefQueue` operate over the shared
  store. Wire `reconcile_on_restart` into the launcher.
- *Out:* live generation, Ray, byte-level backpressure.
- *Files:* `control_plane/metadata_store.py`, `control_plane/controller.py`,
  `data_plane/sample_ref_queue.py`, store index hook in `data_plane/{feature_store,
  disaggregated,mooncake_store}.py`.
- *Test (CPU):* two controllers over one store — producer commits, consumer leases/acks
  across processes; restart reconcile; dedup at-least-once.
- *Done when:* a separate producer and consumer share commit/lease/ack via the durable store;
  the "single-host in-process index" caveat is removed for the shared-store path.

**O1.2 — `build_disagg_online_eagle3_runtime` + async loop** · **M** · CPU / small-GPU
- *Goal:* the async streaming loop, replacing synchronous drain-then-fit — using the
  **existing in-process generator** (`Eagle3TargetModel.generate_eagle3_data`) as a stub so
  this PR carries no engine risk.
- *In:* new builder in `launch.py`; producer loop (pull prompts → generate → `put` + commit
  ref); consumer loop (lease → `get` → train → ack at optimizer step); bounded sample pool
  (count-based).
- *Out:* live SGLang server (O1.3), Ray (O2), byte-backpressure (O2.3).
- *Files:* `runtime/launch.py`, `runtime/rollout_worker.py`, a small online driver.
- *Test:* tiny-model 1+1 online run; trainer consumes a **streamed** (not precomputed) ref
  set; metrics align with the offline baseline on the same prompts/seed.
- *Done when:* end-to-end live 1+1 run with in-process generation.

**O1.3 — Live SGLang-server hidden-state capture** · **L** · GPU + engine · 🔴 *gating risk*
- *Pre-req:* the capture **spike** (size how far `generate_eagle3_data` is from server-backed
  capture — see §7).
- *Goal:* replace the in-process generator with a real SGLang **server** emitting aux+final
  hidden states straight into the FeatureStore.
- *In:* SGLang server launch/attach in the producer; hidden-state capture hook (engine-side,
  cf. TorchSpec's SGLang patch) writing to `MooncakeFeatureStore`; producer commits
  `SampleRef`s.
- *Out:* multi-engine pool, Ray.
- *Files:* `runtime/sglang_adapter.py` (server-backed), `runtime/rollout_worker.py`, possibly
  a small SGLang patch.
- *Test (gated GPU):* server generates → hidden states land in store → consumer trains;
  accept metrics vs. offline.
- *Done when:* a live SGLang server feeds training with **zero** precomputed features.

### O2 — Ray orchestration + scale-out

**O2.1 — Ray actor + placement layer (1+1 under Ray)** · **M** · infra
- *Goal:* prove the actor layer at 1+1 before scaling.
- *In:* `RayActor`-style wrappers for rollout pool, trainer pool, and a controller actor;
  placement groups with independent GPU counts/sharding (≈ `create_placement_groups`).
- *Out:* multi-actor fan-out (O2.2).
- *Files:* new `runtime/ray/` (actors, placement), thin adapters over O1 builders.
- *Test:* 1+1 run under Ray reproduces the O1.3 result.
- *Done when:* rollout/trainer/controller run as independent Ray actors with explicit
  placement.

**O2.2 — Scale-out: multi-producer / multi-trainer + DP-resharding** · **M**
- *In:* N rollout actors + M trainer actors; wire the reserved `dp_partition` leasing seam
  (today `partition_key` is accepted-and-ignored, no launcher passes a partition);
  per-DP-rank queues; round-robin engine pick (≈ `EnginePool`).
- *Files:* `data_plane/sample_ref_queue.py` (partition leasing on), `runtime/ray/*`,
  `control_plane/controller.py` (multi-trainer).
- *Test:* N+M run; every sample leased exactly once across a reshard; no double-train.
- *Done when:* independent N producers and M trainers scale without re-leasing/dropping.

**O2.3 — Cross-pool backpressure** · **S–M**
- *In:* turn on the existing `backpressure.py` policy; controller returns pool-size feedback;
  dispatch blocks when the pool is full (≈ `_await_pool_capacity`); rollout throttles.
- *Files:* `control_plane/backpressure.py` (wire), `control_plane/controller.py`, `runtime/ray/*`.
- *Test:* a slow trainer makes the producer throttle rather than OOM the store.
- *Done when:* "trainer behind rollout" is a bounded-memory backpressure signal, not an OOM.

**O2.4 — Streaming-pool lifetime + online store-bug fixes** · **M**
- *In:* consume-then-GC / TTL lifetime for the streaming pool (cf. `DeferredDeleteManager`);
  fix the online-only `MooncakeFeatureStore` bugs that become reachable here —
  release-pending + re-put gc-clobber, and the cross-process abort tombstone (the
  `expectedFailure`), both of which close once O1.1's shared index exists; RDMA buffer-pool
  registration (replace per-op register/unregister) before flipping `protocol=rdma`.
- *Files:* `data_plane/mooncake_store.py`, shared-index hook from O1.1.
- *Test:* re-put-while-pending no longer deletes the fresh blob; cross-process abort blocks a
  separate consumer (flip the `expectedFailure` to a real assertion).
- *Done when:* consume-once streaming is correct cross-process; RDMA path registers buffers
  once.

### Parallel — Eval & breadth (separate stack)

**E1 — Acceptance-length eval harness** · **M** · and **E2 — algorithm breadth (MTP/DFlash)** · **L**.
Orthogonal to disaggregation; owner can run independently. Matches DeepSpec's edge.

### Dependency graph

```
O1.1 ─▶ O1.2 ─▶ O1.3 ─▶ O2.1 ─▶ O2.2 ─▶ O2.3 ─▶ O2.4
(spike for O1.3 runs before/with O1.1)         E1, E2 ── independent
```

---

## 7. Risks & open questions

- 🔴 **SGLang hidden-state capture (gates O1.3, sizes the whole effort).** Unknown how far
  `Eagle3TargetModel.generate_eagle3_data` is from server-backed capture. **Run the spike
  first** — it's the difference between O1.3 being days vs. a quarter. SpecForge is SGLang's
  own project, so reuse is plausible; TorchSpec needed an SGLang codebase patch.
- Shared metadata backend choice (SQLite single-host vs. Redis) — O1.1 picks; SQLite is
  enough for single-host O1, Redis for true multi-node O2.
- Fault tolerance / actor restart is deferred to O3 (TorchSpec is weak here too — near parity,
  not a competitive gap).

## 8. Explicitly out of scope (this roadmap)

- Draft-weight sync, weight-version registry, hot draft-update, staleness gate — not needed
  for train-with-decode (§1).
- True on-policy / draft-coupled training — research bet, separate track.
- Elastic/dynamic engine scaling at runtime (TorchSpec fixes engines at init; we match).
