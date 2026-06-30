# SpecForge Architecture Plan (Reconciled)

> **Status**: Draft — reconciles the original "SpecForge Redesign Plan" (a from-scratch,
> torchrun-native, HTTP-only target architecture) with the **landed DataFlow `runtime/`
> spine** (control/data/inference/training planes, SampleRef + FeatureStore + Mooncake).
>
> **Supersedes** the previous redesign draft. That draft's detailed component sketches
> (CheckpointManager, Evaluator, Pydantic config, MLA draft, SGLang export, train-with-decode)
> remain valid as the **domain-layer target** and are preserved verbatim in
> [`docs/redesign-draft-legacy.md`](docs/redesign-draft-legacy.md); this document
> re-frames *where they sit* now that the DataFlow runtime exists, and corrects the one bet
> the original draft got wrong for our actual workloads (see §6).

---

## 0. TL;DR — the decision

There were two architecture efforts in flight for the same goal (organize SpecForge so new
draft models / training modes / disaggregation compose cleanly without a `train_XXX.py` per
method):

- **The redesign draft** (this doc's predecessor): clean domain decomposition
  (`models/drafts` registry, `TargetEngine`, `HiddenStateStream`, `DraftTrainStrategy`,
  `Trainer`, config/eval/export/cli) — but **explicitly torchrun-native with no
  Ray/Mooncake**; disaggregation was "the target is an HTTP service" (`RemoteStream`).
- **The DataFlow `runtime/` spine** (landed: the `dataflow-up-1..20` stack, M1–M7/O1): a
  metadata-only **control plane** + a tensor-carrying **data plane** (`SampleRef` +
  `FeatureStore` incl. **Mooncake** + `StreamingRefChannel` + durable metadata + backpressure
  + resume), with disaggregated producer/consumer and an interleaved online loop.

**The fact that settles it:** there is a real **multi-node / >100 GB/s / isolated-pool**
requirement (separate rollout and trainer pools, no shared FS / no viable HTTP path for the
hidden-state volume). That **breaks the original draft's central bet** ("HTTP/gRPC to SGLang
is sufficient; Mooncake only matters >100 GB/s which we never hit"). So the runtime's
Mooncake control/data plane is **necessary, not premature**.

**Decision: do not pick one — layer them.**

- **Canonical substrate = the DataFlow `runtime/` control + data plane.** `SampleRef` (metadata
  on the control plane) + `FeatureStore` (tensors, Local/SharedDir/Mooncake) + `FeatureDataLoader`
  → `TrainBatch` is the canonical data path. online / offline / disaggregated converge at
  `SampleRef`.
- **Front-end = the redesign draft's domain layer.** `TargetEngine` / `DraftTrainStrategy` /
  `Trainer` / `CheckpointManager` / `Evaluator` / config / eval / export / cli become the
  user-facing surface. The **canonical "stream" is `FeatureDataLoader` over `SampleRef` +
  `FeatureStore`** — we do **not** re-introduce a `HiddenStateStream` as a separate source of
  truth (the loader already plays that role). The online/offline/disaggregated variation lives
  in the *ref source* (`OfflineManifestReader` / `RolloutWorker` / `StreamingRefChannel`) + the
  *transport* (`FeatureStore`: Local for colocated, Mooncake/SharedDir for isolated pools), and
  is shielded from training by `FeatureDataLoader → TrainBatch`. (`RemoteStream`-over-HTTP is
  just one more ref source / engine — the light cross-node option, not the base abstraction.)

This is, in the predecessor's own terms, pulling **feature #32 ("Mooncake streaming backend")
forward from Phase 7 to now** because the requirement demands it — and implementing it with the
already-built `runtime/` work, behind the clean domain interfaces.

**Two scope decisions that bound this plan** (consolidating with the online-disaggregation
roadmap, #618):

1. **Frozen target — no weight sync.** "Train-with-decode" means a *frozen* target streams
   hidden states; the draft is never in the generation loop. Weight sync / hot draft-update /
   weight-version registry / on-policy are **out of scope**; `draft_weight_version` is provenance
   only. (This removes the predecessor's W4 weight-lifecycle workload.)
2. **Ray is an open decision, not a non-goal.** It is a *candidate* for the O2 scale-out
   orchestrator (multi-node N-producer/M-trainer) — likely necessary, but not committed. Until the
   decision gate fires we keep the home-grown metadata-only control plane and add nothing.

**Per-phase target/implementation detail lives in [`docs/roadmap/`](docs/roadmap/)** (domain,
online-disaggregation [folds in #618], eval & breadth); this document is the architecture; the
roadmap is the build order.

---

## 1. Why each side is right (and where each is incomplete)

Two orthogonal slicings. The runtime is sliced by **plane** (how bytes/metadata move); the
draft is sliced by **domain object** (draft / target / stream / strategy). They compose; they
do not conflict.

| Layer | Verdict | Rationale |
|---|---|---|
| **control plane** (`runtime/control_plane`) | **Keep — canonical. No predecessor equivalent.** | Metadata-only (every entrypoint runs `assert_no_tensors`), lease/ack/dedup/reconcile/backpressure behind a `MetadataStore` seam. This is exactly the cross-pool machinery the isolated-pool requirement needs, and the HTTP design has nothing like it. |
| **data plane** (`runtime/data_plane`) | **Keep — canonical; it *is* the stream.** | `SampleRef` (metadata) ÷ `FeatureStore` (tensors) ÷ `FeatureDataLoader` (materialize) is a *finer* decomposition than the predecessor's coarse `HiddenStateStream`, and it's the better substrate for backpressure / lease-ack / resume / Mooncake. `FeatureDataLoader` over `SampleRef`+`FeatureStore` already plays the role plan.md gave `HiddenStateStream`, so **no separate `HiddenStateStream` source-of-truth is introduced** — `seek()`/prefetch land on the loader/ref-source; `FeatureStore` is the transport swap point. |
| **inference plane** (`runtime/inference`) | **Converge to the predecessor's `TargetEngine`.** | Today it conflates "wrap a target engine" + "capture" inside `SGLangAdapter`, and is bound to `generate_eagle3_data` / EAGLE3 names. The `FeatureSource` Protocol already exists (good); the gap is a real `TargetEngine` abstraction (hf / sglang / **sglang_server** / custom) + de-EAGLE3-ifying. This is the layer that most borrows the draft. |
| **training plane** (`runtime/training`) | **Keep the seam; fill the managers.** | `DraftTrainStrategy` / `TrainerCore` / `TrainingBackend` (+ the `StepContext` added for Domino) is *already* the predecessor's `training/strategies` shape — keep it. What's missing are the **managers**, not the seam: `no_sync()` accumulation, full optimizer/scheduler/RNG resume, `CheckpointManager` (rotation/best), `Evaluator`. |

**Landed (merged — the spine):**

- `runtime/` planes (M1–M7/O1): control/data/inference/training, `SampleRef`, `FeatureStore`
  (Local/SharedDir/Mooncake), `StreamingRefChannel`, durable `MetadataStore`, disagg
  producer/consumer + interleaved online loop.

**In the stacked PRs #627/#628/#629 (validated, in review — not yet merged):**

- **Composable launch** (`StrategySpec` registry + parameterized `launch.py`): adding a model
  is a spec entry, not a `build_*_runtime` family. eagle3 / **dflash** / **domino** all train
  end-to-end through one strategy-parameterized path (validated: 197 `tests/test_runtime` OK on
  H200). Domino added `StepContext{global_step, total_steps}` threaded through `forward_loss` —
  the one deliberate contract extension for schedule-dependent loss.

**Explicitly not yet implemented in `runtime/`** (flagged in-source — `contracts.py`,
`trainer.py`, `controller.py`, both `DESIGN.md`s, `runtime/README.md`): live frozen-target online
capture from a real SGLang server (O1.3), full optimizer/scheduler resume, and `no_sync()`
accumulation. These are the §3 gaps. (The in-source weight-publication NOTEs — `WeightVersion` /
`WeightPublisher` / `update_draft_weights` — are **descoped**: the target is frozen, see §8.)

---

## 2. Target architecture

### 2.1 Canonical data path (unchanged from the runtime, this is the spine)

```
ref source            ─┐
  OfflineManifestReader │  (offline: re-iterable refs)
  RolloutWorker         │  (online: produced into a SampleRefQueue)
  StreamingRefChannel   │  (disaggregated: cross-process/pool stream)
                        ▼
DataFlowController  ── metadata only (SampleRef), assert_no_tensors
                        │
FeatureStore  ── tensors only: LocalFeatureStore (mem://, colocated)
                              SharedDirFeatureStore (shared mount)
                              MooncakeFeatureStore (RDMA, isolated pools)
                        ▼
FeatureDataLoader (per_sample_transform + collate) ─► TrainBatch
                        ▼
DraftTrainStrategy.forward_loss(batch, ctx)  ─► TrainerCore / TrainerController ─► FSDP
```

### 2.2 Domain layer on top (user-facing abstractions over the substrate)

The substrate's source of truth is `SampleRef` + `FeatureStore` + `FeatureDataLoader` (§2.1).
The domain layer does **not** re-introduce a `HiddenStateStream` as a parallel source of truth —
`FeatureDataLoader` already *is* the stream. The domain layer is what user/training code sees:

```
PRODUCE (inference):
  TargetEngine        ── the engine rollout wraps to produce features; the abstraction that
                         replaces the EAGLE3-bound SGLangAdapter.
     HFTargetEngine / SGLangTargetEngine / CustomTargetEngine (in-process)
     SGLangServerEngine (frozen target as a live SGLang server, cross-node). ONE engine, two
       feature transports (the engine is identical; only WHERE features land differs):
         · capture transport — engine-side hook writes hidden states INTO a FeatureStore
           (Mooncake/SharedDir). This is W3 / online O1.3.
         · inline-HTTP transport — features serialized in the HTTP response, no shared store
           (RemoteStream-style). This is the light W3′ path, the one case features do NOT live
           in a FeatureStore.
  RolloutWorker       ── drives a TargetEngine → writes tensors to FeatureStore → commits
                         SampleRef to the control plane. Stays at the domain↔substrate seam.

CONSUME (training):
  Trainer             ── owns the lifecycle: loop / eval / checkpoint. WRAPS the runtime
                         TrainerController/TrainerCore; does NOT replace them. (No weight-sync —
                         the target is frozen, §8.)
  DraftTrainStrategy  ── per-algorithm forward+loss (eagle3 TTT / dflash block / domino). Kept
                         in runtime/training (the seam) — already plan-shaped.
  CheckpointManager / Evaluator / lr_scheduler / fsdp seam  ── the managers (G1).

COMPOSE:
  models/drafts       ── DRAFT_REGISTRY (@register_draft) for draft *architecture* classes
                         (llama / deepseek-MLA / dflash-qwen3). Separate axis from strategy.
  StrategyRegistry    ── per-algorithm spec (today's StrategySpec, converged here).
  config / cli / export ── first-class run surface (predecessor §4.3–4.8, carried forward).
```

The online / offline / disaggregated distinction is **not visible to `Trainer`/strategy** — it
is fully absorbed by (ref source + `FeatureStore`) behind `FeatureDataLoader → TrainBatch`.

### 2.3 Module layout (runtime spine + domain layer)

```
specforge/
├── runtime/                         # CANONICAL DataFlow spine (keep)
│   ├── contracts.py                 # SampleRef, TrainBatch, PromptTask, *Strategy literal
│   ├── control_plane/               # metadata-only: controller, metadata_store, backpressure
│   ├── data_plane/                  # FeatureStore (Local/SharedDir/Mooncake), loader, readers,
│   │                                #   SampleRefQueue, StreamingRefChannel
│   ├── inference/                   # rollout: RolloutWorker, capture, adapters  ──┐ converge to
│   │                                #                                              │  TargetEngine
│   ├── training/                    # DraftTrainStrategy seam, TrainerCore/Controller, backend,
│   │                                #   StepContext. (The StrategySpec registry converges into the
│   │                                #   domain training/strategies/ in Phase E — see §6.)
│   └── launch.py                    # spec-driven builders (topology = builder, model = strategy=)
│
├── models/                          # TARGET layout (today the draft/target code lives under
│   │                                #   specforge/modeling/ — Phase E moves it here)
│   ├── drafts/                      # DRAFT_REGISTRY + @register_draft (NEW — predecessor §4.2)
│   │   ├── base.py  llama3_eagle.py  deepseek_eagle3.py(MLA)  dflash.py  auto.py
│   │   │                            #   (today: modeling/draft/{base,dflash,llama3_eagle,flex_attention}.py)
│   └── targets/                     # TargetEngine ABC, EXTRACTED from modeling/target/*TargetModel
│       │                            #   (runtime/inference adapters then wrap an engine, §G2)
│       ├── base.py  hf_engine.py  sglang_engine.py  sglang_server_engine.py  custom_engine.py
│
│                                    # (NO separate data/streams package — FeatureDataLoader over
│                                    #  SampleRef+FeatureStore IS the stream. Ref sources
│                                    #  (offline/rollout/streaming) live in runtime/data_plane;
│                                    #  live frozen-target capture is just another ref source)
│
├── training/                        # domain lifecycle + managers (WRAPS runtime/training)
│   ├── trainer.py                   # owns loop/eval/checkpoint; delegates the step to runtime
│   │                                #   TrainerCore + DraftTrainStrategy (kept seam)
│   ├── checkpoint.py  lr_scheduler.py  fsdp.py   # NEW managers (§3).
│   ├── strategies/                  # StrategySpec registry converges HERE in Phase E (§6); the
│   │                                #   per-step DraftTrainStrategy seam stays in runtime/training.
│
├── eval/  export/  config/  cli.py  # NEW — carried forward from predecessor §4.4–4.8
└── core/  optimizer.py  tracker.py  distributed.py  # kept verbatim (predecessor §1)
```

> The one real structural move is extracting a `TargetEngine` from the EAGLE3-bound
> `modeling/target/*TargetModel` into `models/targets` (the `runtime/inference` adapters then wrap
> it) plus a thin domain `training/` (Trainer + managers) wrapping the kept `runtime/training`
> seam. The control + data planes stay exactly where they are — they are the substrate, **not**
> re-housed behind a new stream package.

---

## 3. Gaps to close (prioritized) — what the domain layer adds on top of the spine

These are the items the landed `runtime/` does **not** have, drawn from the predecessor and
from the in-source `NOTE`s. Each lands behind the canonical spine without re-plumbing it.

### G1 — Training managers (highest leverage; the spine works but is bare)
- **`no_sync()` accumulation.** `runtime/training` has **zero** `no_sync()` — FSDP all-reduces
  every micro-step, defeating `accumulation_steps`. (Predecessor §4.2 #10; verify with one
  profiler check: one all-reduce per `optimizer.step()`.)
- **Full resume.** `save_checkpoint` persists training state only; no optimizer/scheduler/RNG
  restore, no rotation, no best-tracking. Add `CheckpointManager` (predecessor §4.3) + a
  `seek()`-equivalent on the colocated streams.
- **`Evaluator` + `EvalCache`.** No `simulated_acc_len` / per-position-acc / best-checkpoint.
  (Predecessor §4.4 — per-position acc aggregated across batches *before* the geometric sum.)
- **lr WSD + per-strategy `fsdp_wrap_policy()`** (predecessor §4.6, §4.2).

### G2 — `TargetEngine` abstraction (inference convergence)
- Introduce a `TargetEngine` ABC **extracted from the `modeling/target/*TargetModel` classes**
  (`Eagle3TargetModel`, `DFlashTargetModel`); the `runtime/inference` adapters
  (`SGLangAdapter`/`DFlashAdapter`) then **wrap** a `TargetEngine` rather than being the engine,
  and stop binding to `generate_eagle3_data` / EAGLE3 names. Keep the existing `FeatureSource`
  Protocol. Add `SGLangServerEngine` (live SGLang server) as the cross-node engine — see §2.2 for
  its two feature transports (capture-into-FeatureStore for W3/O1.3, inline-HTTP for the light
  W3′).

### G3 — Live online capture (frozen target; **no** weight sync)
- Replace the in-process generator with **live SGLang-server hidden-state capture**: a *frozen*
  target streams aux+final hidden states into `MooncakeFeatureStore`; the producer commits
  `SampleRef`s. The cross-process control plane + async loop are in-review; live capture (the
  gating spike) is next; scale-out (Ray = **open**) and hardening follow.
- **Weight sync / hot draft-update / on-policy are explicitly out of scope.** The target is
  frozen, so the streamed data is independent of draft weights — there is nothing to re-sync and
  no staleness. `draft_weight_version` is kept **only as provenance**.
- Detailed phases (O1–O3): [`docs/roadmap/online-disaggregation.md`](docs/roadmap/online-disaggregation.md).

### G4 — Composition + models (predecessor Phase 1)
- `models/drafts` `DRAFT_REGISTRY` (`@register_draft`) for draft **architectures** (separate
  axis from strategy); **MLA Eagle3 draft** (deepseek/Kimi); converge `StrategySpec` →
  `training/strategies` registry. Note: draft-arch registry and strategy registry are two
  registries, not one.

### G5 — Run surface (predecessor Phase 3)
- Pydantic `config/` + `specforge` CLI; `export/to_sglang` (+ the documented MLA weight-name
  map) and `export/to_hf` with vocab pruning.

---

## 4. Topologies & the colocated lightweight path

The workloads differ only in which *ref source* + `FeatureStore` + `TargetEngine` compose. All
consume **one** `FeatureDataLoader → TrainBatch` iterator; the trainer/strategy/backend are
identical regardless (no per-topology stream class). **In every case the target is frozen** — the
draft is never in the generation loop, so the streamed features do not depend on draft training
progress (no staleness, nothing to re-sync).

| # | Workload | Ref source (→ FeatureDataLoader) | FeatureStore | Control plane |
|---|---|---|---|---|
| W1 | Offline (precomputed) | `OfflineManifestReader` (refs) | Local (`file://`/`mem://`) | **no-op** |
| W2 | In-process online (frozen target) | `RolloutWorker` → `SampleRefQueue` | Local (`mem://`) | **no-op** |
| W3 | Disaggregated online (frozen target; isolated pools, high BW) | `RolloutWorker` (producer pool) → `StreamingRefChannel`, `SGLangServerEngine` *capture transport* | **Mooncake** | **active** (lease/ack/reconcile/backpressure) |
| W3′ | Disaggregated online (light/cross-node) | `SGLangServerEngine` *inline-HTTP transport* (RemoteStream-style source) | n/a — features inline over HTTP, no shared store | minimal |

> **"Train-with-decode" = live *frozen-target* generation** — i.e. W2 (colocated) or W3
> (disaggregated), **not** a separate workload. The predecessor's dual-purpose
> serve-and-push-weights "W4" is **out of scope**: a frozen target means there is nothing to push.
> See [`docs/roadmap/online-disaggregation.md`](docs/roadmap/online-disaggregation.md).

**Colocated lightweight path (W1/W2): keep it, but as a *no-op control plane*, not a fork.**
The control-plane machinery (lease/ack/reconcile/backpressure/durable metadata/cross-process
transport) only earns its keep when producer and consumer are decoupled. For W1/W2 (same
process) it is pure overhead. Resolution:

- **Single canonical path** through `SampleRef` + `FeatureStore`, so colocated and disaggregated
  produce **byte-identical batches for free** (today's property — "disaggregation changes *where*
  features live, not their values"). We do **not** fork a second trainer/launch path.
- "Lightweight" = for colocated, the controller's lease/ack/metadata-store are **opt-in / no-op**,
  `FeatureStore` is `LocalFeatureStore(mem://)`, and there is **no SQLite, no cross-process, no
  backpressure**. The colocated ref sources (`OfflineManifestReader` / in-process `RolloutWorker`)
  flow through the spine but skip the heavy bits.
- Cost to own honestly: this preserves the free byte-identical guarantee **only if** we keep one
  path. A numerical-equivalence gate (§7) locks "colocated stream output == disagg stream output"
  so the no-op path can never silently diverge.

---

## 5. What changed from the predecessor draft (explicit deltas)

1. **"No Mooncake in Phase 1-6" → revised.** The Mooncake control/data plane is now the
   **canonical** disaggregation backend, because the isolated-pool / >100 GB/s requirement is
   real and HTTP cannot serve it. The predecessor's `RemoteStream`-over-HTTP remains as the
   **light** second backend (W3′). Mooncake is used as a *transport*; the scale-out
   **orchestration** layer (Ray vs. home-grown) is an **open decision** (see §6, §8 and
   [`docs/roadmap/online-disaggregation.md`](docs/roadmap/online-disaggregation.md) §O2), no longer
   a flat non-goal.
2. **Primary abstraction / `HiddenStateStream` dropped.** Predecessor: a coarse
   `HiddenStateStream` produces `TrainBatch` and is the source of truth. Reconciled:
   `SampleRef` + `FeatureStore` + `FeatureDataLoader` is canonical, and `FeatureDataLoader`
   **already is** the stream — so `HiddenStateStream` is **not introduced** as a parallel
   abstraction. Topology variation lives in (ref source + `FeatureStore`); `seek()`/prefetch
   land on the loader/ref-source. This is strictly finer and is what makes
   lease/ack/backpressure/Mooncake clean.
3. **`runtime/` is not "thrown away."** The predecessor's "what we throw away" never mentioned
   `runtime/` (it predated it). The DataFlow spine is the foundation, not debt.
4. **Module layout** gains a top-level `runtime/` (the spine) under the domain layer; `models/`
   (drafts + targets), `training/` (Trainer + managers), `eval/`, `export/`, `config/`, `cli.py`
   are the layer on top. No separate `data/streams/` package — the loader is the stream.
5. **W4 weight lifecycle → dropped.** The predecessor's serve-and-push-weights workload + weight
   registry are **out of scope**. "Train-with-decode" is *frozen-target* live generation (W2/W3);
   weight sync / hot draft-update / staleness gate / on-policy are explicitly cut, aligning with
   #618. `draft_weight_version` survives only as provenance metadata.
6. **Ray reframed from non-goal → open.** It is a *candidate* for the O2 scale-out orchestrator
   (multi-node N-producer/M-trainer), not committed and not forbidden; the decision gate lives in
   the roadmap.

Everything else in the predecessor (workloads W1–W3, the domain abstractions, the testing
discipline, the MLA/export/eval/config detail) **carries forward unchanged**.

---

## 6. Tradeoffs (updated)

### Mooncake transport vs HTTP/gRPC — *transport bet reversed*
The predecessor bet "HTTP is sufficient, Mooncake is gated behind profiling." For the **data
transport** that bet is now wrong (isolated-pool / >100 GB/s requirement) — Mooncake is in,
canonically. The `RemoteStream`-over-HTTP path stays as the light cross-node backend (W3′).

### Scale-out orchestration: Ray vs. home-grown — *OPEN*
Today's metadata-only control plane handles a single producer-pool ↔ trainer scope. **O2**
(multi-node N-producer/M-trainer scale-out) needs an orchestration layer, and **whether that is
Ray or a home-grown scheduler is undecided** — likely necessary, but not committed. It is **not**
a non-goal anymore; the decision gate (when ≥1 of: >1 producer pool, cross-node failover,
multi-job pool sharing) lives in
[`docs/roadmap/online-disaggregation.md`](docs/roadmap/online-disaggregation.md) §O2. Until then
we keep the home-grown control plane and add nothing.

### One canonical path vs a light colocated fork
Chosen: **one path** (spine everywhere) + control-plane-as-no-op for colocated. Keeps
byte-identical free and avoids two implementations; costs an equivalence gate. (See §4.)

### Two registries, not one
`models/drafts` `DRAFT_REGISTRY` (architecture) and `training/strategies` (algorithm) are
**separate** axes — an algorithm (eagle3) runs on multiple draft architectures. Don't merge
them; the current `StrategySpec` is the *strategy* registry and should converge there, not into
`models/drafts`.

*(The predecessor's other tradeoffs — Pydantic vs OmegaConf, MLA cache compressed vs expanded,
registry vs HF AutoModel — are unchanged and carry forward.)*

---

## 7. Migration path (incremental, gated)

The spine is already landed, so migration is "grow the domain layer on top + close the gaps,"
not a rewrite. Every phase that touches the training path passes the numerical-equivalence gate
(below) against the prior commit. **Per-phase target/implementation/tests/done-when detail lives
in [`docs/roadmap/`](docs/roadmap/)** — the phases below are the index; the online track there
also folds in the former online-disaggregation roadmap (#618).

- **Phase A — composable launch (in review).** `StrategySpec` registry + parameterized
  `launch.py`; eagle3/dflash/domino end-to-end in the stacked PRs #627/#628/#629 (validated,
  in review). 197 `tests/test_runtime` OK.
- **Phase B — domain abstractions (no behavior change).** Introduce `TargetEngine` (wrap the
  existing adapters; de-EAGLE3 the names) and the domain `Trainer` wrapping the kept
  `runtime/training` core. No `HiddenStateStream` — `FeatureDataLoader` stays the stream. Gate:
  byte-identical batches/loss vs the pre-refactor path.
- **Phase C — colocated no-op control plane + equivalence gate.** Make lease/ack/metadata
  opt-in; add the colocated≡disagg equivalence test.
- **Phase D — training managers (G1).** `no_sync()`, full optimizer/scheduler/RNG resume,
  `CheckpointManager`, `Evaluator`. Gate: loss/eval parity at fixed steps.
- **Phase E — `models/drafts` registry + MLA draft + config/CLI/export (G4/G5).** Adding a
  draft arch = one decorated file; one validated YAML per run; export-loop test.
- **Online track (parallel; G3) — live *frozen-target* capture.** O1.1 shared control plane +
  O1.2 async loop (in review) → **O1.3** live SGLang-server hidden-state capture (next; gated by a
  throughput spike) → **O2** scale-out (orchestrator: Ray = open) → **O3** hardening. **No weight
  sync.** Detail: [`docs/roadmap/online-disaggregation.md`](docs/roadmap/online-disaggregation.md).
- **Eval track (parallel) — E1** acceptance-length eval harness → **E2** algorithm breadth (new
  algo = a `StrategySpec` + loss). Detail: [`docs/roadmap/eval-and-breadth.md`](docs/roadmap/eval-and-breadth.md).

Doc debt to fix alongside Phase B: revise the predecessor's "No Mooncake / HTTP is sufficient"
statements (now §5/§6) so the code and the plan stop contradicting each other.

---

## 8. Non-goals (updated)
- **No weight sync / hot draft-update / on-policy training.** The target is **frozen**; the
  draft is never in the generation loop. No `WeightPublisher`, no weight-version registry, no
  staleness gate. `draft_weight_version` is provenance metadata only.
- **No two-stack fork for colocated.** One canonical data path; colocated is the spine with the
  control plane as a no-op, not a parallel implementation.
- **No vLLM target backend** (possible via `TargetEngine`, not prioritized).
- **No multi-engine load balancing / multi-job inference-pool sharing** for now — gated behind
  ≥5 concurrent jobs sharing one target.

**Open decisions (NOT non-goals):**
- **Ray (or a home-grown scheduler) for O2 scale-out** — likely necessary for multi-node
  N-producer/M-trainer; undecided. Decision gate in §6 / the online roadmap.

*(Reversed from the predecessor: "No Mooncake" → Mooncake transport is now canonical for W3;
"No Ray" → Ray is now an open scale-out decision, not a flat non-goal.)*

---

## 9. Success criteria
| Area | Criterion |
|---|---|
| Composable launch (in review) | eagle3/dflash/domino train via one strategy-parameterized path; full `test_runtime` green. |
| Abstractions (B) | `TargetEngine` (wrapping existing adapters) + domain `Trainer` produce byte-identical batches/loss vs the direct spine path. |
| Colocated (C) | W1/W2 run with control plane as no-op; colocated≡disagg equivalence gate passes. |
| Managers (D) | resume reproduces the no-resume loss curve; one all-reduce per optimizer step; best-checkpoint tracked. |
| Drafts/MLA (E) | MLA Eagle3 trains + loads in SGLang; new draft arch = one `@register_draft` file. |
| Online (O1.3) | a live **frozen-target** SGLang server feeds training with zero precomputed features; loss/eval matches the offline baseline on the same prompts+seed. (Scale-out O2 / Ray = open.) |

---

## 10. Testing strategy (gating)

Carries the predecessor's gates, plus the reconciliation-specific one.

- **10.1 Numerical-equivalence gate** (per PR touching `core/` / `runtime/training/` /
  `runtime/data_plane/` / `models/drafts`): fixed seed, 3×4 batches; per-step loss
  `atol/rtol=1e-4` at steps 0/1/100/500/1000; per-position eval acc + `simulated_acc_len`
  `atol=1e-3`.
- **10.2 Colocated ≡ disaggregated gate (NEW):** same data through the colocated (no-op control
  plane, Local store) and the disaggregated (control plane, SharedDir/Mooncake) paths must yield
  identical `TrainBatch`es and identical loss. This is what licenses the lightweight path.
- **10.3 Smoke tests:** each draft arch trains 20 steps under TP=1 / TP=2 / TP=2+SP=2;
  checkpoint save+resume matches no-resume (validates `seek`); eval cache miss==hit.
- **10.4 Distributed correctness:** MLA + Yunchang USP with asymmetric head dims; gradient
  accumulation = one all-reduce per `optimizer.step()`.
- **10.5 Export-loop test:** train MLA draft 100 steps → export → load in SGLang → 32 gens,
  acceptance > 0 (catches weight-name-map regressions).
- **10.6 Online-capture parity gate (O1.3):** a live frozen-target capture run produces
  features + loss matching the offline-precomputed baseline on the same prompts/seed (the target
  is frozen, so this must hold exactly up to nondeterminism tolerance). No weight-sync gate —
  weight sync is out of scope.
