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

**Already landed (the spine + first cleanup):**

- `runtime/` planes (M1–M7/O1): control/data/inference/training, `SampleRef`, `FeatureStore`
  (Local/SharedDir/Mooncake), `StreamingRefChannel`, durable `MetadataStore`, disagg
  producer/consumer + interleaved online loop.
- **Composable launch** (`StrategySpec` registry + parameterized `launch.py`): adding a model
  is a spec entry, not a `build_*_runtime` family. eagle3 / **dflash** / **domino** all train
  end-to-end through one strategy-parameterized path (PRs #627 / #628 / #629, validated:
  197 `tests/test_runtime` OK on H200). Domino added `StepContext{global_step, total_steps}`
  threaded through `forward_loss` — the one deliberate contract extension for schedule-dependent
  loss.

**Explicitly not yet implemented in `runtime/`** (flagged in-source — `contracts.py`,
`trainer.py`, `controller.py`, both `DESIGN.md`s, `runtime/README.md`): the weight-publication
lifecycle (`WeightVersion`, `WeightPublisher`, `update_draft_weights`, serving accept-length
gate), full optimizer/scheduler resume, and `no_sync()` accumulation. These are the §3 gaps.

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
     SGLangServerEngine (HTTP service; owns update_draft_weights + serving_metrics for W4)
  RolloutWorker       ── drives a TargetEngine → writes tensors to FeatureStore → commits
                         SampleRef to the control plane. Stays at the domain↔substrate seam.

CONSUME (training):
  Trainer             ── owns the lifecycle: loop / eval / checkpoint / weight-sync. WRAPS the
                         runtime TrainerController/TrainerCore; does NOT replace them.
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
│   ├── training/                    # DraftTrainStrategy, TrainerCore/Controller, backend,
│   │                                #   StepContext, strategy registry
│   └── launch.py                    # spec-driven builders (topology = builder, model = strategy=)
│
├── models/
│   ├── drafts/                      # DRAFT_REGISTRY + @register_draft (NEW — predecessor §4.2)
│   │   ├── base.py  llama_eagle3.py  deepseek_eagle3.py(MLA)  dflash_qwen3.py  auto.py
│   └── targets/                     # TargetEngine ABC (NEW — absorbs runtime/inference adapters)
│       ├── base.py  hf_engine.py  sglang_engine.py  sglang_server_engine.py  custom_engine.py
│
│                                    # (NO separate data/streams package — FeatureDataLoader over
│                                    #  SampleRef+FeatureStore IS the stream. Ref sources
│                                    #  (offline/rollout/streaming) live in runtime/data_plane;
│                                    #  ServingTrafficStream (W4) is just another prompt/ref source)
│
├── training/                        # domain lifecycle + managers (WRAPS runtime/training)
│   ├── trainer.py                   # owns loop/eval/checkpoint/weight-sync; delegates the step
│   │                                #   to runtime TrainerCore + DraftTrainStrategy (kept seam)
│   ├── checkpoint.py  lr_scheduler.py  fsdp.py   # NEW managers (§3). Strategies + StrategySpec
│   │                                #   registry stay in runtime/training (the seam, unchanged).
│
├── eval/  export/  config/  cli.py  # NEW — carried forward from predecessor §4.4–4.8
└── core/  optimizer.py  tracker.py  distributed.py  # kept verbatim (predecessor §1)
```

> The one real structural move is `runtime/inference` → `models/targets` (extract a
> `TargetEngine` from the EAGLE3-bound adapter) plus a thin domain `training/` (Trainer +
> managers) wrapping the kept `runtime/training` seam. The control + data planes stay exactly
> where they are — they are the substrate, **not** re-housed behind a new stream package.

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
- Introduce `TargetEngine` ABC; refactor `SGLangAdapter`/`DFlashAdapter` to wrap a
  `TargetEngine` and stop binding to `generate_eagle3_data` / EAGLE3 names. Keep the existing
  `FeatureSource` Protocol. Add `SGLangServerEngine` (HTTP) as the light cross-node engine.

### G3 — W4 weight lifecycle (explicit gap, flagged in `runtime/` source)
- `WeightVersion`, `WeightPublisher`, `TargetEngine.update_draft_weights`,
  `SGLangServerEngine` decode-mode + `serving_metrics()`, `ServingTrafficStream`, and the
  serving accept-length gate (predecessor §4.10). One trainer ↔ one server scope.

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

The four workloads are unchanged; they differ only in which *ref source* + `FeatureStore` +
`TargetEngine` compose. All consume **one** `FeatureDataLoader → TrainBatch` iterator; the
trainer/strategy/backend are identical regardless (no per-topology stream class).

| # | Workload | Ref source (→ FeatureDataLoader) | FeatureStore | Control plane |
|---|---|---|---|---|
| W1 | Offline (precomputed) | `OfflineManifestReader` (refs) | Local (`file://`/`mem://`) | **no-op** |
| W2 | In-process online | `RolloutWorker` → `SampleRefQueue` | Local (`mem://`) | **no-op** |
| W3 | Disaggregated online (isolated pools, high BW) | `RolloutWorker` (producer pool) → `StreamingRefChannel` | **Mooncake** | **active** (lease/ack/reconcile/backpressure) |
| W3′ | Disaggregated online (light/cross-node) | `SGLangServerEngine` over HTTP (RemoteStream-style source) | n/a (HTTP) | minimal |
| W4 | Train-with-decode | `RolloutWorker` / `ServingTrafficStream` + `SGLangServerEngine` | per above | per above |

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
   **light** second backend (W3′), and the "Ray actor topology" non-goal still holds (we use
   Mooncake transport, not Ray scheduling).
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
5. **W4 weight lifecycle** is reframed from "additive HTTP-only" to "an interface that has both
   an HTTP (`SGLangServerEngine`) and, where needed, a Mooncake-backed implementation."

Everything else in the predecessor (workloads W1–W4, the domain abstractions, the testing
discipline, the MLA/export/eval/config detail) **carries forward unchanged**.

---

## 6. Tradeoffs (updated)

### Ray + Mooncake vs HTTP/gRPC  — *bet partially reversed*
The predecessor bet "HTTP is sufficient, Mooncake is gated behind profiling." For the **data
transport** that bet is now wrong (isolated-pool / >100 GB/s requirement) — Mooncake is in,
canonically. For **scheduling/orchestration**, the bet still holds: we use Mooncake as a
*transport*, with our own metadata-only control plane, **not** Ray actors. Multi-job inference-
pool sharing (Ray's real value, predecessor T2 / #34) stays a non-goal until ≥5 concurrent jobs
share one target.

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
(below) against the prior commit.

- **Phase A — composable launch (DONE / in review).** `StrategySpec` registry + parameterized
  `launch.py`; eagle3/dflash/domino end-to-end (#627/#628/#629). 197 `tests/test_runtime` OK.
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
- **Phase F — W4 weight lifecycle (G3).** `update_draft_weights`, `SGLangServerEngine`
  decode-mode + serving metrics, `ServingTrafficStream`, accept-length gate.

Doc debt to fix alongside Phase B: revise the predecessor's "No Mooncake / HTTP is sufficient"
statements (now §5/§6) so the code and the plan stop contradicting each other.

---

## 8. Non-goals (updated)
- **No Ray dependency / no multi-engine load balancing.** One trainer ↔ one serving engine
  (incl. W4). Multi-job pool sharing is gated behind ≥5 concurrent jobs.
- **No two-stack fork for colocated.** One canonical data path; colocated is the spine with the
  control plane as a no-op, not a parallel implementation.
- **No vLLM target backend** (possible via `TargetEngine`, not prioritized).
- **No in-trainer serving-prompt interception** — `ServingTrafficStream` reads an external
  buffer the serving cluster populates.

*(Reversed from the predecessor: "No Mooncake" — Mooncake transport is now canonical for W3.)*

---

## 9. Success criteria
| Area | Criterion |
|---|---|
| Composable launch (done) | eagle3/dflash/domino train via one strategy-parameterized path; full `test_runtime` green. |
| Abstractions (B) | `TargetEngine` (wrapping existing adapters) + domain `Trainer` produce byte-identical batches/loss vs the direct spine path. |
| Colocated (C) | W1/W2 run with control plane as no-op; colocated≡disagg equivalence gate passes. |
| Managers (D) | resume reproduces the no-resume loss curve; one all-reduce per optimizer step; best-checkpoint tracked. |
| Drafts/MLA (E) | MLA Eagle3 trains + loads in SGLang; new draft arch = one `@register_draft` file. |
| W4 (F) | one server serves spec-decode traffic + produces training data; weight push every N steps; serving acceptance rate trends up; weight-sync parity gate passes. |

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
- **10.6 Weight-sync correctness gate (W4):** after `update_draft_weights`, served draft logits
  match trainer draft within tolerance; serving acceptance non-decreasing across a sync.
