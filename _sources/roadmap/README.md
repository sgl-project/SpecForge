# SpecForge Roadmap

The consolidated, phase-by-phase roadmap behind [`../../plan.md`](../../plan.md) (the reconciled
architecture). It **folds in** the former online-disaggregation roadmap PR (#618) so there is one
roadmap home. Each track doc gives, per phase: **Goal / Target state / Implementation
(files + symbols) / Tests / Done-when**.

## Standing decisions (apply across all tracks)
- **Substrate is canonical.** `SampleRef` (metadata, control plane; `assert_no_tensors`) +
  `FeatureStore` (tensors: Local/SharedDir/Mooncake) + `FeatureDataLoader` → `TrainBatch`. There is
  **no separate `HiddenStateStream`** source of truth — the loader *is* the stream; online/offline/
  disaggregated vary only in (ref source + `FeatureStore`), shielded from training.
- **Frozen target, no weight sync.** "Train-with-decode" = a **frozen** target streams hidden
  states over a **fixed** prompt set; the draft is not in the generation loop. Weight-sync /
  hot draft-update / weight-version registry / staleness gate / on-policy are **out of scope**;
  `draft_weight_version` is kept **only as provenance**.
- **Ray is OPEN.** A *candidate* for the O2 scale-out orchestration layer — likely needed for
  multi-node N-producer/M-trainer scale-out — but **not committed and not a non-goal**. See the
  decision gate in [online-disaggregation.md](./online-disaggregation.md) §O2.
- **Preserve the training seam** (`TrainerCore` / `DraftTrainStrategy` / `TrainingBackend` +
  `StepContext`); a domain `Trainer` + managers *wrap* it, they do not replace it. It is relocated
  **intact** (not rewritten) from `runtime/training` to top-level `training/` in the move-only
  step `E0` — see [domain-refactor.md](./domain-refactor.md).
- **One implementation home per concern; `runtime/` is substrate-only.** `runtime/` holds only the
  DataFlow spine (`control_plane` + `data_plane` + `contracts`). All training-execution code lives
  in top-level `training/`, all rollout/capture-execution code in top-level `inference/`, and
  `modeling/` holds model definitions only (no orchestration, no capture factory). **New code is
  born in its final home** — the Phase-D managers land directly in `training/`, never deeper in
  `runtime/`; the existing seam and target engine are relocated once, in `E0`. There is **no facade
  package**.

## Tracks
| Track | Doc | Scope |
|---|---|---|
| Domain / architecture | [domain-refactor.md](./domain-refactor.md) | Strategy/registry, `TargetEngine`, domain `Trainer` + managers, drafts registry, config/CLI/export |
| Online disaggregation | [online-disaggregation.md](./online-disaggregation.md) | Live frozen-target generation, cross-process control plane, scale-out (Ray = open), hardening |
| Eval & breadth | [eval-and-breadth.md](./eval-and-breadth.md) | Acceptance-length eval harness; new algorithms = a `StrategySpec` + loss |

## Phase status at a glance
| Phase | Track | Size | Status |
|---|---|---|---|
| A — Composable launch | domain | L | **in review** (#627/#628/#629) |
| O1.1 — Shared cross-process control plane | online | M | in review (~#624) |
| O1.2 — Async streaming loop + one-process builder | online | M | in review (~#625) |
| B — Domain abstractions (`TargetEngine` + `Trainer`) | domain | L | next |
| O1.3 — Live SGLang-server hidden-state capture | online | L | next (🔴 spike gates it) |
| E1 — Acceptance-length eval harness | eval | M | next |
| C — Colocated lightweight path | domain | M | later |
| D — Training managers (no_sync / resume / ckpt / eval) | domain | L | later |
| E0 — Layout consolidation (move-only: seam + target engine → top-level homes) | domain | M | later (front of E) |
| E — Composition & run surface (drafts registry, config/CLI, export) | domain | L | later |
| O2 — Scale-out orchestration (Ray = open) | online | L | later |
| O3 — Hardening (RDMA pool, restart, observability) | online | L | later |
| E2 — Algorithm breadth (MTP/…) | eval | L | later |

## Dependencies (cross-track)
```
domain:  A(rev.) ─┬─▶ B ─┬─▶ C
                  │       └─▶ D ─▶ E0 ─▶ E
                  └─▶ B.TargetEngine ─────────────┐
online:  O1.1(review) ─▶ O1.2(review) ─▶ O1.3 ◀───┘ ─▶ O2(Ray=open) ─▶ O3
eval:    E1 ─▶ E2          (parallel / orthogonal to both tracks)
```
- Domain **B** (`TargetEngine`) unblocks online **O1.3** (a real `SGLangServerEngine` capture
  backend), so it is the highest-leverage next domain step.
- **E1**'s `Evaluator` is the same one domain **D** wires into the trainer loop — build once.
