# Domain / Architecture Refactor Track

> **Historical planning record.** The layout consolidation, typed config, and
> single `specforge train --config ...` surface described by this track are now
> implemented. This file preserves design rationale; it is not a usage guide or
> a migration contract. See [Training](../basic_usage/training.md) for the
> current supported interface. The public CLI currently supports resume only
> for single-rank local offline training; every online config rejects
> `training.resume_from`.

This is the **domain/architecture** track — making the runtime support multiple
draft algorithms (eagle3 / dflash / domino / future) and a real product surface,
*orthogonal* to the online-disaggregation scale-out work in
[./online-disaggregation.md](./online-disaggregation.md). It builds on the
canonical substrate (SampleRef + FeatureStore + FeatureDataLoader → TrainBatch),
keeps the runtime training seam (`TrainerCore` / `DraftTrainStrategy` /
`TrainingBackend` + `StepContext`) and only *wraps* it with domain objects. The
online target is FROZEN — there is **no** weight-sync, no `HiddenStateStream`
source of truth, and `draft_weight_version` survives only as provenance metadata.
Sibling tracks: [./online-disaggregation.md](./online-disaggregation.md) and
[./eval-and-breadth.md](./eval-and-breadth.md).

**Historical dependency order:** A → B → {C, D}; D → E0 → E (C is parallel — not a prerequisite
of E). `E0` is a move-only layout consolidation (zero functional change) that runs at the front of
E so E's composition work lands in the final layout. B's `TargetEngine` extraction also unblocks
the online O1.3 multi-backend producer in [./online-disaggregation.md](./online-disaggregation.md).

---

### A — Composable launch · size L · GPU · status: implemented
- **Goal** One strategy-parameterized launch path so adding an algorithm is a
  registry entry, not a new `build_*` family — and so a schedule-dependent loss
  (domino) flows through the same `TrainerCore`.
- **Target state** `eagle3`, `dflash`, `domino` all run end-to-end (offline +
  online) through a single set of topology builders that take `strategy=` and
  resolve a `StrategySpec`. `launch.py` grows as `topologies + one spec per
  model`, never `topologies × models`.
- **Implementation** (landed)
  - `training/registry.py` — `StrategySpec` dataclass + `register_strategy` /
    `resolve_strategy` / `available_strategies`; one entry each for `eagle3`,
    `dflash`, `domino`. Each spec carries `make_strategy`, `required_features`,
    the offline reader/transform/collate triple + `offline_target_repr`, the
    online collate (`concat_collate`), and `make_adapter` (None ⇒ default
    `SGLangAdapter`).
  - `launch.py` — `_assemble_trainer` / `_offline_io` / `_assemble_rollout_workers`
    plus topology builders `build_offline_runtime`, `build_disagg_offline_runtime`,
    `build_online_runtime`, `build_disagg_online_{producer,consumer,runtime}`;
    every one resolves the spec. The final public run surface does not expose
    method-specific builder aliases.
  - `training/strategy.py` — `DraftTrainStrategy` ABC + `Eagle3TrainStrategy`,
    `DFlashTrainStrategy`, `DominoTrainStrategy`; `StepContext`
    (`global_step` / `total_steps`) is threaded into `forward_loss` so domino's
    `_lambda_base` decay reads the schedule without ad-hoc kwargs on the model
    forward.
  - `inference/dflash_adapter.py` — `DFlashAdapter` over `generate_dflash_data`,
    emitting the `{input_ids, hidden_states, loss_mask}` schema (no
    `_project_target` / `t2d`); domino reuses it.
- **Tests / gates** `tests/test_runtime/` incl. `test_dflash_launch.py`,
  `test_dflash_online_launch.py`; full suite **197 tests/test_runtime green on
  H200**.
- **Done when** All three strategies run offline + online via the one parameterized
  path with the suite green (validated, in review).

---

### B — Domain abstractions · size L · GPU · status: implemented
- **Goal** De-EAGLE3 the target boundary and give the runtime a domain `Trainer`
  that *wraps* (does not replace) the runtime controller, so the architecture
  reads in product terms while the seam stays intact.
- **Target state** A backend-agnostic `TargetEngine` ABC sits where the
  EAGLE3-named `Eagle3TargetModel` is today; `hf` / `sglang` / `sglang_server` /
  `custom` are interchangeable backends behind it. A domain `Trainer` is the
  caller-facing object; under it the `TrainerController` / `TrainerCore` /
  `DraftTrainStrategy` / `TrainingBackend` seam is byte-for-byte unchanged. The
  `FeatureSource` Protocol in `rollout_worker.py` stays the worker's only
  contract. **No `HiddenStateStream`** — `FeatureDataLoader` over
  SampleRef+FeatureStore *is* the stream.
- **Implementation**
  1. Extract `TargetEngine` (ABC) from
     `specforge/modeling/target/eagle3_target_model.py:79` (`Eagle3TargetModel`).
     De-EAGLE3 the names: the ABC's extraction method becomes a generic
     `generate_features(...)`-style capture call with EAGLE3 specifics
     (`generate_eagle3_data`, `aux_hidden_states_layers`,
     `set_aux_hidden_states_layers`) moved into an `Eagle3TargetEngine` subclass;
     `DFlashTargetModel` (`dflash_target_model.py:33`, `set_capture_layers`)
     becomes a sibling subclass. Replace
     `get_eagle3_target_model(backend=...)` with the generic target-engine
     factory at the package boundary.
  2. Add an explicit `backend` attribute on each engine (today `SGLangAdapter.health`
     / `DFlashAdapter.health` read `getattr(target_model, "backend", "unknown")`
     — make it real), and add the `sglang_server` backend branch to the factory
     (currently only `sglang` / `hf` / `custom`). The *depth* of this `sglang_server`
     branch is informed by the O1.3 capture spike (see
     [online-disaggregation.md](./online-disaggregation.md) §O1.3): the de-EAGLE3
     extraction (step 1) and the domain `Trainer` carry no engine risk and can land
     regardless of the spike's outcome — only the live-server capture backend does.
  3. Keep the adapters (`SGLangAdapter`, `DFlashAdapter`) as the
     `FeatureSource` implementations over the engine; nothing in
     `rollout_worker.py` changes — its `FeatureSource` Protocol
     (`generate_features(tasks, *, capture)`) is already the seam.
  4. Introduce a domain `Trainer` (new `training/` module) that composes
     `resolve_strategy` → `FSDPTrainingBackend` → `TrainerCore` →
     `TrainerController` (exactly what `launch._assemble_trainer` wires today)
     behind one object, and calls `.fit()`. The controller/core seam is wrapped,
     not edited.
- **Tests / gates** **Byte-identical batches and loss vs the pre-refactor run**
  for all three strategies (snapshot the first-N `TrainBatch.tensors` digests +
  per-step loss before refactor, assert equality after). `tests/test_runtime/`
  stays green; add a `TargetEngine` backend-parity test (hf vs sglang produce the
  same captured features on a fixed prompt set).
- **Done when** The EAGLE3 name no longer appears in the target ABC or the
  trainer-facing API, `sglang_server` is selectable, and the byte-identical gate
  passes.

---

### C — Colocated lightweight path · size M · CPU · status: implemented
- **Goal** Make colocated runs (W1/W2) pay nothing for the disagg control plane,
  on the *same* code path — not a fork.
- **Target state** One canonical path. For colocated, the control plane is
  opt-in / no-op: `LocalFeatureStore` over `mem://`, no SQLite metadata store, no
  lease / ack / backpressure. Online / offline / disagg differ only in (ref
  source + FeatureStore), shielded from training.
- **Implementation**
  - `control_plane/` — make the metadata store (`metadata_store.py`), leasing
    (`controller.py`), and backpressure (`backpressure.py`) selectable as no-op
    implementations rather than required collaborators. Drive the choice from a
    `DeploymentMode` (`contracts.py` already defines
    `local_colocated` / `dataflow_colocated` / `disaggregated`).
  - `launch.py` — in the colocated builders, pass the no-op control plane and the
    `mem://` `LocalFeatureStore` (`data_plane/feature_store.py`); the
    `TrainerController.ack_fn` is `None` (already supported — the loader is
    assumed to ack) so the durable ack transaction is skipped.
  - Keep the trainer/loader code identical between colocated and disagg.
- **Scope note (as landed, #636)** The no-op axis is the *(metadata store +
  durable-ack transaction)* — the pieces with real I/O/txn cost. Prompt leasing
  and the in-process `SampleRefQueue` bookkeeping stay shared across modes
  (single-process lock-guarded dict/deque ops, no I/O; forking them per mode
  would reintroduce the divergence this phase removes), and backpressure was
  already opt-in (`None` default) everywhere. Consequences of a store that
  retains nothing are documented on `NoOpMetadataStore` (`status()` counts read
  0; no `TrainLease` ref reconstruction). `deployment_mode` is selectable on the
  offline builder; the colocated ONLINE builder stays pinned to
  `local_colocated` because its rank-private queue is fed by commit dedup — a
  shared durable store there belongs to the disagg online builders.
- **Tests / gates** A **colocated == disagg numerical-equivalence gate**: same
  seed + fixed feature set produce bit-identical loss curves through the one
  builder (`build_offline_runtime(deployment_mode=...)`), no-op colocated plane
  vs full disagg plane over SQLite, with the durable ack marker asserted.
- **Done when** Colocated runs with zero metadata-store/durable-ack overhead
  (leasing/backpressure per the scope note) and the equivalence gate is green.

---

### D — Training managers · size L · GPU · status: implemented
- **Goal** Bring the training loop up to production parity: real grad
  accumulation, full resume, checkpoint lifecycle, and an evaluator — the pieces
  `runtime/training` does not have yet.
- **Target state**
  - **Grad accumulation with `no_sync()`.** Today `runtime/training` has ZERO of
    this: `TrainerCore` honors `accumulation_steps` for the optimizer boundary,
    but `FSDPTrainingBackend.backward` is a bare `loss.backward()`, so FSDP
    all-reduces gradients **every micro-step**. Add `no_sync()` over the
    non-boundary micro-steps so the reduction fires once per optimizer step.
  - **Full resume (offline public surface; library seam elsewhere).**
    `TrainerController.save_checkpoint` persists only
    `draft_state_dict` + step/epoch; `FSDPTrainingBackend.load_state_dict` only
    half-loads the optimizer. Add optimizer + LR-scheduler + RNG state — per
    rank (they are FSDP-shard-local) — plus the mid-epoch data position, and a
    seek()-equivalent on the offline stream, so a resumed run continues on the
    same data with the same state. *Continuity precision (as landed, #637):*
    draft weights restore bit-for-bit and the data stream repositions exactly;
    the loss curve is tolerance-continuous, not bit-exact, because
    `BF16Optimizer` rebuilds its fp32 master from the persisted bf16 weights
    (matches the previous optimizer behavior; persisting the master would change the
    optimizer itself and is out of scope here).
  - **CheckpointManager** — rotation (keep-last-N), `best` (by eval metric) and
    `latest` symlinks; owns the `output_dir` layout the controller writes today.
  - **Evaluator** — `simulated_acc_len` and per-position acceptance, with
    per-position accuracy aggregated **before** the geometric sum (the eagle3
    `acces` / `acc_corrects` / `acc_denoms` already flow through
    `StepOutput.metrics`; `TrainerController.evaluate` currently just means
    scalar metrics, which is wrong for acc-len).
- **Implementation**
  - `training/backend.py` — `FSDPTrainingBackend.backward(loss, *, is_boundary)`
    wrapping `self.module.no_sync()` on non-boundary micro-steps; extend
    `state_dict` / `load_state_dict` to round-trip optimizer + scheduler + RNG.
  - `training/trainer.py` — `TrainerCore` passes the boundary flag into
    `backward`; `TrainerController` gains scheduler stepping and a real
    `evaluate` that aggregates per-position correct/denom across batches before
    computing acc-len. Replace the inline `save_checkpoint` body with a
    `CheckpointManager`.
  - New `training/checkpoint.py` (`CheckpointManager`) and `eval/evaluator.py`
    (`Evaluator`), with eval caching in `eval/cache.py`.
- **Tests / gates** `tests/test_runtime/test_checkpoint_resume.py` extended to
  assert loss-curve continuity across a save→resume boundary (optimizer + RNG);
  a `no_sync` gate asserting one all-reduce per optimizer step (and identical
  grads to per-step reduction); an evaluator test asserting per-position
  aggregation precedes the geometric sum.
- **Done when** A run can be killed and resumed continuously (weights + data
  position exact; loss within the documented bf16 fp32-master tolerance) at the
  same world size, accumulation reduces once per optimizer step, checkpoints
  rotate with best/latest surviving restarts, and the evaluator reports correct
  `simulated_acc_len` with all metrics DP-reduced.

---

### E0 — Layout consolidation (move-only) · size M · CPU · status: implemented
- **Goal** Collapse the scattered execution code into **one implementation home per concern** with
  **zero functional change**, so E's composition work is written in the final layout rather than
  re-moved afterward. In the consolidated target tree, `runtime/` =
  substrate only; top-level `training/` and `inference/` are the single execution homes; `modeling/`
  is model definitions only; no facade package.
- **Target state** `runtime/` contains only `contracts.py` + `control_plane/` + `data_plane/`.
  Every training symbol resolves from top-level `training/`, every rollout/capture symbol from
  top-level `inference/`.
- **Implementation** — pure `git mv` + import fixes, one reviewable **rename-only** diff:

  | From (today) | To (consolidated) |
  |---|---|
  | `runtime/training/trainer.py` (`TrainerCore`+`TrainerController`) | `training/controller.py` (kept as one file — not split) |
  | `runtime/training/backend.py` | `training/backend.py` |
  | `runtime/training/strategy.py` | `training/strategies/base.py` |
  | `runtime/training/registry.py` | `training/strategies/registry.py` |
  | `specforge/training/trainer.py` (B3 domain `Trainer`) | stays `training/trainer.py` |
  | `modeling/target/base.py` · `factory.py` | `inference/target_engine/base.py` · `factory.py` |
  | `modeling/target/{eagle3,dflash}_target_model.py` | `inference/target_engine/` (relocated as-is; **collapsed to per-backend engines in E, not here**) |
  | `modeling/target/sglang_backend/capture.py` | `inference/target_engine/sglang_capture_backend.py` |
  | `runtime/inference/rollout_worker.py` · `capture.py` | `inference/rollout_worker.py` · `capture.py` |
  | `runtime/inference/{sglang,dflash}_adapter.py` | `inference/adapters/{eagle3,dflash}.py` |
  | `runtime/launch.py` | `launch.py` (top-level; topology assembly only) |
  | `modeling/target/{target_head.py, custom_backend/}` | **stays** in `modeling/target/` (model defs) |

  The completed hard cut removes the old module-path shims after internal
  consumers have migrated to the consolidated homes.
- **Tests / gates** The full `tests/test_runtime` suite stays green **unchanged** (only import paths
  move); the Phase-B byte-identical gate (`test_phase_b_gate.py`) still passes bit-for-bit — a
  move-only change must not alter a single tensor or loss value.
- **Done when** `runtime/` is substrate-only (`contracts.py` + `control_plane/` + `data_plane/`);
  every training/inference symbol resolves from its top-level home; suite + B-gate green; no
  functional diff.

---

### E — Composition & run surface · size L · GPU · status: implemented
- **Goal** A real product surface, built **on the consolidated layout from `E0`**: a
  draft-architecture registry (separate axis from the strategy registry), the (algorithm × backend)
  target-engine collapse, MLA Eagle3, a typed config + CLI, and exporters.
- **Target state**
  - **`DRAFT_REGISTRY` / `@register_draft`** for draft *architecture* classes
    (`modeling/draft/`), a **distinct axis** from the per-algorithm strategy
    registry. Today there is neither: draft model classes live ungoverned in
    `specforge/modeling/draft/` (`base.py`, `dflash.py`, `llama3_eagle.py`,
    `flex_attention.py`). Add a `modeling/draft/registry.py`; the strategy registry stays in
    top-level `training/strategies/` (relocated in `E0`) so the two registries (architecture vs
    algorithm) are cleanly separate.
  - **Collapse the (algorithm × backend) engine matrix** in `inference/target_engine/`: the
    per-algorithm files relocated by `E0` converge into per-backend generic engines
    (`hf.py` / `sglang.py` / `custom.py`) parameterized by a per-algorithm `CaptureSpec` /
    capture-policy, so adding an algorithm is a spec, not a class-per-backend. NB the SGLang side
    merges cleanly (shared backend; only `wrap_eagle3_logits` + returned fields + shaping differ),
    but the **HF side is a real code difference** (eagle3 = forward hooks on 3 aux layers + concat +
    logits; dflash = `output_hidden_states=True` + layer select, no logits) — so it MUST be a policy
    object, not a data table.
  - **MLA Eagle3 draft** — an MLA-attention draft architecture registered via
    `@register_draft`. *(The draft itself landed early as PR #640 on the Auto* mapping;
    E re-registers it through `@register_draft`.)*
  - **Pydantic config + `specforge` CLI** — a typed run config (Pydantic is
    already a dep in `pyproject.toml`) replacing the argparse-style launch knobs;
    a console-script entry point that builds the config and calls the domain
    `Trainer` (from B).
  - **Exporters** — `export/to_sglang` (with a documented MLA weight-name map)
    and `export/to_hf`.
- **Implementation**
  - `modeling/draft/registry.py` (new) — `DRAFT_REGISTRY` + `register_draft`
    decorator; register the existing eagle3 / dflash drafts and the new MLA
    eagle3 draft.
  - `inference/target_engine/` — collapse the `eagle3.py` / `dflash.py` per-algorithm files
    (relocated in `E0`) into `hf.py` / `sglang.py` / `custom.py` + a per-algorithm `CaptureSpec`;
    `sglang_server.py` is filled by online O1.3.
  - `training/strategies/` (relocated in `E0`) — finalize the `StrategySpec` registry + the
    per-algorithm specs; keep `register_strategy` / `resolve_strategy` import-compatible.
  - `config/schema.py` (new) — Pydantic config; `specforge` CLI entry in
    `pyproject.toml` `[project.scripts]`.
  - `export/to_sglang.py`, `export/to_hf.py` (new) — exporters + the MLA
    weight-name map.
- **Tests / gates** A round-trip export test (train → `to_sglang` → load in
  sglang serving → speculative decode runs); a `to_hf` load test; an MLA-eagle3
  draft training smoke test; a CLI/config test (config parses → builds the same
  `Trainer` the programmatic path does).
- **Done when** A user trains via `specforge train --config <config>`, registers a new draft
  architecture with `@register_draft`, trains an MLA Eagle3 draft, and exports a
  checkpoint that loads in both sglang and HF — all from the consolidated layout.
