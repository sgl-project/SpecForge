# Inference Plane Design (`specforge.inference`)

This is the design note for the **inference**, scoped to this plane.
The cross-plane picture (whole-system map, endpoint reference, autonomy) lives in
[`../runtime/ARCHITECTURE.md`](../runtime/ARCHITECTURE.md); the shared records
every plane exchanges are in
[`../runtime/contracts.py`](../runtime/contracts.py).

## Responsibility

The rollout/inference plane turns leased PromptTasks into per-sample feature tensors and commits only their typed SampleRef metadata to the controller — it never hands a tensor to the controller. It owns the clean boundary to the target engine (a `TargetEngine` via its generic `capture()`; the sglang-version glue lives behind `SGLangCaptureBackend`), the only place target→draft projection/pruning happens, and the loud pre-write validation (`verify_capture` against a typed `CaptureConfig`) that converts layer/name/width/target-dim mismatches into immediate, localized errors at the extraction boundary instead of downstream trainer bugs.

## Internal mechanics

```mermaid
flowchart TD
  classDef compute fill:#e6f6ea,stroke:#3bb061,color:#0b4a22;
  classDef control fill:#e8f0fe,stroke:#3b6fd6,color:#0b2e6b;
  classDef data fill:#fdeede,stroke:#d6893b,color:#6b3a0b;

  A[lease_prompt_tasks] --> B[generate_features per batch]
  B --> C[PolicyFeatureAdapter group by len single forward]
  C --> D[TargetEngine.capture]
  D --> E[_project_target logits or pruned_logits]
  E --> F[verify_capture vs CaptureConfig]
  F -->|mismatch| G[fail_prompt_tasks non retryable]
  F -->|ok| H[FeatureStore put]
  H -->|put error| I[abort then fail retryable]
  H -->|ok| J[append SampleRef]
  J --> K[commit_samples metadata only]

  class A,K control;
  class B,C,D,E,F,J compute;
  class H,I data;
  class G control;
```

The rollout plane turns leased `PromptTask`s into per-sample feature tensors and commits only their typed `SampleRef` metadata — it never hands a tensor to the controller. `RolloutWorker.run_once` is the core loop: lease up to `max_tasks`, call `feature_source.generate_features(tasks, capture=...)` once for the whole batch, enforce a strict `len(feats)==len(tasks)` contract, then per sample pop the out-of-band `__aux_layer_ids__`, run `verify_capture`, and on success `FeatureStore.put` (tensors go straight to the data plane). Every leased task ends in exactly one terminal controller action — `commit_samples` on success or `fail_prompt_tasks` on generate failure / wrong count / contract mismatch / put failure — with `sample_id = f"{run_id}:{task.task_id}"` deterministic and a put exception triggering `abort`. `CaptureConfig` is a frozen, strategy-derived contract carrying `feature_names`, `aux_hidden_state_layer_ids`, `target_repr`, and the derived `expected_aux_width` / `expected_target_dim()`. `verify_capture` is the loud pre-`put` validator: it checks name presence, aux-layer-id equality, aux last-dim width, and target last-dim, gating `pruned_logits` on a non-None `vocab_map_version`, raising `CaptureMismatchError` at the boundary. `PolicyFeatureAdapter` is the ONE runtime adapter (per-strategy `FeatureSchema` decides the emitted dict; `SGLangAdapter` / `DFlashAdapter` are thin schema-pinning subclasses) and the only place target to draft projection happens (`_project_target`: passthrough for `logits`, `t2d`-indexing for `pruned_logits`), batching equal-length tasks into one padding-free `TargetEngine.capture` forward — which must return a typed `TargetCaptureBatch` — and slicing rows back into original task order.

## Endpoints

### What this plane calls into

| From | Endpoint | Plane |
|---|---|---|
| `RolloutWorker` | `DataFlowController.register_rollout_worker` | control |
| `RolloutWorker` | `DataFlowController.lease_prompt_tasks` | control |
| `RolloutWorker` | `PolicyFeatureAdapter.generate_features` | compute |
| `PolicyFeatureAdapter` | `TargetEngine.capture` (→ `SGLangCaptureBackend`) | compute |
| `RolloutWorker` | `FeatureStore.put` | data |
| `RolloutWorker` | `FeatureStore.abort` | data |
| `RolloutWorker` | `DataFlowController.commit_samples` | control |
| `RolloutWorker` | `DataFlowController.fail_prompt_tasks` | control |
