# Eval + Algorithm-Breadth Track

This track runs **parallel and orthogonal** to the domain refactor (./domain-refactor.md)
and the online/disagg work (./online-disaggregation.md): it is where DeepSpec's edge actually
shows up — a trustworthy acceptance-length number and a low-cost path to new draft algorithms.
It has **no Ray and no weight-sync content** by construction: the eval target is the same
**frozen** online/offline target the rest of the system uses (see the weight-sync-out-of-scope
decision in ../../plan.md), so there is nothing to re-sync and no staleness to model. Both
phases build directly on the canonical Substrate (SampleRef + FeatureStore + FeatureDataLoader)
and the runtime training seam (`TrainerCore` / `DraftTrainStrategy` / `StrategySpec`) rather than
replacing any of it.

Sibling tracks: ./domain-refactor.md · ./online-disaggregation.md · root plan ../../plan.md.

---

### E1 — Acceptance-length eval harness · size M · GPU · status: next
- **Goal** Produce a correct, cache-backed `simulated_acc_len` / `avg_loss` / `avg_acc` eval
  pass that is batch-size-independent and tracks the best checkpoint. This is the same
  `Evaluator` the domain refactor's Phase D wires into the domain `Trainer` — see Phase D in
  ./domain-refactor.md and cross-link back here.
- **Target state** Calling eval over any `FeatureDataLoader` stream yields per-position accuracy
  aggregated across the **whole** eval pass *before* the geometric sum; the number does not move
  when you change micro-batch size; repeat runs over an unchanged (eval-data, target, revision,
  tokenizer, template, aux-layers, seqlen) tuple hit an on-disk cache instead of recomputing
  hidden states; the controller records and can restore the best checkpoint by
  `eval/simulated_acc_len`.
- **Implementation**
  - **Fix the existing bug first.** `specforge/runtime/training/trainer.py::TrainerController.evaluate`
    today means per-batch scalars: `TrainerCore._result` collapses the per-position vectors
    (`acces`, `acceptance_rates`, `plosses`) to a single `_scalar` per batch, then `evaluate`
    averages those scalars across batches. That destroys per-position structure and makes any
    acceptance-length derived from it batch-size-dependent. The harness must aggregate the raw
    per-position **sum/count** tensors that `Eagle3TrainStrategy.forward_loss` already emits
    (`acc_corrects`, `acc_denoms` in `StepOutput.metrics`) — do **not** reduce them to scalars in
    the eval path.
  - Add `specforge/runtime/eval/evaluator.py` with `Evaluator` + `EvalConfig`, following the
    sketch in ../../docs/redesign-draft-legacy.md (§4.4 Evaluation system). It consumes a
    `forward_fn` (a thin wrapper over `TrainerCore.eval_step`) and an eval stream that is a plain
    `FeatureDataLoader` over SampleRef+FeatureStore — there is no separate eval source of truth.
  - Aggregation contract (the load-bearing bit): keep running `per_pos_acc_sum` and
    `per_pos_acc_count`, each shape `[ttt_length]`, accumulated over **all** micro-batches; only
    after the full pass compute `per_position_acc = sum / count.clamp_min(1)` and feed *that*
    single aggregated vector into the geometric sum
    `acc_0 + acc_0·acc_1 + acc_0·acc_1·acc_2 + …`. The common bug — treating each batch's
    per-position vector as if it were positions — is explicitly what this ordering prevents.
  - Emit `{eval/avg_loss, eval/avg_acc, eval/simulated_acc_len}`; `avg_loss` is
    token-weighted (`Σ loss·num_tokens / Σ num_tokens`), `avg_acc` is the position-0 aggregated
    accuracy.
  - Add `specforge/runtime/eval/cache.py` with `EvalCache`. Key on
    **eval-data path + target path + target revision + tokenizer path + chat template +
    aux-layer ids + max seqlen** (§4.4 `cache_key`). Missing any field silently serves stale
    tensors after a target swap or template change. Keep this **separate** from the tokenization
    cache (`data/cache.py`) — different keys, different lifecycle; do not merge.
  - Best-checkpoint tracking: have `TrainerController` (trainer.py) keep `best_metric` /
    `best_checkpoint_uri` keyed on `eval/simulated_acc_len`, updated in the eval hook of `fit`,
    and persist a `best` pointer next to the step checkpoints written by `save_checkpoint`.
  - Strategy-agnostic by construction: EAGLE3 supplies per-position vectors; DFlash/Domino supply
    a single-position `accuracy`, which is the `ttt_length=1` degenerate case of the same
    aggregation (geometric sum of one term). No branching in the evaluator.
- **Tests / gates**
  - **Batch-size invariance gate:** the same fixed eval set produces identical
    `simulated_acc_len` at micro-batch sizes 1, 4, 16 (the regression that proves the
    aggregate-before-geometric-sum ordering).
  - Geometric-sum unit test on a hand-computed per-position vector.
  - `EvalCache` key test: flipping any one keyed field (target revision, template, aux layers,
    seqlen, tokenizer) changes the key; a clean re-run with all fields equal is a cache hit.
  - colocated==disagg equivalence: eval metrics match within tolerance whether the stream is a
    `LocalFeatureStore` (mem://) colocated loader or a disagg FeatureStore (reuses the track-wide
    numerical-equivalence gate).
- **Done when** `evaluate` returns the three metrics from a single full-pass aggregation, the
  batch-size-invariance gate is green, the cache hits on an unchanged tuple and misses on any
  changed field, and the controller restores the best checkpoint by `eval/simulated_acc_len`.

---

### E2 — Algorithm breadth · size L · GPU · status: later
- **Goal** Make "add a new speculative-decoding algorithm" a small, well-bounded contribution:
  a `DraftTrainStrategy` + a `StrategySpec` entry (+ optionally a draft arch), reusing the entire
  runtime spine. The first new targets are **MTP** and **Medusa**.
- **Target state** dflash and domino are **already landed** through the strategy registry
  (Phase A) — see `Eagle3TrainStrategy` / `DFlashTrainStrategy` / `DominoTrainStrategy` in
  `specforge/runtime/training/strategy.py` and their three `register_strategy(...)` entries in
  `specforge/runtime/training/registry.py`. Because of that, breadth is no longer "write a new
  `build_*_runtime` family" — it is "add a `StrategySpec` + a loss." Adding MTP or Medusa touches
  only the strategy class, one registry entry, and (if its draft network is new) one file under
  the draft model package.
- **Implementation** — what a new algorithm **must add**:
  1. **A `DraftTrainStrategy` subclass** in `specforge/runtime/training/strategy.py` (or a sibling
     module imported there). It must set `name`, `required_features`, implement
     `trainable_module()` and `forward_loss(batch, ctx) -> StepOutput`, and — if its persisted
     weights are a subset of the wrapped module — `checkpoint_state_filter`. `forward_loss` returns
     `StepOutput(loss, metrics)`; put per-position `acc_corrects`/`acc_denoms` (or a single
     `accuracy`) in `metrics` so the E1 `Evaluator` works unchanged. Only read `StepContext` if the
     loss depends on *where in training* you are (Domino's decayed `lambda_base` is the existing
     precedent — every other strategy ignores `ctx`).
       - *MTP*: multi-token-prediction heads → emit per-position accuracy like EAGLE3 (TTT-style),
         reuse the `ploss_decay`-style weighting.
       - *Medusa*: independent per-head losses over a shared trunk → a single combined loss; per-head
         accuracy can be reported as the per-position vector E1 already aggregates.
  2. **One `StrategySpec` entry** via `register_strategy(...)` in `registry.py`, declaring:
     `make_strategy`, `required_features` (frozenset, drives `CaptureConfig.from_strategy` +
     loader validation), `uses_target_head`, the offline data path
     (`make_offline_reader` / `make_offline_transform` / `make_offline_collate` /
     `offline_target_repr`), `make_online_collate`, `make_adapter` (None ⇒ default `SGLangAdapter`),
     and `supports_online`. If a data path isn't wired yet, leave that factory `None` — the builder
     raises an actionable `NotImplementedError` instead of silently feeding EAGLE3-shaped features.
  3. **Optionally, a draft architecture** under `specforge/modeling/draft/` (alongside
     `dflash.py` / `llama3_eagle.py`) — only if the algorithm needs a network the existing draft
     models don't provide. Medusa heads or MTP heads would live here; an algorithm that reuses an
     existing draft body (as Domino reuses DFlash's draft with a different head/loss) needs **no**
     new arch file.
- **What it reuses (unchanged):** `TrainerCore` / `TrainerController` (trainer.py),
  `FSDPTrainingBackend` (backend.py), `FeatureDataLoader` + FeatureStore (Substrate), checkpointing,
  the online capture adapters, and the E1 `Evaluator`. The topology builders stay
  `topologies + one spec per model` rather than `topologies × models` — the model is the
  `strategy=` parameter resolved through `resolve_strategy`.
- **Tests / gates**
  - A new strategy registers and resolves: `resolve_strategy("mtp")` / `available_strategies()`
    includes it; `required_features` round-trips through loader validation.
  - `forward_loss` emits metrics in the shape E1 consumes (per-position sum/count or a scalar
    `accuracy`); E1 produces a finite `simulated_acc_len` for the new strategy with no evaluator
    changes.
  - Offline↔online schema parity for the new strategy where both paths exist (same
    `required_features`, equivalent collate output).
  - `NotImplementedError` is raised (not a wrong-feature run) when an unwired data path is invoked.
- **Done when** at least one new algorithm (MTP **or** Medusa) is registered as a `StrategySpec`,
  trains through the unchanged `TrainerCore`, is evaluated by the unchanged E1 `Evaluator`, and the
  only files touched are its strategy class, its registry entry, and (if needed) one draft arch file.
