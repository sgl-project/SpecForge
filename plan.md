# SpecForge Redesign Plan

> **Status**: Draft (train-with-decode promoted to Phase 5)
> **Last updated**: 2026-05-31

---

## 1. Context

SpecForge trains speculative decoding draft models (EAGLE3, DFlash) aligned with SGLang
serving. It works well for single-cluster torchrun workflows, but has accumulated structural
debt that makes it hard to add new architectures, new training modes, and production tooling.

TorchSpec (a sibling project) solves many of these gaps via Ray + Mooncake disaggregation,
but at the cost of heavy dependencies and operational complexity. This plan takes a different
approach: **keep SpecForge's simple torchrun-native design, but restructure internals so
new capabilities compose cleanly.**

### What we keep verbatim

- `specforge/core/loss.py` — Triton `LogSoftmaxLoss` and `_compute_loss`.
- `specforge/optimizer.py` — `BF16Optimizer` (FP32 master weights + AdamW + grad clip).
- `specforge/lr_scheduler.py` — `CosineAnnealingWarmupLR` and the `TwoStageScheduler` family.
- `specforge/tracker.py` — `Tracker` ABC + `TRACKER_REGISTRY` (wandb/tensorboard/swanlab/mlflow).
- `specforge/distributed.py` — `init_distributed`, device meshes, yunchang USP integration.
- `specforge/core/eagle3_adapters.py` — `BackendAdapter` / `StepState` / `UspAdapter`.
- SGLang target backend code in `specforge/modeling/target/sglang_backend/`.
- All 30+ existing draft model configs.

### What we throw away

- `scripts/train_eagle3.py` and `scripts/train_dflash.py` as god scripts (become thin shims).
- Hardcoded `_model_mapping` / `_config_mapping` dicts in `specforge/modeling/auto.py`.
- `configs/deepseek-v3-671b-eagle3.json` that claims `model_type: llama` for a DeepSeek target.
- The argparse-flags + per-arch JSON config split.
- `QwenVLOnlineEagle3Model` (VLM handled by target engine + data pipeline, not a separate model class).

### Workloads in scope

These are the concrete training workflows the redesign must support. All four share the same
trainer / strategy / draft surface; they differ only in which `HiddenStateStream` and
`TargetEngine` get composed at the top.

| # | Workload | Description | Engine | Stream |
|---|---|---|---|---|
| W1 | **Offline** | Pre-computed hidden states from disk; trainer runs DP-only | none (`TargetHead` for logits) | `OfflineStream` |
| W2 | **In-process online** | Target on the same GPUs as draft (TP-collective); current default | `SGLangEagle3TargetModel` / `HFEagle3TargetModel` / `CustomEagle3TargetModel` | `OnlineStream` |
| W3 | **Disaggregated online** | Target on a separate SGLang server cluster; trainer talks HTTP | `SGLangServerEngine` | `RemoteStream` |
| W4 | **Train-with-decode** *(new in this revision — promoted from Phase 6)* | One long-lived SGLang server simultaneously **(a)** generates training data via prefill+aux and **(b)** serves real spec-decoding traffic. Trainer pushes draft weights into the same server every N steps so production traffic immediately benefits. | `SGLangServerEngine` (decode mode + weight push) | `OnlineStream` over static jsonl, or new `ServingTrafficStream` over a serving-traffic buffer |

W4 is what makes the no-Ray bet non-trivial: TorchSpec gets W4 "for free" because every
SglEngine actor already supports both modes. SpecForge needs to add three things explicitly
— `TargetEngine.update_draft_weights`, a decode-mode flag on `SGLangServerEngine`, and a
periodic weight-sync hook in `Trainer` — but **no actor topology change**: it stays one
torchrun-native trainer process talking HTTP to one always-on SGLang server. See §4.10.

---

## 2. Current State Gap Analysis

This section grounds the design in what is concretely wrong or missing in the current
codebase. Two flavors of gap: **(A) missing capabilities** — things the design assumes but
no code exists for; **(B) structural problems** — code that exists but the shape is wrong.

### 2.1 Missing capabilities

| Gap | Evidence in current tree | Fixed by |
|---|---|---|
| **MLA-aware EAGLE3 draft** | Zero references to `DeepseekV3Config` / `Eagle3Deepseek*` outside test data. `configs/deepseek-v3-671b-eagle3.json` is a *Llama* draft mislabeled as DeepSeek (`model_type: "llama"`, `architectures: ["LlamaForCausalLMEagle3"]`). | Phase 1 #2 — port `deepseek_eagle.py` from TorchSpec, register via `@register_draft`. |
| **Backbone-agnostic DFlash** | `DFlashDraftModel` extends `Qwen3PreTrainedModel`; uses `Qwen3MLP`, `Qwen3DFlashAttention`, `Qwen3RMSNorm` directly (`specforge/modeling/draft/dflash.py:212`). All DFlash configs are Qwen3-only. | Phase 6 #25 — parameterize MLP/Norm/RoPE; add `llama_dflash.py`. |
| **Remote / disaggregated target** | `modeling/target/sglang_backend/` only does in-process SGLang (target loaded on same node as trainer). No HTTP client. Online mode for 671B targets is infeasible. | Phase 4 #17 — `SGLangServerEngine` over HTTP. |
| **Eval during training (proper)** | No `EvalCache`, no `simulated_acc_len`, no per-position-accuracy aggregation, no best-checkpoint tracking — grep is clean. | Phase 2 #9 — `Evaluator` + `EvalCache`. |
| **Checkpoint rotation / best tracking** | `train_eagle3.py:552:def save_checkpoints(...)` just writes every N steps. No `max_checkpoints`, no `best_checkpointed_iteration.txt`, no `meta.json`. | Phase 2 #8 — `CheckpointManager`. |
| **Resume that actually advances the stream** | No `stream.seek()`. Current `--resume` reloads weights but the dataloader yields from sample 0, silently re-training on the prefix. | §4.2 trainer pseudo-code + Phase 2 #5 — `HiddenStateStream.seek()`. |
| **Correct gradient accumulation** | Zero `no_sync()` calls anywhere in the codebase. FSDP all-reduces on every micro-step, defeating the point of `--accumulation-steps`. | Phase 2 #10 — `model.no_sync()` on non-sync micro-steps. |
| **Plugin registry for drafts** | Hardcoded `_model_mapping = {LlamaConfig: LlamaForCausalLMEagle3}` (`specforge/modeling/auto.py:35`). Code already has TODO: "should support lazy model mapping via registry". | Phase 1 #1 — `@register_draft` decorator. |
| **Train-with-decode (W4)** | No `update_draft_weights` on any target abstraction. `SGLangEagle3TargetModel.from_pretrained` hardcodes `disable_cuda_graph=True` (training-data-gen only); no notion of a long-lived dual-purpose server. No serving-traffic stream — `core/eagle3.py` only knows about static jsonl batches. No periodic weight push from trainer. | Phase 5 #21–#24 — `update_draft_weights` on `TargetEngine`, decode-mode flag on `SGLangServerEngine`, `ServingTrafficStream`, `Trainer._maybe_sync_draft_weights`. See §4.10. |
| **SGLang export with MLA weight map** | No `export/to_sglang.py`. MLA draft weights (`q_a_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`) need explicit rename to whatever SGLang's spec-decoder loader expects. | Phase 3 #15 + `docs/export_weight_map_mla.md`. |
| **FSDP2 readiness** | `apply_fsdp` is FSDP1-only, inlined in scripts. No seam for FSDP2 swap. Locks the trainer to a pattern that composes poorly with `torch.compile`. | §4.9 — versioned `apply_fsdp` seam in Phase 2 (FSDP2 impl deferred to Phase 7 #31). |
| **Pydantic config + CLI** | `specforge/args.py` is 219 lines of argparse; architecture lives in JSON (`--draft-model-config`); training flags live in CLI. Two sources of truth, no validation. | Phase 3 #13/14 — one validated YAML per run. |
| **VLM unification** | Separate `QwenVLOnlineEagle3Model` class for VLM targets — parallel hierarchy that doesn't fit the abstraction. | Phase 4 #19 — typed `MediaInputs` in `TargetEngine`. |

### 2.2 Structural / workflow problems

Code exists for these, but the shape is wrong: duplication, missing abstractions,
maintenance debt that compounds with each new arch.

| Problem | Concrete evidence | Fixed by |
|---|---|---|
| **God-script duplication** | `train_eagle3.py` = 1012 lines / 60 `add_argument`s; `train_dflash.py` = 562 lines / 44 `add_argument`s. Same argparse, same distributed init, same checkpoint save, same logging — copy-pasted twice. Total ~1574 lines across two top-level scripts. | Phase 2 #6 — single `Trainer`, strategy dispatch. Reduces to one ~70-line loop + strategy classes. |
| **Target-model duplication in modeling/** | `modeling/target/eagle3_target_model.py` (873 lines) and `modeling/target/dflash_target_model.py` (315 lines) are parallel hierarchies with no shared base. | Phase 2 #4 — single `TargetEngine` ABC; three concrete impls. |
| **Online ≠ class, offline ≠ class** | `core/eagle3.py` is the "online" wrapper (606 lines); offline is a different path through `OfflineEagle3Dataset` + `TargetHead`. The trainer must branch by mode. | Phase 2 #5 — both become `HiddenStateStream` implementations. Trainer takes one iterator regardless. |
| **Arch dispatch by 4-file shotgun** | Adding a new arch today touches `modeling/auto.py:35`, `modeling/auto.py:88`, `modeling/auto.py:134`, plus `modeling/draft/__init__.py`. | Phase 1 #1 — one file, one decorator. |
| **30 hand-maintained shell scripts** | `examples/run_*_online.sh` × 30. Defaults drift between them; new features need 30 edits. | Phase 3 #14 — 30 YAML files derived from one schema; shell scripts collapse to `specforge train --config ...`. |
| **No data prefetch overlap** | `core/eagle3.py` runs target forward synchronously inside the training step. Draft backward blocks on target forward. | Phase 2 #5 — `OnlineStream.prefetch_factor` overlaps producer and consumer. |
| **Misleading config in repo** | `configs/deepseek-v3-671b-eagle3.json` declares `model_type: "llama"` for a DeepSeek target — silently wrong if read as "MLA Eagle3 for V3." | §4.7 — rename to `..._llama_draft.json`, add real MLA config, deprecation warning. |
| **No numerical-equivalence gate for refactors** | Nothing prevents Phase 2's `Trainer` extraction from quietly changing loss curves. | §10.1 — `atol/rtol` gate at fixed steps on every PR touching `core/`, `training/`, `models/drafts/`. |
| **No FSDP all-reduce verification** | No profiler check that accumulation actually skips sync. `--accumulation-steps` flag exists but the sync isn't suppressed (no `no_sync()`) — same number of all-reduces as without accumulation. | §10.3 — one-time profiler check; one all-reduce per `optimizer.step()`. |
| **`Eagle3DraftModel.backbone()` ABC return shape** | Returns `Tensor`; KV cache mutates `past_key_values` in place (HF `DynamicCache` pattern). DFlash respects this. A naive MLA port that returns `(out, k, v)` would break the ABC. | §4.7 — explicit "use `DynamicCache.update`, return `Tensor`" rule for MLA. |

### 2.3 The two highest-leverage chunks

If only two things ship this quarter, do these:

**Chunk 1 — Phase 1 (Week 1-2): `@register_draft` + MLA draft + Kimi-K2.5 TP/SP smoke test.**
Unblocks the actual feature ask (DFlash + MLA Eagle3 on SGLang). Additive — old
`_model_mapping` dicts stay as a fallback. ~3 new files, ~10 lines of edits to `auto.py`.
The TP/SP smoke test (§4.7) is the de-risking step that turns Kimi-K2.5 from "should
work" into a verified deliverable.

**Chunk 2 — Phase 2 (Week 3-6): `TargetEngine` + `HiddenStateStream` + `Trainer` together.**
These ship as a unit. Extracting one without the others either leaves the trainer talking
directly to target models (Phase 4 has to undo it) or leaves the stream abstraction
without a consumer (can't be tested end-to-end). The Phase 2 acceptance gate is that the
legacy shells, now calling new internals, match the old loss curves to numerical-
equivalence tolerance (§10.1). Pass that, and Phase 4 becomes drop-in plugins
(`SGLangServerEngine`, `RemoteStream`) without re-plumbing.

Everything else (export tool, CLI, VLM cleanup, FSDP2, WSD, Mooncake) is debt cleanup
that compounds — important but not blocking the stated goal.

---

## 3. Design Principles

1. **One trainer, many drafts, many targets.** Config-driven dispatch via a plugin registry.
   New arch = new file, not a new script.

2. **The target is an interface, not a process.** Same trainer whether the target is in-process
   HF, a separate SGLang server, or a remote cluster.

3. **Three data modes behind one abstraction.** Offline-cached, online-local, online-remote —
   all consumed via the same `HiddenStateStream` iterator. But respect the asymmetry: online
   streams own GPU resources and backpressure; offline streams are pure readers.

4. **Strategies, not forks.** EAGLE3 (TTT unroll) and DFlash (block-causal) are two
   `DraftTrainStrategy` implementations sharing the trainer.

5. **Keep what works.** Don't rewrite battle-tested code for symmetry.

---

## 4. Target Architecture

### 4.1 Module layout

```
specforge/
├── config/                          # Structured configuration
│   ├── schema.py                    # Pydantic models (Config, ModelConfig, DatasetConfig, ...)
│   ├── loader.py                    # YAML load + merge + CLI override + validation
│   └── draft_configs/               # Draft model JSONs (moved from top-level configs/)
│       ├── llama3_8b_eagle3.json
│       ├── qwen3_8b_eagle3.json
│       ├── qwen3_8b_eagle3_mla.json       # NEW
│       ├── kimi_k25_eagle3_mla.json       # NEW
│       ├── deepseek_v3_671b_eagle3.json   # NEW — real MLA config
│       └── ...
│
├── core/                            # Core algorithms (preserved)
│   ├── eagle3.py                    # Eagle3Model (renamed from OnlineEagle3Model)
│   ├── dflash.py                    # DFlashModel (renamed from OnlineDFlashModel)
│   ├── loss.py                      # Triton LogSoftmaxLoss (UNCHANGED)
│   └── adapters.py                  # BackendAdapter / UspAdapter (UNCHANGED)
│
├── models/
│   ├── drafts/                      # Draft model plugin registry (see §4.2 for the registry + naming contract)
│   │   ├── __init__.py              # DRAFT_REGISTRY + @register_draft decorator
│   │   ├── base.py                  # Eagle3DraftModel ABC (from modeling/draft/base.py)
│   │   ├── <model>_<strategy>.py    # one file per arch — ${model_name}_${strategy}
│   │   │                            #   e.g. llama_eagle3, deepseek_eagle3, qwen3_dflash
│   │   └── auto.py                  # AutoEagle3DraftModel (registry-backed, no hardcoded dicts)
│   │
│   └── targets/                     # Target engine abstraction
│       ├── base.py                  # TargetEngine ABC + TargetOutput dataclass
│       ├── hf_engine.py             # In-process HF target (lazy import)
│       ├── sglang_engine.py         # In-process SGLang target (lazy import)
│       ├── sglang_server_engine.py  # SGLang-as-service over HTTP (NEW)
│       ├── custom_engine.py         # Custom TP backend (existing custom_backend/)
│       └── target_head.py           # TargetHead for offline logits (existing)
│
├── data/
│   ├── streams/                     # HiddenStateStream abstraction
│   │   ├── base.py                  # HiddenStateStream protocol
│   │   ├── online.py               # In-process target generates hidden states
│   │   ├── offline.py              # Pre-computed hidden states from disk
│   │   └── remote.py               # Target on separate SGLang server (NEW)
│   ├── template.py                  # TEMPLATE_REGISTRY (UNCHANGED)
│   ├── parse.py                     # Parsers + KimiK25Parser, MiniMaxParser (NEW)
│   ├── preprocessing.py             # build_eagle3_dataset etc. (UNCHANGED)
│   ├── collator.py                  # DataCollatorWithPadding (extracted from utils.py)
│   └── cache.py                     # Tokenization cache + eval cache (NEW)
│
├── training/
│   ├── trainer.py                   # Trainer: unified training loop
│   ├── strategies/
│   │   ├── __init__.py              # DraftTrainStrategy protocol
│   │   ├── eagle3_ttt.py            # Eagle3 TTT unroll + forward-KL
│   │   └── dflash_block.py          # DFlash block-causal + anchor sampling
│   ├── optimizer.py                 # BF16Optimizer (UNCHANGED)
│   ├── lr_scheduler.py              # CosineWarmup + WSD scheduler (NEW)
│   ├── checkpoint.py                # CheckpointManager (NEW)
│   ├── fsdp.py                      # apply_fsdp (FSDP1 SHARD_GRAD_OP + future FSDP2)
│   └── distributed.py               # init_distributed etc. (moved from specforge/distributed.py)
│
├── eval/                            # Evaluation system (NEW)
│   ├── evaluator.py                 # Evaluator: eval loop + metric aggregation
│   ├── cache.py                     # EvalCache: MD5-keyed disk cache
│   └── metrics.py                   # simulated_acc_len, avg_loss, avg_acc
│
├── export/                          # Model export tools (NEW)
│   ├── to_hf.py                     # FSDP checkpoint → HF format + vocab pruning
│   └── to_sglang.py                 # Checkpoint → SGLang spec-decoder layout
│
├── tracker.py                       # UNCHANGED
├── utils.py                         # UNCHANGED
│
├── cli.py                           # `specforge train|prepare|export|eval` (NEW)
└── __init__.py

scripts/
├── legacy/                          # Old scripts as thin shims
│   ├── train_eagle3.py
│   └── train_dflash.py
├── prepare_data.py                  # UNCHANGED
├── prepare_hidden_states.py         # UNCHANGED
└── regenerate_train_data.py         # UNCHANGED
```

### 4.2 Key abstractions

#### Draft model registry

This is the single source of truth for the `models/drafts/` layout sketched in §4.1.
One file per architecture, named `${model_name}_${strategy}` (strategy ∈ `eagle3`,
`dflash`, `domino`, `dspark`) — e.g. `llama_eagle3.py`, `deepseek_eagle3.py`,
`qwen3_dflash.py`; the registry key matches the filename stem.

```python
# models/drafts/__init__.py
DRAFT_REGISTRY: dict[str, type] = {}

def register_draft(name: str):
    """Decorator. New arch = new file + @register_draft("deepseek_v3_eagle3")."""
    def wrapper(cls):
        DRAFT_REGISTRY[name] = cls
        return cls
    return wrapper

# models/drafts/llama_eagle3.py
@register_draft("llama_eagle3")
class LlamaForCausalLMEagle3(Eagle3DraftModel):
    ...

# models/drafts/deepseek_eagle3.py — MLA (NEW)
@register_draft("deepseek_v3_eagle3")
class Eagle3DeepseekV2ForCausalLM(Eagle3DraftModel):
    config_class = DeepseekV3Config
    ...

# models/drafts/qwen3_dflash.py — DFlash (existing)
@register_draft("qwen3_dflash")
class DFlashDraftModel(Eagle3DraftModel):
    ...
```

The `Eagle3DraftModel` ABC is preserved exactly as-is. Its interface is already well-designed:

```python
class Eagle3DraftModel(PreTrainedModel, ABC):
    def embed_input_ids(self, input_ids: Tensor) -> Tensor: ...
    def project_hidden_states(self, hidden_states: Tensor) -> Tensor: ...
    def backbone(self, input_embeds, hidden_states, cache_hidden,
                 attention_mask, position_ids, past_key_values=None,
                 use_cache=True) -> Tensor: ...
    def compute_logits(self, hidden_states: Tensor) -> Tensor: ...
    # Concrete: load_embedding, freeze_embedding, load_vocab_mapping, t2d/d2t
```

`AutoDraftModelConfig.from_file()` is updated to look up `DRAFT_REGISTRY` first, fall back to
HF config type dispatch for backward compatibility.

#### Target engine

```python
# models/targets/base.py
@dataclass
class TargetOutput:
    aux_hidden_states: torch.Tensor  # [batch, seq, hidden * num_aux_layers] — raw concat (NOT projected)
    target_logits: torch.Tensor      # [batch, seq, vocab] — pre-softmax logits in draft vocab space
    loss_mask: torch.Tensor          # [batch, seq]
    input_ids: torch.Tensor          # [batch, seq]
    attention_mask: torch.Tensor     # [batch, seq]
    last_hidden_states: Optional[torch.Tensor] = None

class TargetEngine(ABC):
    @abstractmethod
    def generate_train_data(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        loss_mask: Tensor,
        media: Optional[MediaInputs] = None,   # typed VLM payload (pixel_values, image_grid_thw, ...)
    ) -> TargetOutput: ...

    @property
    @abstractmethod
    def aux_layer_ids(self) -> list[int]: ...

    def set_aux_hidden_states_layers(self, layers: list[int]) -> None: ...
```

Naming:
- `target_logits` (not `target`) — the previous overload of "target" with "target model" was confusing.
- `aux_hidden_states` — raw concatenated multi-layer hidden states from the target.
  The projection (3·hidden → hidden) **stays inside the draft model** (`project_hidden_states`),
  so streams and engines never need to know the draft's hidden_size.

This is a rename + mild generalization of the existing `Eagle3TargetModel` +
`Eagle3TargetOutput`. The existing SGLang/HF/Custom implementations become concrete engines
with no API changes. VLM is handled via `**media_kwargs` — no separate model class needed.

Lazy imports ensure SGLang is never imported unless `backend="sglang"`:

```python
def get_target_engine(backend: str, **kwargs) -> TargetEngine:
    if backend == "sglang":
        from specforge.models.targets.sglang_engine import SGLangTargetEngine
        return SGLangTargetEngine(**kwargs)
    elif backend == "sglang_server":
        from specforge.models.targets.sglang_server_engine import SGLangServerEngine
        return SGLangServerEngine(**kwargs)
    elif backend == "hf":
        from specforge.models.targets.hf_engine import HFTargetEngine
        return HFTargetEngine(**kwargs)
    elif backend == "custom":
        from specforge.models.targets.custom_engine import CustomTargetEngine
        return CustomTargetEngine(**kwargs)
    raise ValueError(f"Unknown target backend: {backend}")
```

#### HiddenStateStream

```python
# data/streams/base.py
class HiddenStateStream(ABC):
    """Produces TrainBatch instances for the draft trainer."""

    def setup(self, draft_config) -> None:
        """Inform stream of draft requirements (aux layers, vocab mapping, etc.)."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[TrainBatch]: ...

    def seek(self, step: int) -> None:
        """Resume from step N. Required for checkpoint resume to behave correctly.

        Default behavior (provided by base class): advance the iterator by `step`
        batches. Implementations with random-access storage (e.g. OfflineStream)
        should override to seek directly without consuming.
        """

    def teardown(self) -> None:
        """Release GPU / network resources."""
        pass

@dataclass
class TrainBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_logits: torch.Tensor     # target logits on draft vocab (pre-softmax)
    loss_mask: torch.Tensor
    aux_hidden_states: torch.Tensor # raw concat of target aux layers — draft projects internally
    position_ids: Optional[torch.Tensor] = None
```

Three implementations:

| Stream | Source | GPU cost | Notes |
|--------|--------|----------|-------|
| `OnlineStream` | In-process `TargetEngine` | High (target on same GPUs) | Current "online" mode; supports `prefetch_factor` to overlap target inference with draft training |
| `OfflineStream` | Pre-computed .pt files | None | Current "offline" mode; overrides `seek()` for O(1) resume |
| `RemoteStream` | SGLang server over HTTP | None locally | NEW — target on separate node(s); supports `prefetch_factor` and request batching |

`OnlineStream` owns the target engine lifecycle and handles TP→DP batch sharding
(currently done manually in `train_eagle3.py:get_dp_data_shard_from_tp`). Exposes
`prefetch_factor: int = 2` so the next batch's target forward overlaps the current
batch's draft backward.

`OfflineStream` wraps the existing `OfflineEagle3Dataset` + `TargetHead` path. Overrides
`seek()` to jump directly to a sample index without iterating intermediate files.

`RemoteStream` sends tokenized batches to an SGLang server, receives hidden states back.
This covers the "671B target on dedicated GPUs" case without Ray/Mooncake. Should also
expose `prefetch_factor` (network roundtrip dominates without it) and a `max_in_flight`
bound so back-pressure is explicit.

#### DraftTrainStrategy

```python
# training/strategies/__init__.py
class DraftTrainStrategy(ABC):
    """Encapsulates one training algorithm's forward + loss computation."""

    @abstractmethod
    def forward_and_loss(
        self,
        draft_model: Eagle3DraftModel,
        batch: TrainBatch,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Returns (loss, metrics_dict). Target data is already in batch."""
        ...

    @abstractmethod
    def build_model(self, draft_model, **kwargs) -> nn.Module:
        """Wrap draft_model in the strategy-specific training wrapper."""
        ...

    def fsdp_wrap_policy(self) -> Optional[Callable]:
        """Optional: return an FSDP auto-wrap policy tailored to this strategy's wrapper.

        The trainer applies this if non-None; otherwise falls back to a default
        transformer-block wrap policy. Lets strategies declare wrapping intent
        instead of the trainer second-guessing.
        """
        return None
```

Two implementations:

- **`Eagle3TTTStrategy`**: wraps draft in `Eagle3Model` (current `core/eagle3.py`), runs
  TTT unroll, forward-KL loss via `LogSoftmaxLoss`. The existing `Eagle3Model.forward()`
  signature maps directly.

- **`DFlashBlockStrategy`**: wraps draft in `DFlashModel` (current `core/dflash.py`), does
  anchor sampling + block-causal CE. The existing `OnlineDFlashModel.forward()` maps directly.

The strategy does NOT touch the target engine — batch already contains materialized tensors.
This is a hard boundary.

#### Trainer

```python
# training/trainer.py
class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.strategy = self._build_strategy()
        self.checkpoint_mgr = CheckpointManager(config.training)
        self.evaluator = Evaluator(config.eval) if config.eval.enabled else None
        self.tracker = create_tracker(config.logging)

    def train(self):
        # 1. Distributed setup
        init_distributed(...)

        # 2. Build draft model from registry
        draft_config = AutoDraftModelConfig.from_file(self.config.model.draft_model_config)
        draft_model = DRAFT_REGISTRY[draft_config.architectures[0]].from_config(draft_config)

        # 3. Build strategy-specific wrapper
        model = self.strategy.build_model(draft_model, ...)

        # 4. Apply FSDP (strategy may override wrap policy)
        model = apply_fsdp(
            model,
            self.config.training,
            wrap_policy=self.strategy.fsdp_wrap_policy(),
        )

        # 5. Build optimizer
        optimizer = BF16Optimizer(model, ...)

        # 6. Build data stream
        stream = self._build_stream()
        stream.setup(draft_config)

        # 7. Resume if needed — note: stream.seek() is required for correctness.
        #    enumerate(stream, start=start_step) only changes the counter; without
        #    seek(), the stream still yields batch 0 first, silently re-training on it.
        start_step = 0
        if self.config.training.resume:
            start_step = self.checkpoint_mgr.load(model, optimizer)
            stream.seek(start_step)

        # 8. Training loop
        accum = self.config.training.accumulation_steps
        for step, batch in enumerate(stream, start=start_step):
            if step >= self.config.training.max_steps:
                break

            # Forward + loss (strategy-specific)
            loss, metrics = self.strategy.forward_and_loss(model, batch)
            scaled_loss = loss / accum

            # Skip gradient sync on all but the final micro-step of an accumulation
            # window. With FSDP this is the difference between one all-reduce per
            # micro-batch and one per optimizer step — non-trivial on large models.
            is_sync_step = ((step + 1) % accum == 0)
            sync_ctx = nullcontext() if is_sync_step else model.no_sync()
            with sync_ctx:
                scaled_loss.backward()

            if is_sync_step:
                optimizer.step()

            # Logging
            if (step + 1) % self.config.training.log_interval == 0:
                self.tracker.log(metrics, step=step)

            # Eval
            if self.evaluator and (step + 1) % self.config.eval.interval == 0:
                eval_metrics = self.evaluator.run(
                    lambda b: self.strategy.forward_and_loss(model, b),
                    self._eval_stream,
                )
                self.checkpoint_mgr.update_best(step, eval_metrics)
                self.tracker.log(eval_metrics, step=step)

            # Save
            if (step + 1) % self.config.training.save_interval == 0:
                self.checkpoint_mgr.save(step, model, optimizer)

        stream.teardown()
        self.tracker.close()
```

This is ~70 lines of logic. Everything else is behind interfaces.

Two non-obvious correctness points the pseudo-code makes explicit:

1. **Resume requires `stream.seek(start_step)`.** Without it, `enumerate(stream, start=N)`
   only relabels the counter — the iterator still yields batch 0 first, so resume
   silently re-trains on the prefix.

2. **Gradient accumulation needs `model.no_sync()` on non-sync steps.** Otherwise FSDP
   all-reduces on every micro-batch, defeating the point of accumulation.

### 4.3 Checkpoint manager

```python
# training/checkpoint.py
class CheckpointManager:
    def __init__(self, config: TrainingConfig):
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.max_checkpoints = config.max_checkpoints    # 0 = keep all
        self.best_score = -float("inf")

    def save(self, step: int, model, optimizer, meta: dict = None):
        """Save model + optimizer + LR state + RNG + meta.json."""
        step_dir = self.checkpoint_dir / f"iter_{step + 1:07d}"
        # torch.distributed.checkpoint.save for model, optimizer
        # rank-0: rng.pt, meta.json (step, timestamp, global_step, world_size)
        self._update_latest(step + 1)
        self._rotate()

    def load(self, model, optimizer=None, continual=False) -> int:
        """Load latest or specified checkpoint. Returns start_step."""
        step_id = self._read_latest()
        # torch.distributed.checkpoint.load
        # If continual: skip optimizer/LR/RNG, only restore weights + rebuild FP32 master
        return step_id

    def update_best(self, step: int, eval_metrics: dict):
        """Track best checkpoint by simulated_acc_len."""
        score = eval_metrics.get("simulated_acc_len", eval_metrics.get("avg_acc", 0))
        if score > self.best_score:
            self.best_score = score
            self._write_best(step + 1, eval_metrics)

    def _rotate(self):
        """Keep only max_checkpoints newest iter_* directories."""
        if self.max_checkpoints <= 0:
            return
        dirs = sorted(self.checkpoint_dir.glob("iter_*"))
        for d in dirs[:-self.max_checkpoints]:
            shutil.rmtree(d)

    def _update_latest(self, step_id: int):
        (self.checkpoint_dir / "latest_checkpointed_iteration.txt").write_text(str(step_id))

    def _write_best(self, step_id: int, metrics: dict):
        (self.checkpoint_dir / "best_checkpointed_iteration.txt").write_text(str(step_id))
        (self.checkpoint_dir / "best_meta.json").write_text(json.dumps(metrics, indent=2))
```

### 4.4 Evaluation system

```python
# eval/evaluator.py
class Evaluator:
    def __init__(self, config: EvalConfig):
        self.cache = EvalCache(config.cache_dir)
        self.micro_batch_size = config.micro_batch_size

    def run(self, forward_fn, eval_stream: HiddenStateStream) -> dict:
        """Run full eval pass, return aggregated metrics.

        Per-position accuracy must be averaged *across all batches first*, then fed
        into the geometric sum. Treating each batch's per-position vector as if it
        were positions makes simulated_acc_len batch-size-dependent.
        """
        total_loss_x_tokens = 0.0
        total_tokens = 0
        # Sum and count per draft position (TTT step), aggregated across the whole pass.
        per_pos_acc_sum: Optional[torch.Tensor] = None       # shape [ttt_length]
        per_pos_acc_count: Optional[torch.Tensor] = None     # shape [ttt_length]

        for batch in self._iter_micro_batches(eval_stream):
            with torch.no_grad():
                loss, metrics = forward_fn(batch)
            total_loss_x_tokens += metrics["loss"] * metrics["num_tokens"]
            total_tokens += metrics["num_tokens"]

            ppa = metrics.get("per_position_acc")        # [ttt_length], weighted by num_tokens
            ppc = metrics.get("per_position_count")      # [ttt_length], token counts per position
            if ppa is not None:
                if per_pos_acc_sum is None:
                    per_pos_acc_sum = torch.zeros_like(ppa)
                    per_pos_acc_count = torch.zeros_like(ppc)
                per_pos_acc_sum += ppa * ppc             # accumulate weighted sum
                per_pos_acc_count += ppc

        # Aggregate first, then geometric-sum.
        per_position_acc = (per_pos_acc_sum / per_pos_acc_count.clamp_min(1)).tolist()
        return {
            "eval/avg_loss": total_loss_x_tokens / max(total_tokens, 1),
            "eval/avg_acc": float(per_position_acc[0]) if per_position_acc else 0.0,
            "eval/simulated_acc_len": self._simulated_acc_len(per_position_acc),
        }

    @staticmethod
    def _simulated_acc_len(per_position_acc: list[float]) -> float:
        """E[accepted tokens] = acc_0 + acc_0*acc_1 + acc_0*acc_1*acc_2 + ...

        `per_position_acc` is the *aggregated* per-position accuracy across the full
        eval set, length = ttt_length. Not a list of per-batch vectors.
        """
        cumulative = 1.0
        total = 0.0
        for acc in per_position_acc:
            cumulative *= acc
            total += cumulative
        return total

# eval/cache.py
class EvalCache:
    """Disk cache for pre-computed eval hidden states.

    Cache key must cover everything that would change the cached tensors:
    eval data, target model identity & revision, tokenizer, chat template,
    aux layer ids, and sequence length. Missing any of these silently serves
    stale data after a target swap or template change.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)

    def cache_key(
        self,
        eval_path: str,
        target_path: str,
        target_revision: str,
        tokenizer_path: str,
        chat_template: str,
        aux_layer_ids: list[int],
        max_seq_len: int,
    ) -> str:
        content = "|".join([
            eval_path,
            target_path,
            target_revision or "",
            tokenizer_path,
            chat_template,
            ",".join(map(str, aux_layer_ids)),
            str(max_seq_len),
        ])
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def try_load(self, key: str) -> Optional[list]:
        path = self.cache_dir / "eval_cache" / key
        if path.exists():
            return [torch.load(f) for f in sorted(path.glob("rank_*.pt"))]
        return None

    def save(self, key: str, rank: int, data: list):
        path = self.cache_dir / "eval_cache" / key
        path.mkdir(parents=True, exist_ok=True)
        torch.save(data, path / f"rank_{rank:04d}.pt")
```

Note: `data/cache.py` (tokenization cache) and `eval/cache.py` (eval hidden-state cache)
are deliberately separate — they key on different things and live at different lifecycle
points. Don't merge them.

### 4.5 Structured config

```python
# config/schema.py
from pydantic import BaseModel, Field
from typing import Optional, Literal

class ModelConfig(BaseModel):
    target_model_path: str
    draft_model_config: str               # path to JSON
    target_backend: Literal["sglang", "hf", "custom", "sglang_server"] = "sglang"
    trust_remote_code: bool = False
    embedding_key: str = "model.embed_tokens.weight"
    lm_head_key: str = "lm_head.weight"
    is_vlm: bool = False

class DatasetConfig(BaseModel):
    train_data_path: str
    eval_data_path: str = ""
    train_hidden_states_path: str = ""    # non-empty = offline mode
    chat_template: str = "llama3"
    max_length: int = 2048
    train_only_last_turn: bool = False
    num_proc: int = 8

class TrainingConfig(BaseModel):
    strategy: Literal["eagle3", "dflash"] = "eagle3"
    num_epochs: int = 1
    max_steps: int = 10000
    batch_size: int = 1
    learning_rate: float = 3e-4
    warmup_ratio: float = 0.015
    max_grad_norm: float = 0.5
    accumulation_steps: int = 1
    ttt_length: int = 7                   # Eagle3-specific
    block_size: int = 16                  # DFlash-specific
    num_anchors: int = 512                # DFlash-specific
    loss_decay_gamma: Optional[float] = None
    attention_backend: Literal["sdpa", "flex_attention", "fa", "usp"] = "sdpa"
    fsdp_strategy: Literal["NO_SHARD", "SHARD_GRAD_OP", "FULL_SHARD", "HYBRID_SHARD"] = "SHARD_GRAD_OP"
    fsdp_version: Literal[1, 2] = 1       # 2 = FSDP2 (PT 2.4+); see §4.9
    compile_model: bool = False
    tp_size: int = 1
    sp_ulysses_size: int = 1
    sp_ring_size: int = 1
    save_interval: int = 500
    log_interval: int = 10
    max_checkpoints: int = 5              # 0 = keep all
    checkpoint_dir: str = "checkpoints"
    resume: bool = False
    seed: int = 42

class EvalConfig(BaseModel):
    enabled: bool = False
    interval: int = 500
    micro_batch_size: int = 4
    cache_dir: str = "eval_cache"

class LRConfig(BaseModel):
    decay_style: Literal["cosine", "WSD"] = "cosine"
    wsd_decay_steps: int = 0
    wsd_decay_style: Literal["linear", "cosine", "exponential"] = "cosine"

class LoggingConfig(BaseModel):
    report_to: Literal["wandb", "tensorboard", "swanlab", "mlflow", "none"] = "wandb"
    wandb_project: str = "specforge"
    wandb_run_name: str = ""

class SGLangConfig(BaseModel):
    attention_backend: str = "flashinfer"
    # ... other SGLang server args

class Config(BaseModel):
    model: ModelConfig
    dataset: DatasetConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    lr: LRConfig = Field(default_factory=LRConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    sglang: SGLangConfig = Field(default_factory=SGLangConfig)
    output_dir: str = "output"
    cache_dir: str = "cache"
```

YAML example for MLA Eagle3 training:

```yaml
model:
  target_model_path: Qwen/Qwen3-8B
  draft_model_config: specforge/config/draft_configs/qwen3_8b_eagle3_mla.json
  target_backend: sglang

dataset:
  train_data_path: data/train.jsonl
  eval_data_path: data/eval.jsonl
  chat_template: qwen3
  max_length: 4096

training:
  strategy: eagle3
  max_steps: 20000
  batch_size: 2
  learning_rate: 3e-4
  ttt_length: 7
  attention_backend: flex_attention
  accumulation_steps: 2
  tp_size: 4
  max_checkpoints: 3

eval:
  enabled: true
  interval: 1000

output_dir: runs/qwen3_8b_mla_eagle3
```

CLI override: `specforge train --config config.yaml --training.learning_rate=1e-4`

### 4.6 WSD learning rate scheduler

> **Deferred to Phase 6** (was Phase 2 in the original draft). Cosine warmup is sufficient
> for typical draft training runs; WSD is most useful for continued pretraining and long
> stable-phase runs, which aren't on the immediate roadmap.

Add to `specforge/training/lr_scheduler.py`:

```python
class WSDScheduler(_LRScheduler):
    """Warmup → Stable → Decay schedule.

    Stable phase holds max LR until (total_steps - wsd_decay_steps).
    Decay phase applies linear/cosine/exponential decay to min_lr.
    """
    def __init__(self, optimizer, total_steps, warmup_steps, min_lr=0.0,
                 wsd_decay_steps=0, wsd_decay_style="cosine", last_epoch=-1):
        ...
```

Wired via `LRConfig.decay_style == "WSD"` in the trainer.

### 4.7 MLA Eagle3 draft model

Port from TorchSpec's `deepseek_eagle.py`. Key design decisions:

**KV cache in TTT context**: During Eagle3 TTT unroll, cache expanded K/V (not compressed
latent). This matches TorchSpec's approach and avoids requiring MLA-specific cache logic in
`core/eagle3.py`. The MLA compression only happens during projection (forward pass), not in
the cache.

**Cache integration**: use the HF `DynamicCache` pattern (mutate `past_key_values` in place
via `past_key_values.update(k, v, layer_idx, cache_kwargs)`) — same as the existing
`DFlashDraftModel` and `Eagle3DraftModel.backbone()` ABC, which returns `Tensor` (hidden
states) and never a `(Tensor, k, v)` tuple. Returning a tuple would break the base ABC.

```python
# models/drafts/deepseek_eagle3.py
@register_draft("deepseek_v3_eagle3")
class Eagle3DeepseekV2ForCausalLM(Eagle3DraftModel):
    config_class = DeepseekV3Config

    class DeepSeekMLAAttention(nn.Module):
        """MLA attention with Q/KV LoRA, decoupled RoPE."""
        def __init__(self, config: DeepseekV3Config):
            # Q path: q_a_proj → RMSNorm → q_b_proj (if q_lora_rank)
            # KV path: kv_a_proj_with_mqa → split(kv_compressed, k_rope_raw)
            #        → kv_a_layernorm → kv_b_proj → split(k_nope, value)
            # RoPE: interleaved rotation on qk_rope_head_dim slice only
            # o_proj: num_heads * v_head_dim → hidden_size
            ...

        def forward(self, hidden_states, past_key_values=None,
                    attention_mask=None, position_ids=None, cache_position=None,
                    use_cache=False):
            # 1. Project Q (with optional LoRA)
            # 2. Project KV (compressed → expand via kv_b_proj)
            # 3. Split k_nope from kv_b_proj, k_rope_raw from kv_a_proj
            # 4. Apply interleaved RoPE to q_rope and k_rope slices
            # 5. Expand k_rope (MQA → MHA) across heads
            # 6. Concat [k_nope, k_rope] → full expanded key
            # 7. If use_cache and past_key_values is not None:
            #        k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
            # 8. Attention (SDPA or FlexAttention)
            # 9. Return attn_output  — k/v stored in the mutable DynamicCache
            ...
```

Config fields consumed from `DeepseekV3Config`:
- `q_lora_rank` — Q low-rank dim (None = dense Q)
- `kv_lora_rank` — KV compressed dim
- `qk_nope_head_dim` — per-head key dim without RoPE
- `qk_rope_head_dim` — per-head key dim with RoPE
- `v_head_dim` — per-head value dim (may differ from key dim)
- `rope_scaling` — YaRN parameters

**Memory budget for expanded TTT cache**. Per token per draft layer:

    key_bytes_per_token   = num_heads * (qk_nope_head_dim + qk_rope_head_dim) * dtype_size
    value_bytes_per_token = num_heads * v_head_dim                            * dtype_size

For DeepSeek-V3 numbers (128 heads, 128 nope + 64 rope, 128 v_dim, bf16 = 2B):
- per token: 128*(128+64)*2 = 49 152 B key + 128*128*2 = 32 768 B value ≈ 80 KB
- at seq=4096, ttt=7, one draft layer, one sample: ~2.3 GB
- the draft typically has 1 layer, so this scales with `batch_size` not with `num_layers`

So per-GPU peak from the TTT cache ≈ `batch_per_gpu * 2.3 GB`. Bound `batch_size`
accordingly, or revisit the compressed-cache option if this blocks longer-context training.

**TP/SP for MLA — validate before Phase 1 sign-off**. MLA has asymmetric per-head dims
(`qk_nope`, `qk_rope`, `v_head_dim`) — Yunchang USP and any TP slicing assume per-head
homogeneity in places. A small smoke test on Kimi-K2.5 with `tp_size=2`,
`sp_ulysses_size=2` is on the Phase 1 critical path, not a Phase 4 concern.

**Migration for the old `configs/deepseek-v3-671b-eagle3.json`**. That file declares
`model_type: "llama"` and trains a Llama-style draft against a DeepSeek target — it's not
an MLA draft despite the filename. Phase 1 plan:
- Move it to `specforge/config/draft_configs/deepseek_v3_671b_eagle3_llama_draft.json`
  (preserves backward compat for anyone training against it).
- Add the new MLA config at `specforge/config/draft_configs/deepseek_v3_671b_eagle3.json`.
- Emit a one-time deprecation warning when the old path is loaded.

### 4.8 SGLang export tool

```python
# export/to_sglang.py
def export_to_sglang(checkpoint_dir: str, output_dir: str, target_model_path: str,
                     prune_vocab: bool = False, prune_dataset: str = None):
    """Convert FSDP training checkpoint to SGLang-loadable spec-decoder format.

    Handles:
    - FSDP state dict → flat state dict
    - Weight key renaming (training names → SGLang spec-decoder expected names)
    - MLA weight naming: q_a_proj, kv_a_proj_with_mqa, kv_b_proj etc.
    - Optional vocab pruning based on dataset token frequency
    - Config generation (draft config JSON + tokenizer copy)
    """
    ...
```

**Weight-name compatibility is the riskiest single piece of the rewrite** — it's silent
when it goes wrong (loader picks up zeros for missing keys, or refuses to load). Before
Phase 3 starts, produce an explicit two-column map per draft arch:

| Trainer key | SGLang spec-decoder loader key |
|---|---|
| `model.layers.{i}.self_attn.q_a_proj.weight` | _e.g._ `draft_model.layers.{i}.self_attn.q_a_proj.weight` |
| `model.layers.{i}.self_attn.q_a_layernorm.weight` | ... |
| `model.layers.{i}.self_attn.q_b_proj.weight` | ... |
| `model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight` | ... |
| `model.layers.{i}.self_attn.kv_a_layernorm.weight` | ... |
| `model.layers.{i}.self_attn.kv_b_proj.weight` | ... |
| `model.layers.{i}.self_attn.o_proj.weight` | ... |
| `model.embed_tokens.weight` | ... |
| `lm_head.weight` | ... |
| `t2d`, `d2t` (vocab mapping) | ... |

Filling the RHS column requires reading SGLang's current spec-decoding draft loader (the
`Eagle3*` or `LightseekSpec*` loader, whichever is current in sgl-project/sglang) and
should be a documented artifact (`docs/export_weight_map_mla.md`), not implicit in code.

### 4.9 FSDP seam (FSDP1 now, FSDP2-ready)

The plan ships on FSDP1 (`SHARD_GRAD_OP`) but `apply_fsdp` is a stable seam from day one,
gated by `TrainingConfig.fsdp_version`:

```python
# training/fsdp.py
def apply_fsdp(model, training_config, wrap_policy=None):
    if training_config.fsdp_version == 1:
        return _apply_fsdp1(model, training_config, wrap_policy)
    elif training_config.fsdp_version == 2:
        return _apply_fsdp2(model, training_config, wrap_policy)
    raise ValueError(...)
```

Rationale: FSDP2 (PT 2.4+) composes much better with `torch.compile` and per-parameter
sharding (`fully_shard` on individual submodules). The compute-friendly default for the
next 12 months is FSDP2. Pinning to FSDP1 in the Trainer interface means rewriting again
once `compile_model: True` becomes the norm. Implementing `_apply_fsdp2` is deferred to
Phase 7 #31 — the seam is what matters now.

### 4.10 Train-with-decode mode (Phase 5)

W4 from §1 ("Workloads in scope") is a real workload: the same long-lived SGLang server
simultaneously generates training data and serves real spec-decoding traffic, with the
trainer pushing freshly-trained draft weights into that server every N steps. The TorchSpec
implementation does this via Ray actors (`SglEngine.update_weights_from_disk` +
`controller/loop._maybe_sync_draft_weights`). SpecForge gets the same workload by extending
the existing `TargetEngine` / `Stream` / `Trainer` interfaces — **no Ray, no Mooncake**.

Three new primitives, all additive to Phase 4 abstractions:

#### (a) `TargetEngine.update_draft_weights` — weight push API

```python
# models/targets/base.py
class TargetEngine(ABC):
    def update_draft_weights(
        self,
        weights_path: str | os.PathLike,
        *,
        blocking: bool = True,
        load_format: Literal["pt", "safetensors"] = "safetensors",
    ) -> dict[str, Any]:
        """Push new draft weights into a long-lived serving target.

        Default impl raises NotImplementedError. In-process engines (sglang/hf/custom)
        do not need this — they're torn down with the trainer. Implemented by
        `SGLangServerEngine`, which forwards to SGLang's
        `/update_weights_from_disk` endpoint (already present in sglang ≥0.4).
        """
        raise NotImplementedError(f"{type(self).__name__} does not support live draft weight updates")
```

`SGLangServerEngine` adds:

```python
class SGLangServerEngine(TargetEngine):
    def update_draft_weights(self, weights_path, *, blocking=True, load_format="safetensors"):
        return self._post_json(
            "/update_weights_from_disk",
            {"model_path": str(weights_path), "load_format": load_format},
        )

    def serving_metrics(self) -> dict[str, float]:
        """Pull serving-side acceptance rate / queue depth from SGLang.

        Mapped to wandb keys like serving/spec_accept_rate, serving/spec_accept_length.
        """
        return self._get_json("/spec_decoding_stats")
```

The shared filesystem assumption is the same one TorchSpec already lives with: trainer
saves to a path readable by the SGLang server (NFS, or rank-0 saves and broadcasts the
path; the server reads from that path).

#### (b) `ServingTrafficStream` — train on real serving prompts (optional)

For pure W4 you can keep using static jsonl with `OnlineStream`/`RemoteStream` — the
training data has nothing to do with the serving traffic, just the engine is shared. But
**true on-policy training** wants to sample prompts from real serving traffic. New stream:

```python
# data/streams/serving_traffic.py
class ServingTrafficStream(RemoteStream):
    """RemoteStream variant that pulls prompts from a serving-traffic buffer
    instead of a static dataset.

    Buffer backends in scope:
    - file: jsonl tail (simplest; rotated by serving)
    - redis: list with LPUSH (serving) / BRPOP (this stream)
    - kafka: topic with consumer group

    Out of scope: anything that requires Ray.
    """
    def __init__(
        self,
        buffer_uri: str,           # "file:///var/log/sgl/prompts.jsonl"
                                   # "redis://host:6379/0/serving_prompts"
        target_engine: SGLangServerEngine,
        sample_rate: float = 1.0,  # PII-safe rate limiting
        cold_start_jsonl: Optional[str] = None,  # bootstrap when buffer is empty
        **kwargs,
    ): ...
```

The privacy / PII / sample-rate decisions live with whoever runs the serving cluster,
**not in the trainer**. Stream just consumes whatever it's pointed at.

#### (c) `Trainer._maybe_sync_draft_weights` — periodic push hook

```python
# training/trainer.py — addition to the loop in §4.2
class Trainer:
    def _maybe_sync_draft_weights(self, step: int) -> None:
        """Save draft state and call target_engine.update_draft_weights.

        Mirrors TorchSpec's _maybe_sync_draft_weights (controller/loop.py:42-73)
        but stays in-trainer. No Ray needed because there's exactly one engine
        to update (one trainer ↔ one serving SGLang).
        """
        cfg = self.config.weight_sync
        if not cfg.enabled or self._target_engine is None:
            return
        if (step + 1) % cfg.interval != 0:
            return

        with rank_0_priority():
            tmp = Path(cfg.tmp_dir or self.config.checkpoint_dir) / "draft_weight_sync"
            tmp.mkdir(parents=True, exist_ok=True)
            self._save_draft_for_serving(tmp)        # FSDP full-state -> safetensors

        # Only rank 0 actually issues the HTTP call; barrier so other ranks wait.
        if dist.get_rank() == 0:
            self._target_engine.update_draft_weights(tmp, load_format="safetensors")
        dist.barrier()
```

Wired into the existing `_maybe_log / _maybe_eval / _maybe_save` chain. ~30 LOC.

#### (d) Decode-mode `SGLangServerEngine` — server-side config

The serving SGLang must be launched with both prefill+aux capture (for training data
generation) **and** spec decoding (for serving traffic) enabled simultaneously. This is
already supported by sglang ≥0.4 via the `--enable-aux-hidden-states` +
`--enable-spec-training-mooncake=False` (we don't need mooncake) +
`--enable-cuda-graph=True` combination. `SGLangServerEngine` gets a `decode_mode: bool`
config flag that simply governs which set of server-args gets validated; it doesn't
launch the server itself (operations responsibility).

#### Acceptance-rate as a first-class metric

In W1-W3 the relevant metric is `simulated_acc_len` (computed offline from logged
per-position accuracy). In W4, the **real serving acceptance rate** is available from
the same SGLang server. The `Trainer.train()` loop calls `target_engine.serving_metrics()`
on the same cadence as eval and merges into the tracker dict:

```python
if self.config.weight_sync.report_serving_metrics and step % self.config.eval.interval == 0:
    sm = self._target_engine.serving_metrics()
    self.tracker.log({
        "serving/spec_accept_rate": sm["accept_rate"],
        "serving/spec_accept_length": sm["accept_length"],
    }, step=step)
```

**Why this is sufficient (no Ray, no Mooncake)**:

- One trainer ↔ one server is a hard scope boundary (§8 non-goal). The dispatcher /
  multi-engine load balancing that TorchSpec's `AsyncInferenceManager` provides isn't
  needed at this scope.
- Weight push is HTTP `/update_weights_from_disk`; SGLang already implements this for
  serving updates, no new server work.
- Training data path stays identical to W2/W3 — `RemoteStream` /
  `ServingTrafficStream` produce `TrainBatch` exactly the same way.
- The trainer has no idea decode mode is on. It just pushes weights and reads metrics.

If profiling later shows that the HTTP weight-push latency itself is a problem (saving +
loading 100MB-1GB of weights every 500 steps), the optimization is to upgrade
`update_draft_weights` to accept an in-memory state_dict + zero-copy transfer, again
without changing the trainer. Same shape as the L2 transport upgrade discussed in §6.

---

## 5. Feature List (Prioritized)

**Ordering note (from review):** the original draft put the unified `Trainer` in Phase 2
and the `TargetEngine` / `HiddenStateStream` abstractions in Phase 4. That meant Phase 2
would build the trainer against the *old* `Eagle3TargetModel`/`DFlashTargetModel`
interface and then Phase 4 would re-plumb it. Reordered so the interfaces (and a single
in-process implementation of each) land in Phase 2 alongside the trainer; Phase 4 then
adds new implementations (`SGLangServerEngine`, `RemoteStream`) of stable interfaces
rather than re-plumbing.

### Phase 1: MLA Draft + Registry (Week 1-2)

| # | Feature | Why | Effort |
|---|---------|-----|--------|
| 1 | Draft plugin registry (`@register_draft`) | Every new arch today touches 4 files. This is the single biggest extensibility blocker. | S |
| 2 | MLA Eagle3 draft (`Eagle3DeepseekV2ForCausalLM`) | The gap for DeepSeek-V3 / Kimi-K2. Port from TorchSpec. Includes MLA-aware KV cache handling via HF `DynamicCache` (no ABC break). | M |
| 3 | MLA draft configs | `qwen3_8b_eagle3_mla.json`, `kimi_k25_eagle3_mla.json`, `deepseek_v3_671b_eagle3.json` (real MLA, not Llama pretending). | S |
| 3a | **MLA + TP/SP smoke test** | Validate Yunchang USP and TP slicing on Kimi-K2.5 with asymmetric MLA head dims before declaring Phase 1 done. | S |

**Deliverable**: Train an MLA Eagle3 draft for Qwen3-8B / Kimi-K2.5 using existing `train_eagle3.py`.
Existing scripts unchanged — registry lives alongside the old `_model_mapping` dicts.

### Phase 2: Interfaces + Trainer + Eval + Checkpoints (Week 3-6)

| # | Feature | Why | Effort |
|---|---------|-----|--------|
| 4 | `TargetEngine` protocol + in-process impls | Define the interface up front so the trainer is built against it from day 1. Adapt existing `Eagle3TargetModel` / `DFlashTargetModel` as the initial concrete impls. | M |
| 5 | `HiddenStateStream` protocol + `OnlineStream` + `OfflineStream` | Same reason — stable abstraction before trainer code is written. `RemoteStream` deferred to Phase 4. | M |
| 6 | `Trainer` class + `DraftTrainStrategy` protocol | Collapse `train_eagle3.py` and `train_dflash.py`. Stop the script-per-arch sprawl. Trainer consumes `TrainBatch` only — no direct target-model calls. | M |
| 7 | `Eagle3TTTStrategy` + `DFlashBlockStrategy` | Extract from existing scripts into strategy implementations. Each declares its own `fsdp_wrap_policy()`. | M |
| 8 | `CheckpointManager` | Rotation (`max_checkpoints`), best tracking, `latest_checkpointed_iteration.txt`, `meta.json`. Must integrate with `stream.seek()` on resume. | S |
| 9 | `Evaluator` + `EvalCache` | Online eval during training: `simulated_acc_len`, `avg_loss`, `avg_acc`. Per-position acc aggregated across batches before geometric sum. Cache key covers eval/target/tokenizer/template/aux/seqlen. | M |
| 10 | Gradient accumulation correctness | `model.no_sync()` on non-sync micro-steps, explicit `zero_grad`, `accumulation_steps` in config. Verify by step-time profiling. | S |
| 11 | `target_layer_ids` as first-class contract | EAGLE3 hardcodes 3 aux layers; DFlash uses a list. Generalize so every draft declares what it needs and the stream materializes accordingly. | S |
| 12 | FSDP seam (FSDP1 default, FSDP2-ready) | `apply_fsdp` dispatches on `fsdp_version`. FSDP2 impl deferred — but the seam is stable now so the trainer doesn't get rewritten when FSDP2 lands. | S |

**Deliverable**: `Trainer` (no CLI yet — legacy scripts call into it) works for both Eagle3
and DFlash, using the new interfaces with single in-process target/stream impls. Eval
metrics logged during training. Best checkpoint auto-saved. Numerical equivalence (§10)
checked against the pre-refactor scripts.

### Phase 3: Config + Export (Week 7-8)

| # | Feature | Why | Effort |
|---|---------|-----|--------|
| 13 | Pydantic config schema | Replace argparse + JSON split with one validated YAML per run. | S-M |
| 14 | CLI (`specforge train\|prepare\|export\|eval`) | Single entry point. Legacy scripts now become shims that build a `Config` and call `Trainer`. | S |
| 15 | SGLang export tool | Checkpoint → SGLang spec-decoder layout. **Weight-name map (§4.8) is a documented artifact, finalized before this phase starts.** | M |
| 16 | HF export tool + vocab pruning | FSDP checkpoint → HF format. Optional vocab pruning by dataset frequency. | S-M |

**Deliverable**: Complete train → eval → export → serve pipeline in one tool. Old MLA
draft trained in Phase 1 successfully loads in a SGLang server.

### Phase 4: Remote Target + VLM + Parsers (Week 9-11)

| # | Feature | Why | Effort |
|---|---------|-----|--------|
| 17 | `SGLangServerEngine` | Target as HTTP service. Unlocks online training against 671B targets on separate GPUs. New impl of the Phase-2 `TargetEngine` interface. | M-H |
| 18 | `RemoteStream` | Stream backend that talks to `SGLangServerEngine`. `prefetch_factor` + `max_in_flight` back-pressure. | M |
| 19 | VLM unification | Delete `QwenVLOnlineEagle3Model`. VLM handled via typed `MediaInputs` in `TargetEngine` + data pipeline. | M |
| 20 | Additional parsers | `KimiK25Parser`, `MiniMaxParser` from TorchSpec. | S |

**Deliverable**: Train Eagle3 for DeepSeek-V3 with the target running on a separate SGLang
server cluster.

### Phase 5: Train-with-decode (Week 12-14)

W4 from §1 — promoted from Phase 6 because it's a real workload, not a future option. All
features build on the Phase 4 `SGLangServerEngine` + `RemoteStream` foundation; **no
trainer / strategy / draft changes**.

| # | Feature | Why | Effort |
|---|---------|-----|--------|
| 21 | `TargetEngine.update_draft_weights` | Weight push API on the ABC; default raises NotImplementedError. `SGLangServerEngine` impl forwards to SGLang's `/update_weights_from_disk`. | S |
| 22 | Decode-mode `SGLangServerEngine` | `decode_mode: bool` config + `serving_metrics()` endpoint. Validates that the server was launched with prefill+aux **and** spec decoding both enabled. | S-M |
| 23 | `Trainer._maybe_sync_draft_weights` | Periodic push hook in the main loop. ~30 LOC. Save draft state on rank 0 → engine.update_draft_weights → barrier. | S |
| 24 | `ServingTrafficStream` | Optional stream that pulls prompts from a serving-traffic buffer (file / Redis / Kafka). Subclass of `RemoteStream`; cold-start fallback to a static jsonl. | M |
| 24a | Acceptance-rate metrics | Trainer reads `serving/spec_accept_rate` + `serving/spec_accept_length` from the engine and logs alongside `eval/simulated_acc_len`. Closes the gap between offline proxy and production reality. | S |

**Deliverable**: A long-lived SGLang server simultaneously serves real spec-decoding
traffic and produces training data; the trainer pushes draft weights every N steps and
production acceptance rate trends up over the run, monotonically across `weight_sync_interval`.
End-to-end smoke test: 1k-step run on a small target, acceptance-rate slope > 0.

### Phase 6: Polish (Week 15-16)

| # | Feature | Why | Effort |
|---|---------|-----|--------|
| 25 | Backbone-agnostic DFlash | Factor `DFlashDraftModel` so MLP/Norm/RoPE are parameterized. Currently hardcoded to Qwen3. | S-M |
| 26 | `torch.compile` support | Optional `compile_model` flag for draft model compilation. | S |
| 27 | `defer_tokenization` + dynamic loss mask | Tokenize at fetch time, not preprocessing time. Enables dynamic loss masking. | M |
| 28 | Tokenization disk cache | Cache tokenized datasets keyed by data path + template + max_length. | S |
| 29 | WSD learning rate scheduler | Warmup-Stable-Decay. Useful for long runs / continued pretraining; defer unless a concrete need surfaces. | S |
| 30 | Remove legacy `scripts/legacy/` shims | After two minor releases with `DeprecationWarning`. Concrete target: SpecForge 0.X. | S |

### Phase 7: Optional / Future

| # | Feature | Why | Effort |
|---|---------|-----|--------|
| 31 | FSDP2 implementation | Fill in `_apply_fsdp2` behind the Phase-2 seam. Required if `torch.compile` becomes default. | M |
| 32 | Mooncake streaming backend | GPU↔GPU tensor transport for multi-node disaggregation. Only if profiling shows HTTP/gRPC is the bottleneck (T1 trigger, §6). | H |
| 33 | In-memory `update_draft_weights` | Bypass save-to-disk in W4 weight sync; transfer state_dict directly over a binary channel. Only if 100MB-1GB / N-step push becomes a measurable slowdown. | M |
| 34 | Multi-job inference pool sharing | Single SGLang cluster amortized across ≥5 concurrent training jobs (T2 trigger, §6). At this point the actor topology becomes worth the complexity. | H |
| 35 | FA4 attention backend | FlashAttention 4 support for draft attention. | S-M |

---

## 6. Tradeoffs Worth Flagging

### Ray + Mooncake vs HTTP/gRPC

TorchSpec gets a lot from Ray + Mooncake, but they're large dependencies with significant
operational complexity. For SpecForge, start with HTTP/gRPC to a SGLang server for cross-node
training — this covers most use cases (target on dedicated GPUs, trainer on others) at a
fraction of the surface area.

**Train-with-decode (W4) is in scope without Ray.** The TorchSpec value-add from actor
topology is concentrated in *multi-job inference pool sharing* and *zero-overhead async
producer-consumer pipelines*. For one-trainer-↔-one-server (which is W4's scope per §8),
HTTP `/update_weights_from_disk` plus a `ServingTrafficStream` over a Redis/Kafka/file
buffer is sufficient. See §4.10. Ray actor topology only starts paying off at the T2
trigger below.

**Triggers that would force a re-evaluation:**

- **T1 — transport bottleneck**: profile shows `last_hidden_states` over JSON/HTTP
  consumes >30% of step time, with target ≥70B. Response: upgrade `RemoteStream`
  transport to a binary protocol (gRPC + protobuf, or Mooncake-as-transport).
  **L1/L3 unchanged.** This is feature #32 in Phase 7.

- **T2 — multi-job inference pool sharing**: ≥5 concurrent training jobs hammering the
  same 671B target, each tearing the engine up/down is wasteful. Response: introduce
  Ray actor pool with `AsyncInferenceManager`-style dispatcher. This is the only
  trigger that genuinely justifies adopting Ray; until then, complexity is pure cost.
  Feature #34 in Phase 7.

- **T3 — Mooncake transport**: only meaningful after T1; even then, evaluate gRPC
  binary first. Mooncake's RDMA edge matters for >100GB/s aggregate, which neither W3
  nor W4 typically hit. Feature #32 in Phase 7.

The bet: 90% of the time we're in W1-W3 territory and the refactor's interface boundaries
do real work. W4 lands in Phase 5 with ~200 LOC of additive code on those boundaries.
TorchSpec's full topology is reserved for the day T2 actually fires.

### Plugin registry vs HF AutoModel pattern

HF's "drop in a config JSON" is a nice affordance. Keep both: decorator registry is primary
for dispatch, HF type dispatch as fallback for configs that specify a known `transformers`
config class. `AutoDraftModelConfig.from_file()` checks `DRAFT_REGISTRY` by architecture name
first, falls back to config type mapping.

### Strategy pattern is an indirection

If only EAGLE3 and DFlash ever exist, two scripts are fine. The bet is that Medusa, MTP, or
hybrid variants will want the same trainer infrastructure — if so, strategies pay off fast.
If not, the cost is one extra level of dispatch that doesn't hurt readability.

### Pydantic vs OmegaConf

Pydantic gives better validation and serialization. OmegaConf gives effortless YAML merge +
CLI override (`--training.lr=1e-4`). Options:
1. **Pydantic + custom CLI overlay** (parse `--key=value` args, `model_validate` from merged dict) — our choice.
2. OmegaConf + dataclass (TorchSpec's approach).
3. Pydantic + typer/click for CLI.

We go with (1) because SpecForge already uses Pydantic (`ChatTemplate`) and validation
matters more than merge ergonomics at this stage. The CLI overlay is ~30 lines of code.

### MLA cache: compressed vs expanded

Two options for KV cache during Eagle3 TTT unroll:
- **Compressed**: Cache `kv_compressed` (low-rank) + `k_rope_raw`. Saves memory but requires
  MLA-specific cache logic in `core/eagle3.py`.
- **Expanded**: Cache full `(K, V)` after projection. More memory but `core/eagle3.py` is
  architecture-agnostic.

We choose **expanded** (matching TorchSpec). The TTT unroll is typically 5-7 steps with
short sequences — memory savings from compressed cache are marginal, and keeping
`core/eagle3.py` architecture-agnostic is worth more.

---

## 7. Migration Path

Each phase ships independently and can be validated without breaking existing users.

### Phase 1 plan

1. Add `models/drafts/__init__.py` with `DRAFT_REGISTRY` and `@register_draft`.
2. Move `LlamaForCausalLMEagle3` to `models/drafts/llama_eagle3.py`, decorate with
   `@register_draft("LlamaForCausalLMEagle3")`.
3. Port `deepseek_eagle.py` from TorchSpec → `models/drafts/deepseek_eagle3.py`, decorate
   with `@register_draft("Eagle3DeepseekV2ForCausalLM")`.
4. Update `AutoDraftModelConfig.from_file()` to check `DRAFT_REGISTRY` first.
5. Update `AutoEagle3DraftModel.from_config()` to check `DRAFT_REGISTRY` first.
6. Add MLA draft configs (`qwen3_8b_eagle3_mla.json`, `kimi_k25_eagle3_mla.json`).
7. Keep old `_model_mapping` / `_config_mapping` as fallback — remove later.
8. Test: train MLA Eagle3 draft using existing `train_eagle3.py` — zero changes to script.

### Phase 2 plan

1. Define `TargetEngine` protocol in `models/targets/base.py`.
2. Adapt existing `SGLangEagle3TargetModel` → `SGLangTargetEngine` (in-process).
3. Adapt existing `HFEagle3TargetModel` → `HFTargetEngine`.
4. Adapt existing `CustomEagle3TargetModel` → `CustomTargetEngine`.
5. Define `HiddenStateStream` protocol + `OnlineStream` + `OfflineStream`. `seek()` method
   required for resume correctness. `OnlineStream` exposes `prefetch_factor`.
6. Create `training/trainer.py` with the `Trainer` class. Trainer consumes `TrainBatch`
   only, never a target model directly.
7. Extract `Eagle3TTTStrategy` from `train_eagle3.py` logic. Implement `fsdp_wrap_policy()`.
8. Extract `DFlashBlockStrategy` from `train_dflash.py` logic. Implement `fsdp_wrap_policy()`.
9. Add the `apply_fsdp` seam dispatching on `fsdp_version` (FSDP1 only for now).
10. Implement `CheckpointManager` — `save` / `load` / `update_best` / `_rotate`.
11. Implement `Evaluator` (per-position acc aggregation across batches) + `EvalCache`
    (full-key MD5).
12. Move old scripts to `scripts/legacy/`, create shims that instantiate `Trainer`.
13. **Numerical equivalence test (§10)**: legacy shim must match the pre-refactor script's
    loss curve to within tolerance on a fixed seed at steps 100/500/1000. Gate.

### Phase 3 plan

1. Define `config/schema.py` with Pydantic models (Literal-typed enums).
2. Implement `config/loader.py` (YAML load + CLI override).
3. Implement `cli.py` (`specforge train|prepare|export|eval`).
4. Finalize the SGLang weight-name map (`docs/export_weight_map_mla.md`) by reading the
   current SGLang spec-decoding loader. Block #5 on this.
5. Implement `export/to_sglang.py` and `export/to_hf.py`.
6. Mechanically migrate existing JSON configs to YAML with the new schema.
7. Test: `specforge train --config x.yaml` end-to-end. MLA draft from Phase 1 successfully
   loads in a SGLang server with non-trivial acceptance rate.

### Phase 4 plan

1. Implement `SGLangServerEngine` (HTTP client to SGLang server). Reuses the Phase-2
   `TargetEngine` interface — no trainer changes.
2. Implement `RemoteStream` (`prefetch_factor`, `max_in_flight`). Reuses the Phase-2
   `HiddenStateStream` interface — no trainer changes.
3. Add typed `MediaInputs` to `TargetEngine.generate_train_data`.
4. Delete `QwenVLOnlineEagle3Model`; wire VLM through `MediaInputs` + data pipeline.
5. Port `KimiK25Parser`, `MiniMaxParser` from TorchSpec.
6. Test: online training with SGLang server on separate node; results within tolerance of
   in-process training (same target weights).

### Phase 5 plan

1. Add `update_draft_weights(weights_path, *, blocking, load_format)` to `TargetEngine`
   ABC with `NotImplementedError` default.
2. Implement `SGLangServerEngine.update_draft_weights` → POST `/update_weights_from_disk`.
   Implement `serving_metrics()` → GET `/spec_decoding_stats`.
3. Add `decode_mode: bool` to `SGLangServerEngine`. Validate that the operator's server
   was launched with prefill+aux **and** spec decoding both enabled (warn loudly otherwise).
4. Add `Trainer._maybe_sync_draft_weights(step)` hook between `_maybe_eval` and
   `_maybe_save`. Save FSDP full-state on rank 0 → call engine update on rank 0 → barrier.
5. Implement `ServingTrafficStream(RemoteStream)` with file/Redis/Kafka buffer backends
   and a cold-start jsonl fallback. PII / sample-rate are operator concerns, not
   trainer concerns.
6. Wire `serving/spec_accept_rate` and `serving/spec_accept_length` through the existing
   `Tracker` (every `eval_interval` steps).
7. **End-to-end test**: 1k-step run with a small target (Qwen3-8B) co-located with a
   spec-decoding load generator. Acceptance-rate slope across `weight_sync_interval`
   buckets must be > 0; final acceptance rate must exceed cold-start by ≥ X%
   (X TBD — set baseline from Phase 4 numbers).
8. **Weight-sync correctness test (§10.5)**: before/after a sync, the served draft's
   token-level outputs match the trainer's draft on a 32-prompt fixed eval set within
   acceptance-rate tolerance.

---

## 8. Non-Goals (Explicit)

- **No Ray dependency.** SpecForge stays torchrun-native, including for train-with-decode (W4).
- **No Mooncake in Phase 1-6.** HTTP/gRPC to SGLang is sufficient for both disaggregated
  training (W3) and train-with-decode (W4). Mooncake is gated behind T1/T3 in §6.
- **No vLLM target backend.** SGLang is primary; HF is the reference. Adding vLLM is possible
  via the `TargetEngine` interface but not prioritized.
- **No multi-engine load balancing / multi-job inference pool sharing.** Exactly **one**
  target engine instance per training job — including in train-with-decode (one trainer
  ↔ one server). Scaling within a single engine is via SGLang's own scaling (multiple
  TP workers, replicas behind a single endpoint). Multi-job sharing is gated behind T2
  in §6 and lands as Phase 7 #34.
- **No on-policy serving-prompt mining without a buffer.** `ServingTrafficStream` reads
  from an external buffer (Redis/Kafka/file) populated by the serving cluster; SpecForge
  does **not** intercept inflight serving requests directly.

---

## 9. Success Criteria

| Phase | Criterion |
|-------|-----------|
| 1 | MLA Eagle3 draft trains on Qwen3-8B and converges to comparable loss as Llama draft. Kimi-K2.5 TP/SP smoke test passes. |
| 2 | Legacy shims (using new `Trainer` internally) match pre-refactor scripts to numerical-equivalence tolerance (§10). Eval metrics agree with manual eval. |
| 3 | `specforge train --config x.yaml` end-to-end. MLA draft from Phase 1 loads in SGLang and produces non-trivial acceptance rate. |
| 4 | Online training with 671B target on separate SGLang server; per-step loss within tolerance of in-process baseline using identical target weights. |
| 5 | **Train-with-decode**: 1k-step run; one long-lived SGLang server simultaneously serves spec-decoding traffic and produces training data; trainer pushes draft weights every `weight_sync_interval`; serving acceptance rate trends up monotonically across sync buckets. Weight-sync correctness gate §10.5 passes. |
| 6 | DFlash on Llama backbone; tokenization cache cuts data prep time by >50%; legacy shims removed. |

---

## 10. Testing Strategy (gating, not optional)

Every phase that changes the training path must pass a numerical-equivalence gate against
the immediately prior commit. Treat these as CI requirements, not nice-to-haves —
behavior-preserving refactors are the most common place silent regressions hide.

### 10.1 Numerical-equivalence gate (Phase 2, 4 critical)

For a fixed seed and a fixed micro-batch sequence (3 batches × 4 samples is enough), the
new code path must match the old:

| Metric | Tolerance | Steps to check |
|---|---|---|
| Per-step training loss | `atol=1e-4, rtol=1e-4` (BF16 master copy) | 0, 1, 100, 500, 1000 |
| Per-position eval acc (vector of length `ttt_length`) | `atol=1e-3` | end of eval at step 1000 |
| `simulated_acc_len` | `atol=1e-3` | end of eval at step 1000 |
| Model state dict (selected keys) | `atol=1e-4` | after step 100 |

The gate run uses: Llama Eagle3 draft, Qwen3-8B target, offline mode, sharegpt eval slice.
Cheap enough to run on every PR that touches `core/`, `training/`, or `models/drafts/`.

### 10.2 Smoke tests (every phase)

- One config per draft arch (`llama_eagle3`, `deepseek_v3_eagle3`, `qwen3_dflash`) trains
  for 20 steps without crashing under TP=1, TP=2, and TP=2+SP=2.
- Checkpoint save + resume produces the same loss curve as no-resume (validates
  `stream.seek()`).
- Eval cache miss + hit produce identical metrics (validates the cache key set).

### 10.3 Distributed correctness (Phase 1 + 2)

- MLA + Yunchang USP with `qk_nope_head_dim != qk_rope_head_dim != v_head_dim` on
  Kimi-K2.5. This is the Phase 1 risk item and must be on the critical path, not the
  Phase 4 wishlist.
- Gradient accumulation: confirm one `all_reduce` per `optimizer.step()`, not per
  `backward()`. Use a NCCL communicator log or `torch.profiler` once.

### 10.4 Export-loop test (Phase 3)

- Train MLA draft for 100 steps, export to SGLang, load in SGLang server, run 32 generation
  requests. Acceptance rate > 0 (i.e. the loader actually consumed the weights, not zeros).
  This catches weight-name-map regressions at the integration boundary.

### 10.5 Weight-sync correctness gate (Phase 5)

Every PR that touches `Trainer._maybe_sync_draft_weights`, `SGLangServerEngine.update_draft_weights`,
or the FSDP-state-save path must pass a parity gate:

| Step | Action | Pass condition |
|---|---|---|
| 1 | Train 50 steps; capture trainer-side draft state (full FSDP state dict, broadcast to rank 0). | — |
| 2 | Call `Trainer._maybe_sync_draft_weights(50)`. | Engine returns 200; serving SGLang reports `update_weights/success=True`. |
| 3 | Run 32 fixed-seed prompts through both: (a) trainer's local draft (offline-style forward), (b) the now-updated serving SGLang. | Logits agree to `atol=1e-2, rtol=1e-2` (allowing for kernel diffs between training and serving stacks). |
| 4 | Compare serving acceptance rate over 256 prompts before vs after sync. | Acceptance rate **non-decreasing**; if 50 training steps moved the loss meaningfully, acceptance rate strictly increases. |

The gate fails if any of: (i) the HTTP push silently 200-no-ops, (ii) the serving stack
loaded a corrupt or partial state dict, (iii) FSDP state-dict gather missed a parameter.
This is the train-with-decode analog of the §10.4 export-loop test.

### 10.6 Long-run weight-sync soak (Phase 5, optional)

- 1k-step run with `weight_sync_interval=100`. Bucket serving acceptance rate by
  100-step windows; the slope across buckets must be monotonically non-decreasing
  (allowing one regression bucket per 10 — to absorb rare unlucky prompt batches).
- Watches for: (a) silent weight push failures (acceptance rate would plateau),
  (b) draft drift due to incorrect FSDP gather, (c) serving-side OOM that quietly
  rolls back to old weights.
