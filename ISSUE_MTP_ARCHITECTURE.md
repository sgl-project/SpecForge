# [RFC/Discussion] MTP training architecture: target-coupled draft, multi-model scaling, and open design questions

## Context

A working MTP (Multi-Token Prediction) training prototype has landed on a fork branch ([curnane-lab/SpecForge_npu, `add_mtp_support`](https://github.com/curnane-lab/SpecForge_npu/tree/add_mtp_support)) — single-layer Qwen3.5 MTP, online target-hidden extraction via the existing SGLang/HF target backends, FSDP training, and a `merge_mtp_to_base.py` helper that produces a single served checkpoint. Training runs end-to-end on NPU/CUDA.

Per the [2026 roadmap (E2 — Algorithm breadth)](https://github.com/sgl-project/SpecForge/blob/main/docs/roadmap/eval-and-breadth.md), MTP is slated to land as a first-class draft algorithm. Before polishing the PR, I'd like to align on the **architectural shape** of MTP training, because MTP differs structurally from EAGLE3/DFlash in a way that affects scaling, loss design, and registry integration.

## The core property: MTP draft is *target-coupled*

Unlike EAGLE3/DFlash, whose draft is an **architecture-agnostic drafter** (a custom 1-layer net that consumes target hidden states and is loaded by its *own* inference code), an MTP draft must **mirror the target model's own MTP module**, because its checkpoint is loaded by SGLang's per-target MTP class.

Concrete code evidence (SGLang `srt/models/qwen3_5_mtp.py`):

```python
# Qwen3_5ForCausalLMMTP.__init__
self.fc = nn.Linear(2 * hidden, hidden, bias=False)
self.pre_fc_norm_embedding = GemmaRMSNorm(...)
self.pre_fc_norm_hidden   = GemmaRMSNorm(...)
config.num_hidden_layers = 1
self.model = Qwen3_5ForCausalLM(config, ...)   # ← wraps the TARGET's ForCausalLM class
self.lm_head = ...  # tied to embed_tokens, or independent ParallelLMHead

# load_weights
params_dict = dict(self.named_parameters())
for name, w in weights:
    if "mtp" not in name: continue
    name = name.replace("mtp.", "model.")           # mtp.layers.0.* -> model.layers.0.*
    name = name.replace("model.fc", "fc")
    name = name.replace("model.pre_fc", "pre_fc")
    if ".self_attn." in name: name = name.replace(".self_attn", "")
```

SGLang's `Qwen3_5ForCausalLM` is **flat** (`self.layers` directly on the ForCausalLM, *not* nested under `self.model` like HuggingFace). So `Qwen3_5ForCausalLMMTP.self.model = Qwen3_5ForCausalLM` yields the param path `model.layers.0.*` (single `model.`). The remap `mtp.`->`model.` turns a checkpoint key `mtp.layers.0.*` into `model.layers.0.*` to match. The **training-side draft must save `mtp.layers.0.*`** (not `mtp.model.layers.0.*`) or the transformer weights silently fail to load at inference. This flat layout is confirmed by the native Qwen3.5-4B checkpoint, which ships 15 `mtp.*` keys (`mtp.layers.0.*`, `mtp.fc.weight`, `mtp.norm.weight`, `mtp.pre_fc_norm_*`).

Each target architecture has its **own** SGLang MTP class (`qwen3_5_mtp.py`, `qwen3_next_mtp.py`, `nemotron_h_mtp.py`, `mimo_mtp.py`, `step3p5_mtp.py`, `exaone_moe_mtp.py`), each wrapping its own ForCausalLM with its own `load_weights` remapping. There is no generic MTP class.

## How this differs from EAGLE3 / DFlash

| | EAGLE3 | DFlash | MTP |
|---|---|---|---|
| draft architecture | 1 Llama-styled class (architecture-agnostic) | 1 Qwen3-styled class (architecture-agnostic) | **1 class per target family** (mirrors target MTP module) |
| checkpoint loaded by | EAGLE3's own inference code | DFlash's own inference code | **target's SGLang per-target MTP class** |
| adding a new target | config/vocab-mapping only, zero new draft code | config only, zero new code | **per-family draft code** (same family may parameterize; different family = new class) |
| draft loaded as | independent drafter | independent drafter | **embedded into the target checkpoint** (via `merge_mtp_to_base.py`) |

The coupling is **inherent** to "MTP = train the target's native MTP module" — it's not a design choice the training framework can sidestep as long as inference goes through SGLang's per-target MTP class.

## Open questions for discussion

### 1. Is target-coupling the accepted long-term shape, or do we want a SpecForge-controlled inference path?

The coupling means SpecForge's MTP draft must track every SGLang `*_mtp.py`'s weight layout. Alternatives: (a) accept the coupling and document the per-family contract; (b) a SpecForge-side generic MTP inference module that owns its own weight format (decoupling train from SGLang's per-target classes, at the cost of not using the target's native MTP inference). (a) is pragmatic; (b) is more work and diverges from "native MTP". Lean toward (a)?

### 2. Multi-model scaling: per-family classes vs parameterized backbone

Today the draft hardcodes Qwen3.5 (`Qwen3_5MTPDraftModel`). Inspection of SGLang's MTP implementations shows they cluster into ~2 structural families:
- **fc + ForCausalLM wrapper** (Qwen3.5, Qwen3Next, ExaOne-MoE): near-identical forward, differ only in backbone class / norm class / head sharing.
- **eh_proj + decoder layer + final_norm** (MiMo, Step3.5, Nemotron-H): different fusion/norm pattern; Nemotron-H adds multi-layer `pattern`.

For the first family, could we parameterize the backbone class (e.g. infer `decoder_layer_cls`/`norm_cls` from the target at runtime, à la [transformers #46229](https://github.com/huggingface/transformers/pull/46229)'s `MtpLayer(config, decoder_layer_cls, norm_cls)`) so one class serves Qwen3.5/Qwen3Next/ExaOne with config-only changes? Or is per-target mirroring (one class each) safer given the strict SGLang weight-layout contract? The second family likely needs its own class regardless.

### 3. Alignment with the strategy registry (Phase A / E2)

EAGLE3/DFlash have migrated to `*TrainStrategy` (`runtime/training/strategy.py`); MTP currently uses the legacy path (`core/mtp.py` + `scripts/train_mtp.py`), consistent with peagle. E2's stated direction is "new algorithm = `StrategySpec` + a loss". Should MTP:
- adopt `MTPTrainStrategy` once the composable-launch registry (Phase A) lands, and
- already shape `OnlineMTPModel.forward`'s return to match the `DraftTrainStrategy.forward_loss` contract (per-position `acc_corrects`/`acc_denoms` lists — currently it already returns length-1 lists, which is E1-evaluator compatible)?

### 4. Loss design: hard-label CE only, or optional KL distillation?

MTP currently trains with next-token CE on the dataset's tokens; `generate_mtp_data` returns `target=None`/`return_logits=False`, so there's no target-distribution signal. EAGLE3 has a KL term. For speculative decoding the draft must mimic the target's distribution; without KL, MTP is more sensitive to data–target distribution mismatch (heavier reliance on `regenerate_train_data.py` to regenerate responses with the target). Should we add an optional KL path (fetch target logits in `generate_mtp_data` when `loss_type="kl"`)? Tradeoff: extra target forward cost / memory vs. reduced dependence on regenerated data.

### 5. Multi-layer MTP

Qwen3.5 is single-layer (`num_nextn_predict_layers=1`), but Nemotron-H and future models are multi-layer. `OnlineMTPModel` already returns per-position metric *lists* (length 1 now), and `ploss_decay` is plumbed but unused. Is extending to N layers (loop + `ploss_decay` weighting, EAGLE3-TTT-style per-position accuracy) the right shape, and should the draft model's `self.layers` (`ModuleList`, currently length 1) grow to N layers from the start, or stay single-layer until a multi-layer target actually lands?

### 6. (minor) Checkpoint hygiene & round-trip CI

- **Resolved in prototype**: `save_checkpoint` now strips frozen/shared `embed_tokens` and `mtp.lm_head.weight` (matching `Eagle3TrainStrategy.checkpoint_state_filter`); checkpoints dropped from ~2.xG to ~200MB (trainable MTP only).
- **Open**: worth adding a CI smoke test (random-weight draft -> save -> SGLang `load_weights` -> assert no missing/unexpected keys) per supported target, since the weight-layout contract is the single highest-risk failure mode. Note: the native Qwen3.5-4B checkpoint ships pretrained `mtp.*` weights, and the prototype now finetunes from them by default (`--no-init-from-native-mtp` to disable) rather than training from scratch.

## Current prototype

- Branch: https://github.com/curnane-lab/SpecForge_npu/tree/add_mtp_support
- Files: `specforge/modeling/draft/mtp.py` (Qwen3.5 draft, **flat** `mtp.layers.0.*` / `mtp.norm.weight` keys matching the native checkpoint + SGLang's flat ForCausalLM), `specforge/core/mtp.py` (`OnlineMTPModel`), `generate_mtp_data` on the 3 target backends (non-abstract optional hook), `scripts/train_mtp.py` (with native-mtp finetune init + embed/lm_head stripping), `scripts/merge_mtp_to_base.py` (flat layout, index-redirect merge), `configs/qwen3.5-4b-mtp.json`.
- Verified: training runs end-to-end on NPU; weight-key layout matches native Qwen3.5-4B checkpoint (`mtp.layers.0.*` flat); EAGLE3/DFlash pipelines unaffected (additive); VLM targets supported via recursive layer lookup. **Not yet verified**: SGLang inference round-trip (merge -> `load_weights` -> speculative decode) - this is the pending smoke test.

Looking for input primarily on **Q1–Q4** (coupling acceptance, scaling abstraction, registry alignment, loss design) before refining the PR. Tagging for visibility — happy to restructure into smaller tracking issues once direction is clear.
