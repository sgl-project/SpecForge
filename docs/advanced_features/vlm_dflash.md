# VLM DFlash Support - Status (add_vl_support)

This branch ports sgl-project/SpecForge PR #585 (commit `9323a510`, author zyk42)
onto the server-only unified runtime **and implements end-to-end multimodal
(image+text) DFlash training on top of it**. This document records what landed,
what was deliberately not ported, and the validation status.

> **Rope decision**: multimodal DFlash drafts use the plain 1D rope convention -
> the same as the text-only path - on purpose. Visual information reaches the
> draft exclusively through the captured target hidden states, never through the
> draft's own positional embedding, so the draft has no use for the target's
> (3, L) mRoPE positions. Staying on one position convention keeps training and
> serving byte-identical to the text pipeline and reuses the stock
> `configs/qwen3.5-4b-dflash.json` draft geometry.

## What this branch contains

### Foundation (ported from PR #585)

- **Draft model (VLM-capable)** - `specforge/modeling/draft/dflash.py`:
  partial rotation in `apply_rotary_pos_emb` (`rotary_dim < head_dim`, for
  Qwen3.5/Qwen3.6 `partial_rotary_factor=0.25`). The draft always uses the
  stock `Qwen3RotaryEmbedding`.
- **Weight-key resolution** - `resolve_target_weight_keys()` in
  `specforge/modeling/target/target_utils.py` auto-selects
  `model.language_model.embed_tokens.weight` for VLM targets;
  `populate_dflash_generated_config` reads language-model depth via
  `text_config`.

### Multimodal capture (new in this branch)

End-to-end data flow: **JSONL (+ image) -> expanded ids/loss mask -> capture
request (single-placeholder ids + base64 image) -> patched SGLang server
expands, runs the ViT, captures aux hidden states -> Mooncake -> collator ->
training forward on plain 1D positions**.

- `model.input_modality: multimodal` (DFlash only): a `FeatureContract`
  (`{input_ids, loss_mask, hidden_states}`) and a
  `ServerStreamingProvider` with a VLM `ServerInputAdapter`
  (`specforge/algorithms/common/vlm_input.py`). Multimodal capture stores the
  same three tensors as text capture - no `position_ids` artifact is requested
  or consumed.
- `specforge/data/vlm_preprocessing.py`: ShareGPT-style records with an
  optional `image` field (path or base64); the target's own chat template and
  HF processor produce the expanded `input_ids`/`loss_mask` (image region
  expanded in id space, mask zeros). One image per sample max (v1); text-only
  samples work in the same run.
- `patches/sglang/v0.5.14/spec-capture.patch`: tracks upstream's rewritten
  async streaming sink. Multimodal capture requests ride the stock
  `input_ids` + `image_data` `/generate` path with
  `SGLANG_MM_AVOID_RETOKENIZE=1` (set by the managed launcher for
  `input_modality=multimodal`), so the server re-expands placeholders in id
  space with zero retokenization drift - and the passthrough/seq-len checks
  fail loudly if client and server expansions ever disagree.
- Training: `OnlineDFlashModel._forward_draft_blocks` builds positions with
  the unconditional text-path 1D `arange` convention; multimodal batches flow
  through the identical forward as text batches. Text runs are
  byte-identical to before.
- Recipe: `examples/configs/online/disaggregated/external/qwen3.5-4b-vl-dflash-disaggregated.yaml`
  (single-node Ascend NPU managed stack; draft config
  `configs/qwen3.5-4b-dflash.json`).

## Not ported (by design)

- HF-backend VLM capture (`dflash_target_model.py`, `_build_vlm_reqs`,
  `mm_token_type_ids`) and the `train_dflash.py --is-vlm` plumbing from the
  pre-#678 script stack - superseded by server capture.
- `QwenVLOnlineDFlashModel` wiring - PR #585 referenced this class but never
  defined it; the unified runtime needs no separate VLM wrapper class.
- mRoPE draft support (`Qwen3InterleavedMultiRotaryEmbedding`, the
  `rope_scaling.mrope_interleaved` switch, and the server `position_ids`
  capture artifact): the draft consumes visual information only through the
  captured target hidden states, so the (3, L) target positions carry no
  signal for it. Retired in favor of the single plain-rope convention above.
- Two accidental reverts in the original #585 diff (domino projector code,
  D-PACE CLI args) were dropped during the cherry-pick.
- Offline (precomputed hidden states) multimodal capture: the offline path
  stays text-only for now.
- Online evaluation for multimodal runs.

## Validation status

- **Verified (CPU, this repo)**: registration parity and provider gates,
  request/payload construction, expansion math, collator, golden
  topology/recipe tests - `tests/test_algorithms/test_dflash_multimodal.py`
  plus the updated `test_config` suites.
- **Verified statically**: the regenerated patch applies cleanly both ways to
  pristine sglang v0.5.14 (`git apply --check` / `--reverse --check`).
- **Not yet verified (needs GPU/NPU)**: a live multimodal capture run
  (Qwen3-VL / Qwen3.5 target + ViT) and an end-to-end training run. This is
  the next step; see the recipe above.

## Reference results from PR #585 (HF stack, author-validated)

- Qwen3-VL-30B-A3B-Thinking, 278K target-regenerated samples, 5-layer draft,
  block_size=8: accept length 3.52, +35.8% inference speedup (4x RTX 5090,
  TP=4, SGLang 0.5.12).
- Data must be target-model greedy-regenerated; system prompt must match
  between training and inference; <10K samples overfit severely.
