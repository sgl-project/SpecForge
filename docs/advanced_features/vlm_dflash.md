# VLM DFlash Support — Status (add_vl_support)

This branch ports sgl-project/SpecForge PR #585 (commit `9323a510`, author zyk42)
onto the server-only unified runtime **and implements end-to-end multimodal
(image+text) DFlash training on top of it**. This document records what landed,
what was deliberately not ported, and the validation status.

## What this branch contains

### Foundation (ported from PR #585)

- **Draft model (VLM-capable)** — `specforge/modeling/draft/dflash.py`:
  partial rotation in `apply_rotary_pos_emb` (`rotary_dim < head_dim`, for
  Qwen3.5/Qwen3.6 `partial_rotary_factor=0.25`) and
  `Qwen3InterleavedMultiRotaryEmbedding` selected by
  `rope_scaling.mrope_interleaved`.
- **Draft config** — `configs/qwen3.5-4b-vl-dflash.json` (new, mirrors the
  Qwen3.5-4B target geometry: head_dim 256, mrope_section [11,11,10],
  partial_rotary_factor 0.25). The wider #585 config set (Qwen3-VL-8B/30B-A3B,
  Qwen3.5-9B/35B-A3B) is intentionally out of scope for this branch; it can be
  added verbatim in a follow-up once more targets are validated.
- **Weight-key resolution** — `resolve_target_weight_keys()` in
  `specforge/modeling/target/target_utils.py` auto-selects
  `model.language_model.embed_tokens.weight` for VLM targets;
  `populate_dflash_generated_config` reads language-model depth via
  `text_config`.

### Multimodal capture (new in this branch)

End-to-end data flow: **JSONL (+ image) → expanded ids/loss mask → capture
request (single-placeholder ids + base64 image) → patched SGLang server
expands, runs the ViT, captures aux hidden states + mRoPE positions →
Mooncake → collator → training forward with 3D position ids**.

- `model.input_modality: multimodal` (DFlash only): a `FeatureContract`
  (`{input_ids, loss_mask, hidden_states, position_ids}`) and a
  `ServerStreamingProvider` with a VLM `ServerInputAdapter`
  (`specforge/algorithms/common/vlm_input.py`).
- `specforge/data/vlm_preprocessing.py`: ShareGPT-style records with an
  optional `image` field (path or base64); the target's own chat template and
  HF processor produce the expanded `input_ids`/`loss_mask` (image region
  expanded in id space, mask zeros). One image per sample max (v1); text-only
  samples work in the same run.
- `ServerCaptureLayout.position_ids_feature` → the capture request's
  `features["position_ids"]`; the patched server writes the request's mRoPE
  positions `(1, L, 3) int64` into Mooncake (`_spec_capture_position_ids` in
  the scheduler sink; text requests get the arange broadcast fallback).
- `patches/sglang/v0.5.14/spec-capture.patch`: regenerated with the
  `position_ids` artifact (`SpecCaptureSink.put_sample(position_ids=...)`).
  Multimodal capture requests ride the stock `input_ids` + `image_data`
  `/generate` path with `SGLANG_MM_AVOID_RETOKENIZE=1` (set by the managed
  launcher for `input_modality=multimodal`), so the server re-expands
  placeholders in id space with zero retokenization drift — and the
  passthrough/seq-len checks fail loudly if client and server expansions ever
  disagree.
- Training: `OnlineDFlashModel.forward(..., position_ids=None)` gathers 3D
  mRoPE positions for context + anchor-offset draft slots
  (`(3, B, S + N·bs)`); `DFlashTrainStrategy` passes the collated
  `position_ids` tensor through. Text runs are byte-identical to before.
- Recipe: `examples/configs/qwen3.5-4b-vl-dflash-disaggregated.yaml`
  (single-node Ascend NPU managed stack).

## Not ported (by design)

- HF-backend VLM capture (`dflash_target_model.py`, `_build_vlm_reqs`,
  `mm_token_type_ids`) and the `train_dflash.py --is-vlm` plumbing from the
  pre-#678 script stack — superseded by server capture.
- `QwenVLOnlineDFlashModel` wiring — PR #585 referenced this class but never
  defined it; the unified runtime needs no separate VLM wrapper class.
- Two accidental reverts in the original #585 diff (domino projector code,
  D-PACE CLI args) were dropped during the cherry-pick.
- Offline (precomputed hidden states) multimodal capture: the offline path
  stays text-only for now.
- Online evaluation for multimodal runs.

## Validation status

- **Verified (CPU, this repo)**: registration parity and provider gates,
  request/payload construction, expansion math, collator, golden
  topology/recipe tests — `tests/test_algorithms/test_dflash_multimodal.py`
  plus the updated `test_config` suites (80 passed locally, 2 torch tests
  skipped pending GPU).
- **Verified statically**: the regenerated patch applies cleanly both ways to
  pristine sglang v0.5.14 (`git apply --check` / `--reverse --check`).
- **Not yet verified (needs GPU/NPU)**: a live multimodal capture run
  (Qwen3-VL / Qwen3.5 target + ViT + mRoPE positions) and an end-to-end
  training run. This is the next step; see the recipe above.

## Reference results from PR #585 (HF stack, author-validated)

- Qwen3-VL-30B-A3B-Thinking, 278K target-regenerated samples, 5-layer draft,
  block_size=8: accept length 3.52, +35.8% inference speedup (4x RTX 5090,
  TP=4, SGLang 0.5.12).
- Data must be target-model greedy-regenerated; system prompt must match
  between training and inference; <10K samples overfit severely.
