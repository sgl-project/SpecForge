# Offline DSpark with Target KV Conditioning

This path trains a DSpark draft from Qwen3.5 text-attention K/V caches instead
of concatenated intermediate hidden states. It supports offline features and
GQA/MHA draft attention. The existing `dspark` hidden-state recipe is unchanged.
Streaming capture, MLA, multimodal positions, and speculative serving are not
implemented by this recipe. Stock SGLang does not automatically support this
new conditioning interface; deployment requires a compatible KV-aware consumer.
The legacy `spec_generate` helper rejects KV drafts rather than capturing the
wrong features.

## Positional Re-encoding

For each context token at absolute position `p`, the target cache contains
post-QK-normalization, post-RoPE keys. Each draft layer applies:

```text
K0       = inverse_target_partial_RoPE(K_cached, p)
K_draft  = draft_RoPE(optional_draft_KNorm(W_K concat_selected_layers(K0)), p)
V_draft  = W_V concat_selected_layers(V_cached)
```

`target_kv_position_mode: derope_reproject` selects this transformation;
`target_kv_context_key_norm: true` reuses the layer's existing draft K RMSNorm.
No extra norm parameters are introduced. Frequencies for the inverse are
computed in FP32, including under BF16 autocast/FSDP, to avoid rounded frequency
buffers. Values and the non-rotary target-key channels are preserved exactly by
the inverse transform. Target key normalization and cache quantization error
are not inverted. This is not hidden-state reconstruction.

The `cached` mode projects cached K/V directly without context re-rotation or
normalization, providing a matched parameter-count control. Position mode and
normalization are explicit resume-contract fields; changing them requires a
fresh training run. No acceptance-rate or length-extrapolation guarantee is
claimed by this implementation.

## Prepare Features

Use `configs/qwen3.5-4b-dspark-kv.json` with the public `Qwen/Qwen3.5-4B` model.
Prepare JSONL rows containing an entire tokenized conversation in `input_ids`
and an equally long `loss_mask` (1 for supervised assistant tokens, 0 otherwise).
Tokenize with the target tokenizer and intended chat/thinking template before
capture; this script does not generate responses or choose a temperature.

```bash
python examples/prepare_dspark_kv.py \
  --target-model Qwen/Qwen3.5-4B \
  --draft-config configs/qwen3.5-4b-dspark-kv.json \
  --input-jsonl ./data/tokenized-train.jsonl \
  --output-dir ./cache/dspark-kv/train \
  --max-length 8192
```

Use `--revision` to pin the target revision. Capture is one unpadded sequence at
a time, with a fresh cache and zero-based positions. Only full-attention layer
IDs are accepted; Qwen3.5 linear-attention states are not K/V caches. The example
selects layers 3, 11, 19, 27, and 31. The cache uses partial NeoX rotation even
though the target's text MRoPE configuration interleaves axis frequencies:
all three position axes are identical for these text-only inputs.

Each `.ckpt` contains `input_ids [S]`, `loss_mask [S]`,
`target_kv [S,L,2,H,D]` (K then V), and final normalized
`target_last_hidden_states [S,hidden_size]`. Final hidden states supervise the
DSpark losses only; they never enter the draft backbone. The capture manifest
records layer order, geometry, model revision, and input/config hashes. Training
validates the manifest's layer order, cache state, rotary geometry, and target
identity before assembly; changing only the draft position mode or key norm
can reuse the same capture. The manifest is metadata, not a content checksum
for every feature tensor or for local model weights. Keep
the same target weights, layer order, and draft configuration for capture and
training; do not mix directories from different captures. Full-sequence KV
storage is large, so budget disk capacity before capturing a full dataset.

## Train

```bash
specforge train --config examples/configs/offline/colocated/qwen3.5-4b-dspark-kv-offline.yaml
```

The sample recipe uses three passes over the supplied features. Adjust the
dataset path, trainer topology, batch size, and total steps for your data.
The normalizer/collator pad only the sequence axis; the existing DSpark packed
prefix mask prevents reading target features at or after each anchor. CE, L1,
confidence, Markov-head behavior, and label alignment use the existing DSpark
implementation. The target is frozen and only the draft is optimized.
