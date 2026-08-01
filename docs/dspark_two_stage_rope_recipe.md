# Two-stage RoPE recipe: native-context training, then YaRN long-context adaptation

How to train a DSpark/DFlash drafter that holds its accept length out to
the target's full addressable context without paying long-sequence cost
for the whole run.

## The recipe

**Stage 1 — native RoPE, short sequences.** Train the drafter at the
target's original context (e.g. `rope_scaling: null`,
`max_position_embeddings: 8192`, `max_length: 8192`) for the bulk of the
token budget. Short sequences keep iteration fast and memory low, so this
stage carries all the epochs.

**Stage 2 — YaRN, long sequences, weights-only warm start.** Switch the
draft config to YaRN (e.g. `factor: 128`,
`original_max_position_embeddings: 8192`, no mscale keys →
`max_position_embeddings: 1048576`) and raise `max_length` (e.g. 65536).
Warm start from the stage-1 final weights ONLY — RoPE is parameter-free,
so the weights are bit-compatible across the config change; drop the
optimizer/scheduler state entirely. One epoch over the stage-1 corpus
plus upsampled genuinely-long data (agentic trajectories) suffices.

Match the YaRN factor and `max_position_embeddings` to the TARGET's
addressable range, not to the longest training sequence — serving will
position-embed wherever the target does.

## Why weights-only

Restoring Adam moments across a sequence-length and data-distribution
change transplants stale curvature estimates; a short LR re-warm (~64
optimizer steps) from fresh moments is cheaper and stabler. It also makes
the warm start world-size-independent (no optimizer resharding concerns).

## Checkpoint schema pitfall

transformers 5.x `save_pretrained` writes only the new `rope_parameters`
schema and drops legacy `rope_scaling`. Serving stacks and older
transformers that read only the legacy key silently lose YaRN — the draft
falls back to unscaled RoPE at serve time and accept length collapses
beyond the original context. Ship checkpoints with BOTH schemas present.

## Validation

Evaluate accept length bucketed by prompt length on held-out long data
(full conversation prefixes, not synthetic fill). A successful adaptation
is FLAT across buckets. Reference run (Inkling-Small v2, block 7,
temperature 0): 3.59 at 8–16K, 3.56 at 16–32K, 3.53 at ≥32K — within
noise of each other; see `docs/inkling_small_dspark_v2_results.md`.
