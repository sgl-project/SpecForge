# DSpark draft backbone recipe

Design rules for sizing a DSpark/DFlash draft against a new target,
distilled from the Inkling-Small v2 program (+17.6% mean accept length
over a v1 draft trained on identical data; see
`docs/inkling_small_dspark_v2_results.md`).

## Attention geometry: match the target

Use the target's head geometry for the draft: same `head_dim`, same
GQA ratio (e.g. 32 query heads / 8 KV heads at head_dim 128). The draft
consumes fused target hidden states through KV injection at every layer;
matching geometry lets those features project into attention without an
impedance mismatch, and it reuses the target's serving kernel shapes.

## Target taps: uniform coverage

Take aux hidden states from N layers spread uniformly over `[1, L-3]` of
the target (8 taps for a 42-layer target: `[1, 6, 12, 17, 23, 28, 34, 39]`).
Avoid clustering taps near the output — early- and mid-stack features
carry the context signal the draft cannot recompute. Keep the tap list in
the draft config (`dflash_config.target_layer_ids`) as the single source
of truth for training, probing, and serving.

## Depth vs width

A 6-layer draft with a wide FFN (12288 at hidden 4096, ~3x) outperformed
a 5-layer narrower one at comparable parameter count. With KV injection
delivering target features to every layer, marginal depth buys less than
FFN capacity for absorbing them.

## Attention pattern: serve-ability first

All draft layers full attention unless the serving kernel supports SWA
for the draft path. A hybrid-SWA draft that cannot serve is a science
project; check the serving kernel BEFORE choosing the pattern.

## Block size: train = serve

Train at the block size you will serve (e.g. 7). Training at a larger
block and serving smaller leaves accept length on the table and makes
the confidence head miscalibrated for the deployed horizon. Adapt the
within-block loss decay to the block size (γ=4 at block 7 — steeper
decay than γ≈9 at block 15, since position-1 correctness dominates the
expected accept length of a short block).

## Objective and schedule

- 0.1 CE + 0.9 L1 feature distillation + 1.0 confidence BCE, ~512
  sampled anchors per sequence.
- Constant LR (5e-4) with a short warmup (4%) then hold beats cosine for
  drafters: the objective is stationary distillation, and decay mostly
  slows late-run absorption of hard examples. Keep global batch fixed
  (e.g. 512) across stages.

## Checklist for a new target

1. Read the target's config: layer count, head geometry, addressable
   context, quantization.
2. Pick taps uniformly; write them into the draft config.
3. Clone the target's attention geometry; size FFN ~3x hidden; 6 layers.
4. Confirm the serving kernel supports the attention pattern and block
   size; set `block_size` = serve block.
5. Probe capture parity before the first smoke (per-layer scale + stream
   distinctness), then smoke the worst-case batch shape for memory.
