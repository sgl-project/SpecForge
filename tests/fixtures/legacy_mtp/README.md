# Historical MTP regression fixture

Captured from the CPU FP32 eager `Qwen3_5MTPDraftModel` at commit `1c0e018a`,
before deleting that implementation. Seed: 731. `config.json` specifies the tiny
model; `model.safetensors` contains its actual historical state dict.

`reference.safetensors` contains two sequences of four token IDs, target features,
nonzero position IDs, forward hidden states/logits, and gradients of
`logits.square().mean()` with the embedding frozen. It detects changes in gated
attention, offset RMSNorm, partial RoPE, fusion, weight mapping and backward.
The historical effective RoPE theta is 10000 despite the JSON's 10000000;
the codec must preserve this when importing a trained old checkpoint.

These files contain synthetic tensors only. Native target initialization uses
the new preset and its explicit target-correct RoPE, not this historical codec.

`original_config.json` preserves the retired full-size JSON recipe for codec tests.
