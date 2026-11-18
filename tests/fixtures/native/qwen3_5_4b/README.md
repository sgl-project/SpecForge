# Qwen3.5-4B native metadata

Source: `Qwen/Qwen3.5-4B`, revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

`config.json` is the original configuration. `headers.json` contains the original
safetensors header entries for all 15 `mtp.*` tensors and the target embedding,
retrieved with HTTP Range requests from the two shards. No tensor data is stored.

Tests use this fixed metadata independently of SpecForge's converter and preset.
