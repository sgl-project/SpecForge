# Training recipes

All bundled recipes use canonical draft configuration. The top-level `draft`
accepts an inline mapping or a path to a YAML/JSON file. Files in
`specforge/config/draft/presets/` are ordinary configurations; their names do
not select model code.

```yaml
draft: specforge/config/draft/presets/qwen3-8b-dflash.yaml
model:
  target_model_path: Qwen/Qwen3-8B
training:
  objective_options:
    num_anchors: 512
  loss:
    fused: true
    terms:
      token:
        kind: ce
    scheduling:
      chunk_size: 128
      checkpoint: true
```

The architecture selects its default objective. Set `training.objective` to
choose another registered objective compatible with that architecture.
Algorithm sampling/rollout options belong in `objective_options`; numerical
losses, coefficients and execution settings belong in `loss`.

Use `initialization.checkpoint` for weight import and `resume` for a canonical
training checkpoint. These sources already record the draft configuration.
Historical checkpoint weights remain importable through the legacy codec;
their optimizer states are not portable across model layout changes.

Offline EAGLE supports `attention_backend: usp` with Ulysses/Ring sequence
parallelism. Online consumers use data parallelism; configure target TP on
the capture server.
