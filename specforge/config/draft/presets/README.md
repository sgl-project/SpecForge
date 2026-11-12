# Draft presets

These are ordinary draft YAML files. Copy a file and edit its contents, or pass
its path as `draft` in a training recipe. Filenames do not select implementations.
`native/` contains structures matching a target's built-in draft; it does not
load target model classes or derive dimensions from the target configuration.

## Qwen3.5-4B native MTP

The bundled preset matches the dense [Qwen3.5-4B configuration](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json), including offset RMSNorm, gated attention and interleaved partial RoPE.
Add the following to a recipe with `model.target_model_path: Qwen/Qwen3.5-4B`
and `training.strategy: mtp`, with `training.attention_backend: eager`.
MTP planning at this stage rejects the default `flex_attention`:

```yaml
draft: specforge/config/draft/presets/native/qwen3_5_4b_mtp.yaml
initialization:
  from_target:
    "":
      from: mtp
      rename:
        layers: backbone.layers
        norm: backbone.final_norm
        fc: fusion.fc
        pre_fc_norm_embedding: fusion.pre_fc_norm_embedding
        pre_fc_norm_hidden: fusion.pre_fc_norm_hidden
```

The root mapping requires every owned draft tensor and rejects unexplained
`mtp.*` tensors. Bound embedding/head tensors are excluded from this copy and
loaded from the target's token tables. Copies require unquantized safetensors;
use the source dtype or exact widening (for example BF16 to FP32). Narrowing
fails before reading tensors. Copied weights remain trainable and are saved;
optional `training.frozen` prefixes freeze them independently of initialization.

For continuation use `initialization.checkpoint`; for full training resume use
`resume`. Neither takes `from_target`. To merge trained weights back into a
separate target directory:

```bash
specforge export --export-mode native --checkpoint CHECKPOINT_PATH \
  --target-model TARGET_PATH --output-dir OUTPUT_DIR
```

A complete native `mtp` root copy checks target computation fields, including
RoPE, during training preparation and initialization. Individual tensor copies
do not impose this native structure contract. Export repeats the same checks,
replaces old MTP tensors, and leaves the source intact.

Current coverage is dense Qwen3.5 MTP and text training inputs. MoE native presets
are not advertised. Tests validate full preset shapes on meta, then use smaller
hidden/vocabulary widths with the same head geometry, gates and RoPE for exact
copy/export/copy checks and numerical comparison with pinned SGLang. Independent
real configuration and weight-header fixtures check the mapping; a CPU comparison
with the HF decoder uses distinct T/H/W positions to check multi-axis RoPE.
