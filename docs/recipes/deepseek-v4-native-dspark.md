# Native DeepSeek V4 DSpark

`DeepseekV4DSparkDraftModel` trains the native three-layer Flash/Pro draft with
SpecForge's DSpark objective. It preserves the `mtp.*` weight layout, MQA sinks,
mHC, MoE, and Markov/confidence heads.

- Flash: `configs/deepseek-v4-flash-0731-dspark.json`, capture layers 40/41/42.
- Pro: `configs/deepseek-v4-pro-0813-dspark.json`, capture layers 58/59/60.

The draft block size is configurable through `block_size` in the model JSON;
these examples default to 10.

## Import

Import from a complete official checkpoint, including its safetensors index:

```bash
python scripts/warmstart_deepseek_v4_dspark.py \
  --target-path /path/to/DeepSeek-V4-Flash-0731 \
  --draft-config configs/deepseek-v4-flash-0731-dspark.json \
  --output ./models/deepseek-v4-flash-0731-dspark
```

Add `--dry-run` to check all keys and shapes without writing weights. The importer
dequantizes FP4/FP8 weights; missing tensors fail rather than initialize randomly.
For Pro, use its matching target checkpoint and model JSON, and write to
`./models/deepseek-v4-pro-0813-dspark`.

## Train

Use the [Flash recipe](https://github.com/sgl-project/SpecForge/blob/main/examples/configs/offline/colocated/deepseek-v4-flash-0731-dspark-offline.yaml)
or [Pro recipe](https://github.com/sgl-project/SpecForge/blob/main/examples/configs/offline/colocated/deepseek-v4-pro-0813-dspark-offline.yaml)
with the existing [training entry](../sections/basic_usage/training.md).
Set `model.target_model_path` to the complete target checkpoint and
`data.hidden_states_path` to captured features. The existing target loader reads
only `embed.weight` and `head.weight`; no separate table export is needed.
Configure output paths and trainer ranks for your environment. General parameter
definitions are in the [configuration reference](https://github.com/sgl-project/SpecForge/blob/main/examples/configs/README.md).

```bash
specforge train \
  -c examples/configs/offline/colocated/deepseek-v4-flash-0731-dspark-offline.yaml
```

The native recipes default to warmstart from the imported official draft weights.
To initialize randomly, skip the import and remove `model.draft_checkpoint_path`
from the YAML (or set it to YAML `null`); leave `training.resume_from` unset.
The teacher embedding and LM head still use frozen pretrained weights.

Each offline `.ckpt` contains `input_ids` and `loss_mask` shaped `[seq]`,
`hidden_states` shaped `[seq, 3 * hidden_size]` from the native mHC-mean capture
layers, and post-normalization `target_last_hidden_states` shaped
`[seq, hidden_size]`. For online training, use
the [Flash recipe](https://github.com/sgl-project/SpecForge/blob/main/examples/configs/online/disaggregated/external/deepseek-v4-flash-0731-dspark-disaggregated.yaml)
or [Pro recipe](https://github.com/sgl-project/SpecForge/blob/main/examples/configs/online/disaggregated/external/deepseek-v4-pro-0813-dspark-disaggregated.yaml)
with pre-tokenized regenerate JSONL containing `input_ids` and `loss_mask`.
The online recipes derive the optimizer-step budget from accepted prompts and
the epoch count.
Use the existing [capture setup](deepseek-v4-flash-dspark-disaggregated.md), with
the matching target checkpoint, native capture layers listed above, and
`--spec-capture-method dspark`. Allow context-length headroom above
`data.max_length` and update the example's endpoints for your hosts.
Capture, teacher embedding/head, tokenizer and warmstart must come from the same
checkpoint version.

```bash
specforge train \
  -c examples/configs/online/disaggregated/external/deepseek-v4-flash-0731-dspark-disaggregated.yaml
```

These recipes use the standard SpecForge launcher, including YAML or JSON run
configs, `section.field=value` overrides, and separate online `--role producer`
/ `--role consumer` launches. Existing `torchrun` workers are also supported;
online workers require `--role consumer`.

Full resume uses `training.resume_from`; remove `model.draft_checkpoint_path`
from the run config and preserve the model/feature paths. These are large MoE
drafts: provision memory for dequantized weights, optimizer states and activations.
Gradient checkpointing is disabled by default. Set `use_gradient_checkpointing`
to `true` in the model JSON to trade recomputation time for lower activation memory.
Training uses floating-point weights/activations; quantized KV is not simulated.

## Export

```bash
specforge export --to sglang --checkpoint /path/to/checkpoint \
  --draft-config configs/deepseek-v4-flash-0731-dspark.json \
  --output-dir ./outputs/native-draft
```

SGLang export writes native BF16 weights with FP32 routing buffers and requires
a serving build supporting `DeepseekV4ForCausalLMDSpark` with floating-point
experts. Use `--to hf` for a SpecForge-reloadable artifact.
