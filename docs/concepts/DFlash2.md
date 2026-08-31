# DFlash2

DFlash2 extends the one-pass DFlash draft with grouped dynamic convolutions and
a low-rank selector that chooses a coherent path through the target head's
top-k candidates. In SpecForge it is a **draft architecture**, not a separate
training strategy or feature protocol.

| Boundary | DFlash2 value |
| --- | --- |
| Training strategy | `training.strategy: dflash` |
| Draft architecture | `DFlash2DraftModel` |
| Online server capture | `--spec-capture-method dflash` |
| Offline feature schema | DFlash `input_ids`, `loss_mask`, `hidden_states` |
| Export target | `specforge export --to hf` |
| SGLang serving algorithm | `DFLASH` |

This reuse is intentional. The convolution and selector run entirely in the
draft, so the target server captures the same layers and the trainer consumes
the same tensors as standard DFlash.

## Architecture

Every attention and MLP sublayer is wrapped by a grouped, dynamic depthwise
convolution. `conv_kernel_size` controls its number of causal taps and
`conv_group_size` controls how channels share dynamic kernel coefficients. The
base kernel is an identity and the kernel projection is zero-initialized, so a
new convolution starts as an exact no-op.

The candidate selector first takes the target LM head's unary top-k at every
proposal position. It then adds a low-rank transition score conditioned on the
draft hidden state and the predecessor token. The successor codebook is
zero-initialized, so a new selector initially preserves unary DFlash scores.

A DFlash2 draft config must explicitly select the architecture and provide its
four additional model fields:

```json
{
  "architectures": ["DFlash2DraftModel"],
  "dflash_config": {
    "block_size": 8,
    "target_layer_ids": [1, 16, 31, 46, 61],
    "mask_token_id": 248070,
    "conv_kernel_size": 2,
    "conv_group_size": 16,
    "selector_rank": 256,
    "selector_top_k": 16
  }
}
```

`conv_kernel_size` must be positive and no larger than `block_size`;
`conv_group_size` must divide `hidden_size`; `selector_rank` must be positive;
and `selector_top_k` must be between 1 and `vocab_size`. The checked-in complete
configuration is
[`configs/qwen3.6-27b-dflash2.json`](../../configs/qwen3.6-27b-dflash2.json).
Unlike a standard DFlash draft, DFlash2 cannot be selected through
target-derived defaults: provide this config or a pretrained DFlash2 directory
whose `config.json` declares `DFlash2DraftModel`.

## Selector objective and schedule

The base head keeps the configured DFlash, D-PACE, CE/LK, and TV behavior. The
selector has a separate categorical cross-entropy objective over the strict
unary top-k candidate set. Tokens whose gold target is outside that set do not
contribute selector CE; `selector_coverage` measures how often it is present.

Configure its weight and optimizer-step schedule in the training YAML:

```yaml
training:
  strategy: dflash
  dflash2_selector_loss_alpha: 1.0
  dflash2_selector_warmup_ratio: 0.0005
  dflash2_selector_ramp_ratio: 0.0005
  dflash2_selector_stop_gradient: false
```

The effective selector weight is zero during warmup, increases linearly during
the ramp, and then stays at `dflash2_selector_loss_alpha`. Warmup and ramp are
fractions of the resolved optimizer-step horizon, not micro-batch counts. With
a positive configured alpha the selector remains in the optimizer and autograd
graph during warmup, which keeps DDP behavior stable. Setting the configured
alpha to exactly zero statically disables selector training, freezes its
parameters, and excludes them from the optimizer.

`dflash2_selector_stop_gradient` controls whether the selector CE also trains
the parameters it reads. It defaults to `false`, which keeps the coupled
behavior: selector CE flows back into the unary logits and draft hidden states.
Setting it to `true` detaches both inputs inside the selector term, so only the
selector's own parameters receive that gradient while the primary
DFlash/D-PACE/LK objective keeps its unchanged gradient path.

The combined base and selector numerator is normalized with the DFlash valid
effective-token denominator. Selector metrics retain their own denominators so
coverage, accuracy, probability, and CE remain interpretable when only part of
the unary candidate set covers the target.

## Train online or offline

The managed-local Qwen3.6-27B example owns one target capture process and one
trainer process. Update its model/data paths and GPU allocation, then run:

```bash
specforge train \
  -c examples/configs/online/disaggregated/managed-local/qwen3.6-27b-dflash2-disaggregated.yaml
```

For external online services, use the same YAML model and training fields but
omit `deployment.disaggregated.managed_local`, provide the Mooncake endpoints
and `server_urls`, and start SGLang with `--spec-capture-method dflash`. Its
auxiliary layer IDs must exactly match `dflash_config.target_layer_ids`.

Offline DFlash2 uses the normal DFlash feature preparation path:

```bash
torchrun --nproc_per_node=8 \
  scripts/prepare_hidden_states.py \
  --strategy dflash \
  --target-model-path /path/to/Qwen3.6-27B \
  --draft-model-config configs/qwen3.6-27b-dflash2.json \
  --data-path /path/to/train.jsonl \
  --output-path ./cache/hidden_states/qwen3.6-27b-dflash2 \
  --chat-template qwen3.5 \
  --max-length 8192
```

Point an offline training YAML at that directory with
`data.hidden_states_path`; keep `training.strategy: dflash` and the same draft
config. DFlash2 supports the DFlash `eager`, `sdpa`, and `flex_attention`
training backends and the same full/sliding per-layer layouts.

## Export and serve

Export the runtime checkpoint as a Hugging Face draft directory:

```bash
specforge export --to hf \
  --checkpoint ./outputs/qwen3.6-27b-dflash2-disaggregated-with-selector/qwen3.6-27b-dflash2-disaggregated-with-selector-latest \
  --draft-config configs/qwen3.6-27b-dflash2.json \
  --output-dir ./exports/qwen3.6-27b-dflash2
```

Use an SGLang release or revision that contains DFlash2 support
([SGLang PR #35371](https://github.com/sgl-project/sglang/pull/35371)). The
serving algorithm remains `DFLASH`; `DFlash2DraftModel` in the exported config
selects the convolution and path-selector implementation. Optional
`dflash_config.output_multiplier` and `final_logit_softcapping` are applied to
unary logits in both SpecForge's local generation path and the matching serving
contract.

Training supports the shared DFlash-family `gqa`, `mha`, and `mla` attention
modes. SGLang DFlash-family serving currently supports GQA/MHA layouts, so do
not use an MLA DFlash2 export for a serving benchmark until the corresponding
loader and kernels are available.
