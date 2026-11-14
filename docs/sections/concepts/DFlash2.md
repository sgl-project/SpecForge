# DFlash2

DFlash2 extends the one-pass DFlash draft with grouped dynamic convolutions and
a low-rank selector that chooses a coherent path through the target head's
top-k candidates. In SpecForge it is a **draft architecture**, not a separate
feature protocol. It has its own default objective for selector supervision.

| Boundary | DFlash2 value |
| --- | --- |
| Default objective | `dflash2` |
| Draft architecture | `DFlash2DraftModel` |
| Online server capture | `--speculative-algorithm DFLASH` |
| Offline feature schema | DFlash inputs; final target states when teacher metrics are enabled |
| Export target | `specforge export --format sglang` |
| SGLang serving algorithm | `DFLASH` |

This reuse is intentional. The convolution and selector run entirely in the
draft, so the target server captures the same layers and the trainer consumes
the same tensors as standard DFlash.

## Architecture

Every attention and MLP sublayer is wrapped by a grouped, dynamic depthwise
convolution. `kernel_size` controls its number of causal taps and
`group_size` controls how channels share dynamic kernel coefficients. The
base kernel is an identity and the kernel projection is zero-initialized, so a
new convolution starts as an exact no-op.

The candidate selector first takes the target LM head's unary top-k at every
proposal position. It then adds a low-rank transition score conditioned on the
draft hidden state and the predecessor token. The successor codebook is
zero-initialized, so a new selector initially preserves unary DFlash scores.

The draft YAML names convolution nodes in `conv_configs` and references one
from each layer's `sublayer_conv`. The selector settings belong to `spec_config`.
For example, these fields are part of a complete canonical draft:

```yaml
architectures: [DFlash2DraftModel]
conv_configs:
  local:
    type: grouped_causal
    kernel_size: 2
    group_size: 16
layer_configs:
  decoder:
    attention: full
    mlp: dense
    sublayer_conv: local
spec_config:
  # target_features and protocol are also required.
  selector_rank: 256
  selector_top_k: 16
```

`kernel_size` must be positive and no larger than the protocol block size;
`group_size` must divide `hidden_size`; `selector_rank` must be positive;
`selector_top_k` must be between 1 and `vocab_size`. Use the complete
[Qwen3.6 preset](https://github.com/sgl-project/SpecForge/blob/main/specforge/config/draft/presets/qwen3.6-27b-dflash2.yaml)
as a starting point. A saved canonical checkpoint already contains its structure.

## Selector objective and schedule

The base head keeps the configured DFlash, D-PACE, CE/LK, and TV behavior. The
selector has a separate categorical cross-entropy objective over the strict
unary top-k candidate set. Tokens whose gold target is outside that set do not
contribute selector CE; `dflash/hard_label/unary_topK_recall` measures how often it is present.
See the [training metrics guide](../basic_usage/training.md#reading-the-acceptance-metrics)
for prefix survival, candidate misses, selector errors, and dashboard migration.

Configure its weight and optimizer-step schedule in the training YAML:

```yaml
training:
  objective_options:
    selector_loss_alpha: 1.0
    selector_warmup_ratio: 0.0005
    selector_ramp_ratio: 0.0005
    selector_stop_gradient: false
```

The effective selector weight is zero during warmup, increases linearly during
the ramp, and then stays at `selector_loss_alpha`. Warmup and ramp are
fractions of the resolved optimizer-step horizon, not micro-batch counts. With
a positive configured alpha the selector remains in the optimizer and autograd
graph during warmup, which keeps DDP behavior stable. Setting the configured
alpha to exactly zero statically disables selector training, freezes its
parameters, and excludes them from the optimizer.

`selector_stop_gradient` controls whether the selector CE also trains
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
  -c examples/configs/online/disaggregated/managed-local/qwen3.6-27b-dflash2.yaml
```

For external online services, use the same YAML model and training fields but
omit `deployment.disaggregated.managed_local`, provide the Mooncake endpoints
and `server_urls`, and start SGLang with `--speculative-algorithm DFLASH`. Its
auxiliary layer IDs must exactly match `spec_config.target_features.target_layer_ids`.

Offline DFlash2 uses the normal DFlash feature preparation path. First start
the matching plugin server and Mooncake using the
[offline capture guide](../../../examples/offline/README.md):

```bash
python scripts/prepare_hidden_states.py \
  --server-urls http://127.0.0.1:30000 \
  --target-model-path /path/to/Qwen3.6-27B \
  --draft specforge/config/draft/presets/qwen3.6-27b-dflash2.yaml \
  --data-path /path/to/train.jsonl \
  --output-path ./cache/hidden_states/qwen3.6-27b-dflash2 \
  --chat-template qwen3.5 \
  --max-length 8192
```

Point an offline training YAML at that directory with
`data.hidden_states_path` and use the same draft config. DFlash2 supports the DFlash `eager`, `sdpa`, and `flex_attention`
training backends and the same full/sliding per-layer layouts.

## Export and serve

Export the runtime checkpoint as an SGLang serving directory:

```bash
specforge export --format sglang \
  --checkpoint ./outputs/qwen3.6-27b-dflash2-disaggregated-with-selector/qwen3.6-27b-dflash2-disaggregated-with-selector-latest \
  --output-dir ./exports/qwen3.6-27b-dflash2
```

See [export restrictions](../../draft_export.md); mixed per-layer causality
is rejected until the pinned SGLang format can express it.

Use an SGLang release or revision that contains DFlash2 support
([SGLang PR #35371](https://github.com/sgl-project/sglang/pull/35371)). The
serving algorithm remains `DFLASH`; `DFlash2DraftModel` in the exported config
selects the convolution and path-selector implementation. The resolved target logit transform is applied to
unary logits in both SpecForge's local generation path and the matching serving
contract.

Training supports the shared DFlash-family `gqa`, `mha`, and `mla` attention
modes. Export checks whether the pinned SGLang format can express the selected
components; unsupported settings fail before writing an artifact.
