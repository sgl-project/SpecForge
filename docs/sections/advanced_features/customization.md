# Customize a training run

Customization starts from a typed YAML config. Pick the closest checked-in
file under
[`examples/configs`](https://github.com/sgl-project/SpecForge/tree/main/examples/configs),
change model and data paths, and keep the same training entry point:

```bash
specforge train --config ./my-run.yaml
```

For one-off changes, use dotted overrides rather than adding another launcher:

```bash
specforge train \
  --config examples/configs/online/disaggregated/external/qwen3-8b-eagle3.yaml \
  model.target_model_path=/models/my-target \
  data.train_data_path=/datasets/my-training-data.jsonl \
  training.learning_rate=5e-5
```

The config is strict: unknown fields, invalid strategy/topology combinations,
and unsupported attention backends are errors. See
[`specforge/config/schema.py`](https://github.com/sgl-project/SpecForge/blob/main/specforge/config/schema.py) and the
[training guide](../basic_usage/training.md) for the accepted fields.

## Chat templates

Register a text chat template in `TEMPLATE_REGISTRY` in
`specforge/data/template.py`, then reference its name with
`data.chat_template`:

```python
TEMPLATE_REGISTRY.register(
    name="your-template-name",
    template=ChatTemplate(
        assistant_header="...",
        user_header="...",
        system_prompt="...",
        end_of_turn_token="...",
    ),
)
```

```yaml
data:
  train_data_path: /datasets/my-training-data.jsonl
  chat_template: your-template-name
  max_length: 4096
```

The server capture runtime accepts text input only. VLM training, including
`qwen2_5_vl`, is not supported. Attention backends are a closed,
strategy-specific set:

| Strategy | Accepted `training.attention_backend` values |
| --- | --- |
| EAGLE3 | `sdpa`, `flex_attention`, `fa`, offline `usp` |
| P-EAGLE | `flex_attention` |
| DFlash, DFlash2, Domino, DSpark | `eager`, `sdpa`, `flex_attention` |

## Target models

For a Hugging Face-compatible target, normally only these model fields need to
change:

```yaml
model:
  target_model_path: organization/model-name
  target_backend: sglang
  trust_remote_code: false
  embedding_key: model.embed_tokens.weight
  lm_head_key: lm_head.weight
```

Every online run uses `model.target_backend: sglang`. Add target-model support
to the SGLang capture server instead of adding an HF/custom target loader to
the trainer. Target TP/EP and model-specific inference stay on that server;
the SpecForge consumer receives only the algorithm's versioned feature schema.
Offline feature training performs no target inference.

EAGLE3 offline sequence parallelism is selected with
`training.attention_backend: usp` plus `training.sp_ulysses_size` and
`training.sp_ring_size`. Evaluation, compact-teacher projection, and experiment
tracking are also config features rather than custom launchers; see the
[training guide](../basic_usage/training.md) for their validated combinations.

## DFlash-family attention modes

DFlash-family draft models select their attention parameterization with
`dflash_config.attention_mode`: `gqa` (the default), `mha`, or `mla`. The mode
swaps only the attention projections inside the shared decoder layer; the
`DFlashDraftModel`, `DFlash2DraftModel`, `DominoDraftModel`, and
`DSparkDraftModel` architectures, target-context injection, per-layer
full/sliding masks, objectives, capture contract, and
`eager`/`sdpa`/`flex_attention` backend selection are identical across modes.
Multi-head Latent Attention is therefore a draft JSON change, not a new
architecture:

```json
{
  "architectures": ["DSparkDraftModel"],
  "hidden_size": 4096,
  "num_attention_heads": 32,
  "q_lora_rank": 1536,
  "kv_lora_rank": 512,
  "qk_nope_head_dim": 128,
  "qk_rope_head_dim": 64,
  "v_head_dim": 128,
  "dflash_config": {
    "projector_type": "dspark",
    "attention_mode": "mla"
  }
}
```

MLA dimensions and behavior use the standard top-level Hugging Face fields:
`q_lora_rank` (`null` selects a direct query projection), `kv_lora_rank`, the
head dims (`qk_nope_head_dim` may be zero; `qk_rope_head_dim` must be even),
and `rope_interleave` for the rotation convention (omitted means interleaved,
the DeepSeek default; `false` selects NeoX-style half rotation). Omitting
`attention_mode` preserves the existing GQA path; `"mha"` requires equal
query/KV head counts.

MLA is a training-side mode: checkpoints train, evaluate through
`spec_generate`, and export through `--to hf`. SGLang serving of DFlash-family
drafts currently implements the GQA/MHA layout only, so plan benchmarks
accordingly. DFlash2 otherwise follows the same mode selection; its convolution
and selector do not change the attention projection contract.

## Draft architectures

Draft classes register through `@register_draft`. The key defaults to the
class name and must match the single entry in the draft JSON's
`architectures` list:

```python
from transformers import PretrainedConfig

from specforge.modeling.draft.models.base import Eagle3DraftModel
from specforge.modeling.draft.registry import register_draft


class MyDraftConfig(PretrainedConfig):
    model_type = "my-draft"


@register_draft
class MyEagle3Draft(Eagle3DraftModel):
    config_class = MyDraftConfig

    def __init__(self, config, **kwargs):
        super().__init__(config)
        ...
```

Import the module from `specforge/modeling/draft/__init__.py` so registration
runs before config resolution. A minimal draft config then contains:

```json
{
  "architectures": ["MyEagle3Draft"],
  "model_type": "my-draft",
  "vocab_size": 128256,
  "draft_vocab_size": 32000
}
```

Embed that mapping under `model.draft_model_config` in the training YAML.
`AutoDraftConfig` (also available under the legacy name `AutoDraftModelConfig`)
resolves configuration types through `specforge.config.draft.registry`; the
model assembler resolves implementations through the model registry. The
`@register_draft` decorator registers custom configuration types and rejects
declarations that conflict with an existing configuration registration. Do not
add a method-specific training launcher.

An architecture alone does not define a new loss. A genuinely new training
algorithm also needs a pure `AlgorithmSpec`, executable `AlgorithmProviders`,
and one immutable `AlgorithmRegistration` under `specforge/algorithms`. Its
step provider may construct a `DraftTrainStrategy`, while its model and data
providers own algorithm-specific assembly. Add builtin registrations to the
explicit catalog used by the application composition root; do not add a
method-specific launcher or a second mutable registry.

### Draft configuration and layer structure

Training reads only the embedded `model.draft_model_config` mapping. Its values
and SpecForge defaults are authoritative: checkpoint or native model metadata
is checked against them, never merged into them. Missing or null
`draft_vocab_size` defaults to `vocab_size`. Capture-layer IDs are explicit;
changing draft depth never chooses a subset of those IDs.

A versioned setting keeps common HF fields at the root, algorithm settings in
their existing sections, and component structure under `layers`:

```yaml
model:
  draft_model_config:
    schema_version: 1
    architectures: [DFlashDraftModel]
    hidden_size: 4096
    intermediate_size: 11008
    num_hidden_layers: 2
    num_attention_heads: 32
    num_key_value_heads: 8
    head_dim: 128
    vocab_size: 151936
    block_size: 8
    dflash_config:
      target_layer_ids: [1, 16, 30]
      mask_token_id: 151669
    layers:
      attention:
        type: gqa
        is_causal: false
      ffn:
        type: dense
      residual:
        type: standard
```

A single layer object is shorthand for repeating the same structure. A list
must have exactly `num_hidden_layers` entries. Internally both forms become
an immutable tuple of `LayerSpec` objects. Each contains an `AttentionSpec`,
`FFNSpec`, `ResidualSpec`, and optional `ConvSpec`. Projection dimensions
and FFN width default from the root HF fields but can be specified per layer.
DFlash2 requires `conv: {type: grouped, kernel_size: ..., group_size: ...}`
on every layer.

Version 1 rejects legacy structural fields such as `layer_types`,
`sliding_window`, `attention_config`, and
`dflash_config.attention_mode`; those belong in `layers`. Unknown layer
or component keys and unsupported schema versions fail before allocation.
Unversioned legacy settings remain accepted by the decoder.

`specforge/config/schema.py` owns the training recipe.
`specforge/config/draft/` owns parsing, defaults, component specs, algorithm
constraints and checkpoint contracts; it does not import model implementations.
`AutoDraftConfig.from_inline` parses YAML data directly into the immutable
runtime `DraftModelConfig`. Typed algorithm options are temporary parsing
objects, not a second live copy. Models and training wrappers read
`model.config`; kernel adapters hold references to that same config and a
single layer. Both the config and its nested settings are fixed after construction.

| Algorithm settings | Definition |
| --- | --- |
| EAGLE3 / P-EAGLE | `config/draft/eagle3.py`, `peagle.py` |
| DFlash / DFlash2 | `config/draft/dflash.py`, `dflash2.py` |
| Domino / DSpark / DSpine | `config/draft/domino.py`, `dspark.py`, `dspine.py` |
| Native Qwen3.5 MTP | `config/draft/mtp.py` |

Algorithms declare constraints, such as causal layers for DSpine and exactly
one layer for EAGLE3. They do not select attention types by name. The component
registry maps a type to a frozen spec class and builders for supported caller
contracts (`context_then_draft`, `eagle_cache`, `peagle`, or `qwen_mtp`).
A new component must implement its contract and register its builder.
Unsupported DSA/QSA/KDA, MoE, GR and mHC choices are rejected; registration
does not silently substitute a dense implementation. Flex/Flash/SDPA are
execution backends, while Ring is a parallel execution strategy.

Cross-layer wiring stays in the model assembly: DFlash injects target context,
DSpine applies injection/refinement, and EAGLE's first layer consumes doubled
hidden width. Specs contain data, not executable hooks.

`model.save_pretrained` writes canonical versioned `config.json`.
`AutoDraftModel.from_pretrained` reloads that layout, including heterogeneous
layers. Legacy HF checkpoint readers also remain available. For serving,
the HF/SGLang exporters write legacy fields to `config.json` and retain
the canonical setting in `specforge_config.json`. Conversion creates a
temporary HF adapter; it never updates the live model config.

Serving conversion validates representability before writing weights. Mixed
full/SWA layers can be exported if their projection structure and shared window
are representable. Different per-layer projection dimensions, FFN widths,
convolution settings, or window sizes fail explicitly until the serving parser
supports them. Being uniform alone does not make an unrecognized component
exportable.

Before loading native Qwen MTP weights, the adapter checks the embedded and
native RoPE types and numerical parameters. Unsupported native RoPE types
fail explicitly. Standalone checkpoints use the full normalized configuration
contract, including all per-layer settings.

## Fused token losses

Import losses from `specforge.core.fused_loss`. Unprefixed functions are eager.

| API pair (eager / fused) | Label input | Per-token objective |
| --- | --- | --- |
| `ce_loss` / `fused_ce_loss` | Teacher logits `[B, T, V]` | CE against the teacher argmax |
| `tv_loss` / `fused_tv_loss` | Teacher logits `[B, T, V]` | `1 - sum(min(p, q))` |
| `ce_l1_loss` / `fused_ce_l1_loss` | Explicit token IDs `[...]`; optional teacher logits `[..., V]` | CE and optional `sum(abs(p - q))`, returned separately |

Here `p = softmax(draft_logits)` and `q = softmax(teacher_logits)`.
The joint CE/L1 API uses the supplied token IDs even when they differ from the
teacher argmax. For normalized distributions, `TV = L1 / 2`. Its backward shares
draft normalization and combines both loss gradients before casting to the
input dtype. This is useful when CE and teacher L1/TV train the same logits.

```python
from specforge.core.fused_loss import fused_ce_l1_loss

logits = lm_head(hidden)                    # [..., V], BF16/FP16/FP32 on CUDA
terms = fused_ce_l1_loss(
    logits,
    target_ids,                             # [...], int64, values in [0, V)
    teacher_logits=teacher_logits,          # optional [..., V], not probabilities
)
per_token_loss = 0.1 * terms.cross_entropy + 0.9 * terms.teacher_l1
loss = (per_token_loss * loss_mask).sum() / loss_mask.sum().clamp_min(1)
loss.backward()
```

The returned `CEL1Loss` contains unreduced FP32 losses. Without teacher logits,
`teacher_l1` is `None`; use `terms.cross_entropy` alone. Both inputs use the same
device and vocabulary. There is no `ignore_index`: masked positions still need
valid token IDs. Teacher logits never receive gradients. Masks, coefficients
and reduction belong to the caller; use `0.5 * terms.teacher_l1` for TV.

`top_k=K` additionally returns differentiable `topk_values` and integer
`topk_ids` with shape `[..., K]`, for DFlash2's selector. `argmax_ids` and the
optional draft/teacher top-1 probabilities are detached diagnostics. The
fused implementation requires CUDA and Triton; use `ce_l1_loss` on CPU.
Importing the public module does not require Triton. The eager reference promotes low-precision
CE to FP32 and preserves explicitly supplied FP64 inputs for numerical checks.

`training.fused_logits_loss` is a boolean, defaulting to `true` for all supported
models, including DFlash2 and Domino. Set `false` to use eager losses.
Unsupported devices and dtypes fall back to eager. Domino uses its specialized
CE kernel under the same flag.

`training.objective_chunk_blocks` sets the chunk size, and
`training.objective_checkpointing` (default `true`) independently controls
recomputation of each chunk's projection and objective. A zero chunk size treats
the whole objective as one chunk; checkpointing still follows its own flag.
Without checkpointing, all chunks can retain
logits until backward: smaller chunks do not bound total saved-logits memory.
DFlash always skips its zero-weight anchor projection in multi-slot objectives;
DSpark supervises every slot. Detailed diagnostics retain the full blocks.

The selector consumes top-k candidates and hidden states, computes its own CE,
and joins the logits backward only through candidate values. Its direct hidden
state gradient is separate; `training.dflash2_selector_stop_gradient` detaches
both inputs.
Weights, masks and loss coefficients are applied outside the logits operators.
