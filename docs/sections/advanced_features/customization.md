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

from specforge.modeling.draft.base import Eagle3DraftModel
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

### Draft configuration parsing

`specforge.config.draft.AutoDraftConfig` is the shared loader used by
training, export, and `AutoDraftModel.from_pretrained` when no explicit config
object is supplied. Use `from_dict`, `from_file`, or `from_pretrained` (a JSON
file, directory, or Hugging Face repository). The configuration registry declares
the HF serialization class and the typed options class for each architecture.
Repository loading reads JSON without executing an `auto_map` config
implementation. Custom draft architectures must be registered locally before
loading.

`specforge/config/schema.py` defines the training recipe and its inline draft
mapping. `specforge/config/draft/` owns draft parsing, defaults, architecture
validation, and checkpoint configuration contracts. It does not import
`specforge.modeling`; models and export code depend on these configuration types.

To support parsing a custom architecture without importing its model, register
its configuration in a configuration-only module:

```python
from specforge.config.draft import register_draft_config

register_draft_config("MyEagle3Draft", MyDraftConfig, MyDraftOptions)
```

`MyDraftOptions` implements `DraftConfig`. The model declares the same classes
as `config_class` and `draft_options_class`; model registration verifies that
they agree. Omit the options class for a custom model that uses only HF fields.

All entry points require exactly one registered `architectures` entry, force
`tie_word_embeddings=False`, and default an omitted or null `draft_vocab_size`
to the resolved `vocab_size`, including through the legacy loader. Explicit
non-positive or non-integer draft vocabulary values are rejected. Nested custom
fields are preserved and copied so parsing does not modify the caller's input.

Training reads only the inline `model.draft_model_config` mapping. It calls
`AutoDraftConfig.from_inline` and the registered options class's
`validate_training_config` before allocating model parameters. File/repository
readers remain available for pretrained artifacts and checkpoint compatibility
checks; they never supply or override training settings.

`config/draft/base.py` defines the immutable `DraftConfig` contract:
`from_model_config` resolves and validates options without allocating parameters;
`to_model_config` writes normalized fields into a temporary parsing object.
These typed options are discarded after parsing. Architecture-specific implementations are:

| Draft architecture | Resolved options | File |
| --- | --- | --- |
| EAGLE3 | `Eagle3Config` | `config/draft/eagle3.py` |
| P-EAGLE | `PEagleConfig` | `config/draft/peagle.py` |
| DFlash | `DFlashConfig` | `config/draft/dflash.py` |
| DFlash2 | `DFlash2Config` | `config/draft/dflash2.py` |
| Domino | `DominoConfig` | `config/draft/domino.py` |
| DSpark | `DSparkConfig` | `config/draft/dspark.py` |
| DSpine | `DSpineDraftConfig` | `config/draft/dspine.py` |
| Qwen3.5 MTP | `Qwen3_5MTPConfig` | `config/draft/mtp.py` |

Each built-in model exposes `draft_options_class` for parsing and validation.
The runtime `DraftModelConfig` lives in `config/draft/model.py`. It contains the
resolved inline values and defaults, with immutable nested mappings and tuples.
Models, layers, and training wrappers read the same `model.config`; initialization
does not retain a second spec or copy algorithm settings into model attributes.
Configuration cannot be changed or replaced after construction.

`model.config.to_hf_config()` creates an independent, mutable HF adapter for
framework initialization, checkpoint comparison, and serving export.
`save_pretrained` uses that adapter without changing the live configuration.
Saving never writes HF serialization metadata back into the live configuration. Legacy checkpoint
readers remain available; direct model construction resolves their HF configs
through the same immutable runtime contract.

New architectures should implement the configuration contract and add a case to
`test_all_draft_configs.py`. Common model-family defaults are explicit in
`config/draft/model.py`; method-specific defaults stay in their typed parsers.

EAGLE's HF adapter remains a `LlamaConfig` subclass and retains `model_type=llama`.
It permits the explicit projection width used by EAGLE when hidden size is not
divisible by query head count. The current EAGLE3 and Qwen3.5 MTP implementations
require one layer; P-EAGLE supports multiple layers.

Before constructing a draft from native Qwen MTP weights, the adapter checks
that the inline and native RoPE types match and are supported. Qwen3.5 MTP
currently supports only default RoPE; its theta and partial rotary factor must
also match. The shared checks in `config/draft/rope.py` read modern
`rope_parameters.rope_type` and legacy `rope_scaling.type` declarations without
modifying either configuration. Conflicting declarations and per-layer layouts
require an explicit adapter instead of falling back to default RoPE. Future
native draft adapters must invoke this check with their own supported types
and validate the numerical parameters used by their implementation.

DSpark currently loads standalone draft checkpoints, not native target draft
weights. Its checkpoint contract already compares RoPE types and scaling
parameters against the inline settings. A draft trained from scratch need not
use the target model's RoPE.

Boolean architecture switches
must be booleans, not strings. P-EAGLE's `norm_before_residual` and EAGLE-family
capture layers live in the inline draft mapping. Loss weights and sampling policy remain
owned by the algorithm/run config.

Finalization returns a copy and preserves unknown JSON fields. Top-level
`block_size` retains precedence over the legacy `dflash_config.block_size` alias.
The resolved runtime config keeps only the top-level value; HF serialization adds
the nested alias to its export copy. Default capture layers and head switches are
made explicit for serialization. Model construction does not mutate the caller's
config. Layers share the resolved config; their weights and derived tensor dimensions
are constructed from it. Create a new model to change its architecture.
Saving uses a private config copy, so HF serving metadata never changes the
live training config, even if serialization fails.
Converting a resolved view back to HF config writes its attention layout, sliding
window, and direction along with the method settings. Attention dimensions and
active sliding windows must be integers; fractional values and booleans are not
silently converted to dimensions.

Changing `num_hidden_layers` requires a matching explicit attention layout.
Capture layer IDs are retained as configured; changing draft depth never silently
selects a subset of those IDs. `config/draft/contract.py` records normalized
settings in training checkpoint provenance. It ignores execution/serialization
metadata and retains new structural fields by default. Native architectures own
their source-specific checks (for example `Qwen3_5MTPConfig.validate_native_config`).

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
