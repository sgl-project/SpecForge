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

MLA checkpoints train, evaluate through `spec_generate`, and export through
`--to hf`. SGLang serves DFlash and DFlash2 MLA exports as is
([SGLang PR #44676](https://github.com/sgl-project/sglang/pull/44676));
DSpark MLA exports do not serve yet. DFlash2 otherwise follows the same mode
selection; its convolution and selector do not change the attention projection
contract.

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

Point `model.draft_model_config` at that JSON. `AutoDraftModelConfig` and the
model assembler resolve both the config and model class from the registry; do
not add a method-specific training launcher.

An architecture alone does not define a new loss. A genuinely new training
algorithm also needs a pure `AlgorithmSpec`, executable `AlgorithmProviders`,
and one immutable `AlgorithmRegistration` under `specforge/algorithms`. Its
step provider may construct a `DraftTrainStrategy`, while its model and data
providers own algorithm-specific assembly. Add builtin registrations to the
explicit catalog used by the application composition root; do not add a
method-specific launcher or a second mutable registry.

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
