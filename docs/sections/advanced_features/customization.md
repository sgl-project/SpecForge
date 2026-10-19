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

## MoE FFN for DFlash-family drafts

Any DFlash-family draft (DFlash, DFlash2, DSpark, Domino) swaps its dense MLP
for a sparse MoE FFN when the draft JSON names a `moe_preset`. The MoE is one
configurable layer (`specforge/modeling/draft/moe/`, see its `DESIGN.md`): the
preset is a target family's routing recipe, registered by that family's
module under `moe/families/`, and the JSON states the per-run sizes:

```json
{
  "moe_preset": "<family preset>",
  "n_routed_experts": 64,
  "num_experts_per_tok": 6,
  "moe_intermediate_size": 2048,
  "dflash_config": {"moe_dispatch": "grouped_mm"}
}
```

- **Architecture keys** at the top level override the preset for ablations
  (`scoring_func`, `norm_topk_prob`, `routed_scaling_factor`, `balance`,
  `n_shared_experts`, `shared_expert_intermediate_size`, ...). `moe_layout`
  overrides the checkpoint file naming the preset selects. Expert counts must
  be positive integers, and `n_routed_experts` without a `moe_preset` is an
  error, never a silent dense fallback.
- **Training-only keys** live under `dflash_config` with an `moe_` prefix and
  never change the checkpoint's architecture: `moe_bias_update_rate` and
  `moe_aux_loss_coeff` (balancing), and `moe_dispatch` (`sorted_loop` or
  `grouped_mm`).
- **Losses and metrics** need no strategy code: the trainer adds every MoE
  layer's auxiliary loss to the objective and logs layer-averaged `moe/*`
  load metrics for every strategy.
- **Checkpoints and exports** name MoE tensors through the preset's layout.
  Serving an MoE draft requires the serving engine to implement the draft
  architecture with an MoE FFN; check the family's documentation.
- **New families and components** register by name: a module under
  `moe/families/` registers a family's preset and layout, and a score
  function, router, balance controller, experts backend or shared expert
  registers from its own module (see `DESIGN.md`).

### Drafters with their own architecture

A drafter that does not use the DFlash decoder layers, such as a target's
official drafter, uses the same layer directly:

- build each FFN as `MoELayer(resolve_moe_config(config), config.hidden_size)`;
- call `apply_pending_balance_updates(self)` at the top of the training
  forward, outside activation-checkpoint regions;
- call `MoELayer.reset_parameters(std)` from `_init_weights`;
- let the preset name the family's layout, so checkpoints and exports keep the
  official tensor names.

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
