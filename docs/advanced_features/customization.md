# Customize a training run

Customization starts from a typed YAML config. Pick the closest checked-in
file under [`examples/configs`](../../examples/configs), change model and data
paths, and keep the same training entry point:

```bash
specforge train --config ./my-run.yaml
```

For one-off changes, use dotted overrides rather than adding another launcher:

```bash
specforge train \
  --config examples/configs/qwen3-8b-eagle3-online.yaml \
  model.target_model_path=/models/my-target \
  data.train_data_path=/datasets/my-training-data.jsonl \
  training.learning_rate=5e-5
```

The config is strict: unknown fields, invalid strategy/topology combinations,
and unsupported attention backends are errors. See
[`specforge/config/schema.py`](../../specforge/config/schema.py) and the
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

The unified runtime currently accepts text-only input. Multimodal/VLM config
fields are rejected. Attention backends are a closed, strategy-specific set:

| Strategy | Accepted `training.attention_backend` values |
| --- | --- |
| EAGLE3 | `sdpa`, `flex_attention`, `fa` |
| P-EAGLE | `flex_attention` |
| DFlash, Domino, DSpark | `eager`, `sdpa`, `flex_attention` |

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

`model.target_backend` is also strategy- and topology-specific. Online
disaggregation requires `sglang`. A `custom` backend is accepted only for
EAGLE3 and P-EAGLE; DFlash-family strategies use `sglang` or `hf` where the
selected topology supports them.

Large target models may use a tensor-parallel custom implementation under
`specforge/modeling/target/custom_backend`. This is an implementation detail of
target inference inside the supported run; it does not make colocated or
offline draft training multi-rank. Follow the existing Transformers
`PreTrainedModel` implementations there and use SpecForge's parallel linear
layers where the target model is sharded:

```python
from specforge.layers.linear import ColumnParallelLinear, RowParallelLinear


class MyModelForCausalLM(MyModelPreTrainedModel, GenerationMixin):
    ...

    def load_weights(self, state_dict):
        ...
```

Register the target config class in `AutoDistributedTargetModel` in
`specforge/modeling/auto.py`, then select `model.target_backend: custom`:

```diff
class AutoDistributedTargetModel(AutoModelForCausalLMBase):
    _model_mapping = {
        Llama4TextConfig: [Llama4ForCausalLM],
+       MyModelConfig: [MyModelForCausalLM],
    }
```

The target backend implements target inference only. Run orchestration remains
in the `specforge train` assembly path.

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
algorithm also needs a `DraftTrainStrategy` and `StrategySpec` registration in
`specforge/training/strategies`, plus a corresponding validated strategy value.
