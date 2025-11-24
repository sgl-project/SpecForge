# 💡 Customize Your Own Training

## 🔧 Customize Training Args

```bash
torchrun \
    --standalone \
    --nproc_per_node 8 \
    ./scripts/train_eagle3.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-config ./configs/llama3-8B-eagle3.json \
    --train-data-path ./cache/dataset/sharegpt.jsonl \
    --output-dir ./outputs/llama3-8b-eagle3 \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template llama3 \
    --cache-dir ./cache
```

If you wish to understand what each argument does, you can run `python scripts/train_eagle3.py --help` to see the full list of arguments. Particularly, we will discuss some important arguments below.
- `--chat-template`: This should be the chat template to use for the model, so please make sure you set it to the correct value.
- `--cache-dir`: This directory contains the dataset cache including the `input_ids`, `loss_mask`, `attention_mask` and `vocab_mapping`. These caches can make your data loading much faster once a cache is generated. The cache file has a name which is obtained by hashing the dataset path to avoid cache collision.

## 💬 Customize Chat Template

You can register a new chat template for your model by adding a new entry to the `TEMPLATE_REGISTRY` in the `specforge.data.template.py` file.

```python
TEMPLATE_REGISTRY.register(
    name="your-template-name",
    template=ChatTemplate(
        assistant_header="xxx",
        user_header="xxx",
        system_prompt="xxx",
        end_of_turn_token="xxx",
    ),
)
```

## 🪅 Customize Model

### Customize Target Model

If you wish to train Eagle3 for other models, you need to modify the `--target-model-path` value. We support loading these models directly from HuggingFace.

However, if your model is too large and requires tensor parallelism, you can implement its tensor parallel version on your own in the `specforge.modeling.target` directory. The CausalLM model should inherit the `DistributedTargetModel` class in the `specforge.modeling.target.base.py` file and apply `ColumnParallelLinear` and `RowParallelLinear` to its submodules.

```python
from .base import DistributedTargetModel
from specforge.layers.linear import ColumnParallelLinear, RowParallelLinear


class MyModelForCausalLM(MyModelPreTrainedModel, GenerationMixin, DistributedTargetModel):
    ...

    def load_weights(self, state_dict: Dict[str, torch.Tensor]):
        ...
```

Afterwards, you need to register this model to the `AutoEagle3TargetModel` class in the `specforge.modeling.auto.py` file.

```diff
class AutoDistributedTargetModel(AutoModelForCausalLMBase):
    _model_mapping = {
        Llama4TextConfig: [Llama4ForCausalLM],
+       MyModelConfig: [MyModelForCausalLM],
    }
```

When `tp_size` is greater than 1, the script will automatically load the distributed version of the model for tensor parallelism.

### Customize Draft Model

If you want to change the draft model configuration, you can write your own configuration file and pass its path to the `--draft-model-config` argument. Or, if you do not provide the `--draft-model-config` argument, the script will automatically generate the draft model configuration based on the target model configuration. If you wish to serve your customized draft model with SGLang, make sure you implement the draft model in SGLang as well and the architecture name must match. To implement your own draft model, you can create a new class and inherit it from the `Eagle3DraftModel` class in the `specforge.modeling.draft.base.py` file.


```python
from .base import Eagle3DraftModel
from transformers import PretrainedConfig


class MyModelConfig(PretrainedConfig):
    model_type = "mymodel"

    def __init__(self, **kwargs):
        ...


class MyModelEagle3(Eagle3DraftModel):

    config_class = MyModelConfig

    def __init__(self, config, quant_config=None) -> None:
        ...
```

You can then register these models to the `AutoEagle3TargetModel` and `AutoDraftModelConfig` classes in the `specforge.modeling.auto.py` file for the automatic model loading.

```diff
class AutoEagle3DraftModel(AutoModelForCausalLMBase):
    # the model mapping is currently hardcoded, we should support lazy model mapping via registry
    _model_mapping = {
        LlamaConfig: [LlamaForCausalLMEagle3],
+       MyModelConfig: MyModelEagle3,
    }


class AutoDraftModelConfig:

    _config_mapping = {
        "LlamaForCausalLMEagle3": LlamaConfig,
+       "MyModelEagle3": MyModelConfig,
    }
```

In this way, as long as your `config.json` specifies the correct architecture name, the script will automatically load the correct draft model for you.
