# ⚡️ SGL-Spec

This repository contains code to train Eagle3 models which are compatible with the SGLang framework.

## 📍 Overview

SGL-Spec is a framework for training speculative decoding models so that you can smoothly port them over to the SGLang serving framework to speed up your inference. We prepared this repository because other open-source repositories are either not well-maintained or not directly compatible with SGLang.


## 📦 Installation

```bash
pip install -v -e .
```

## 📝 Data Preparation

In order to run the training script smoothly, you should prepare the dataset in jsonl format and the schema should look like this:

```json
{
    "id": "xxxx",
    "conversations": [
        {
            "role": "user" | "assistant",
            "content": "The message content"
        }
    ],
}
```

We have also provided a script to prepare ultrachat (200k) and sharegpt (120k) datasets for demo purpose. You can easily process the dataset by running the following command. The jsonl files will be placed in the `cache/dataset/<dataset_name>` directory of the project path by default.

```bash
# ultrachat
python scripts/prepare_data.py --dataset ultrachat

# sharegpt
python scripts/prepare_data.py --dataset sharegpt
```

## 🚀 Training

### 🔥 Online Training

We have provided a simple startup script to train the Eagle3 model for Llama-3.1-8B-Instruct model. You can run the following command to start the training.

```bash
# make sure you have sharegpt data prepared
bash ./examples/run_llama3_eagle3.sh
```

**Customize target model**: If you wish to train Eagle3 for other models, you need to modify the `--target-model-path` value. We support loading these models directly from HuggingFace.

**Customize draft model**: If you want to change the draft model configuration, you can write your own configuration file and pass its path to the `--draft-model-config` argument. The architecture name must match the draft model types provided in the SGL-Spec library. If you want to implement your own draft model, you can create a new class and inherit it from the `Eagle3DraftModel` class in the `sgl_spec.modeling.draft.base.py` file.


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

You can then register these models to the `AutoEagle3TargetModel` and `AutoDraftModelConfig` classes in the `sgl_spec.modeling.auto.py` file for the automatic model loading.

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

**Customize Data**: If you want to use your own dataset, make sure you prepared the dataset in the same format as given in the [data preparation section](#-data-preparation). If you have multiple datasets, you can just merge them into the one jsonl file. For example, you can do something like `cat dataset1.jsonl dataset2.jsonl > merged_dataset.jsonl`.

**Wandb integration**: If you wish to log the training progress to WanDB, you can add `--wandb`, `--wandb-key`, `--wandb-project` and `--wandb-name` to the command line in the provided sh file.


### 💨 Offline Training

To be added.

## 🤖 Serving and Benchmarking


You can serve your trained model with SGLang with the following command.

```bash
python3 -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct  \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path <saved_checkpoint_path> \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --mem-fraction-static 0.75 \
    --cuda-graph-max-bs 2 \
    --tp 1 \
    --context-length 8192 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16
```


In SGLang, we have provided a benchmark script to evaluate the performance of the trained Eagle model. You can run the following command in SGLang to execute the benchmark script.

```bash
# prepare data
cd sglang/benchmark/mtbench
wget -O question.jsonl https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl


# run benchmarking
python3 bench_sglang_eagle.py --num-questions 80 --port 30000
```
