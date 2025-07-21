<div align="center">

  # SGLang Speculative

####  This repository contains code to train Eagle3 models which are compatible with the SGLang framework. 

  <a href="https://huggingface.co/spaces/<your-org-or-username>/<your-repo>">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Repo-yellow.svg?style=flat&logo=huggingface" alt="Hugging Face">
  </a>

[Online Training Quick Start]() |
[Offline Training Quick Start]()

</div>

## 📍 Overview

SGL-Spec is a framework for training speculative decoding models so that you can smoothly port them over to the SGLang serving framework to speed up your inference. We prepared this repository because other open-source repositories are either not well-maintained or not directly compatible with SGLang.

In this repo, we offer two ways to train your eagle model. One is **online training**, which means freezing target model and training draft model at same time. The other is **offline training**, which means using target model get the hidden states first, then train the draft model.

## How to choose training method
### Online:
1. If you disk space is less than 2T, choose online. Offline need more space to train.
### Offline:
2. If you have enough disk space, you can try offline method. It can speedup training.


## 📦 Installation

```bash
pip install -v -e .
```

## 📝 Data Preparation

### Prepare Online Training Dataset

#### Using dataset on huggingface

We have provided a script to prepare ultrachat (200k) and sharegpt (120k) datasets for demo purpose. You can easily process the dataset by running the following command. The jsonl files will be placed in the `cache/dataset/<dataset_name>` directory of the project path by default.

```bash
# ultrachat
python scripts/prepare_data.py --dataset ultrachat

# sharegpt
python scripts/prepare_data.py --dataset sharegpt
```


### Prepare Offline Training Dataset

#### Using dataset on huggingface

We need to filter Dataset same with Online Training, We have provided a script to prepare ultrachat (200k) and sharegpt (120k) datasets for demo purpose. You can easily process the dataset by running the following command. The jsonl files will be placed in the `cache/dataset/<dataset_name>` directory of the project path by default.

```bash
# ultrachat
python scripts/prepare_data.py --dataset ultrachat

# sharegpt
python scripts/prepare_data.py --dataset sharegpt
```

#### Extract dataset hidden states (Offline Only)

> ⚠️ This extract may take about 5T Disk

You need to do one more step for Offline Training: Hidden states generation. the data-path is actuall the output path from the previous `prepare_data.py` script. **This may take about 2 hours**.
- For now this script assumes `TP == WORLD_SIZE`.
- `--num-samples` are used to control the storage. By default it will use all the data from `data-path`.
```bash
export TP=8
torchrun --nproc_per_node=$TP --master_port=29500 scripts/prepare_hidden_states.py \
    --model-path $MODEL_PATH --enable-aux-hidden-states \
    --data-path $DATA_PATH --chat-template llama3 --max-length 2048 \
    --tp-size $TP --batch-size 4 --mem-frac=0.75 \
    --num-samples 1000
```

### Prepare your own dataset

#### Custom data schema

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

#### Prepare data

Then you can following Online/Offline Training.

## 🚀 Training

### 🔥 Online Training

We have provided a simple startup script to train the Eagle3 model for Llama-3.1-8B-Instruct model. You can run the following command to start the training.

```bash
# make sure you have sharegpt data prepared
bash ./examples/run_llama3_eagle3_online.sh
```

### Custom Training

```python
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# train eagle3 for llama3.1-8b
NUM_GPUS=${1:-8}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_online.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-config $ROOT_DIR/configs/llama3-8B-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt.jsonl \
    --output-dir $ROOT_DIR/outputs/llama3-8b-eagle3 \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template llama3 \
    --cache-dir $ROOT_DIR/cache
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

**Wandb integration**: If you wish to log the training progress to Wandb, you can add `--wandb`, `--wandb-key`, `--wandb-project` and `--wandb-name` to the command line in the provided sh file.


### 💨 Offline Training

We have provided a simple startup script to train the Eagle3 model for Llama-3.1-8B-Instruct model. You can run the following command to start the training. Almost Everything is the same as the Online Training Step, except that you don't need to config anything about target model. You need to parse `--train-hidden-states-path` to the file.

```bash
# make sure you have sharegpt data prepared
bash ./examples/run_llama3_eagle3_offline.sh
```


## 🤖 Serving and Benchmarking On SGLang


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

## ✨ Acknowledgements
We would like to express our sincere gratitude to the EAGLE official team, especially Hongyang Zhang and Yuhui Li, for their valuable work and support. We are also thankful to the NVIDIA team, in particular Avery H and Izzy Putterman, as well as the Google team, especially Ying Wang, for their insightful discussions and generous help throughout this project.

We would like to extend our special thanks to Meituan for their strong support and contributions, which have played a key role in driving this project forward.

This project has also been inspired by many outstanding open-source projects from the LLM community, including [EAGLE](https://github.com/SafeAILab/EAGLE), [BaldEagle](https://github.com/NickL77/BaldEagle), and [TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) and others. Their contributions and shared knowledge have greatly benefited our work.
