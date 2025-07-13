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

```bash
# make sure you have sharegpt data prepared
bash ./examples/run_llama3_eagle3.sh
```


### 💨 Offline Training

To be added.

## 🤖 Serving
