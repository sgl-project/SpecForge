---
base_model: meta-llama/Meta-Llama-3.1-8B
datasets:
- llama-duo/synth_summarize_dataset_dedup
library_name: peft
license: llama3.1
tags:
- alignment-handbook
- trl
- sft
- generated_from_trainer
model-index:
- name: llama3.1-8b-summarize-gpt4o-128k
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# llama3.1-8b-summarize-gpt4o-128k

This model is a fine-tuned version of [meta-llama/Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B) on the llama-duo/synth_summarize_dataset_dedup dataset.
It achieves the following results on the evaluation set:
- Loss: 4.0859

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 4
- eval_batch_size: 2
- seed: 42
- distributed_type: multi-GPU
- num_devices: 4
- gradient_accumulation_steps: 2
- total_train_batch_size: 32
- total_eval_batch_size: 8
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 10

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 1.0008        | 0.9990 | 519  | 2.1032          |
| 0.9747        | 2.0    | 1039 | 2.1444          |
| 0.9289        | 2.9990 | 1558 | 2.2517          |
| 0.8818        | 4.0    | 2078 | 2.4632          |
| 0.8109        | 4.9990 | 2597 | 2.7084          |
| 0.7513        | 6.0    | 3117 | 2.9358          |
| 0.7004        | 6.9990 | 3636 | 3.2769          |
| 0.6466        | 8.0    | 4156 | 3.6948          |
| 0.6132        | 8.9990 | 4675 | 3.9708          |
| 0.5965        | 9.9904 | 5190 | 4.0859          |


### Framework versions

- PEFT 0.12.0
- Transformers 4.44.0
- Pytorch 2.4.0+cu121
- Datasets 2.20.0
- Tokenizers 0.19.1
