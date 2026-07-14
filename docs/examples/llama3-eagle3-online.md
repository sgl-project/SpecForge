# EAGLE3 for Llama 3.1 8B: online training

Online training captures the target model's hidden states while the draft model
is training. This walkthrough uses ShareGPT for a small example; a broader,
target-regenerated dataset is recommended for production checkpoints.

## 1. Prepare ShareGPT

Run from the repository root:

```bash
python ./scripts/prepare_data.py --dataset sharegpt
```

## 2. Create the run config

Create `llama3-eagle3-online.yaml` with the following contents. This is the same
typed contract used by the checked-in [Qwen online
example](../../examples/configs/qwen3-8b-eagle3-online.yaml).

```yaml
model:
  target_model_path: meta-llama/Llama-3.1-8B-Instruct
  draft_model_config: configs/llama3-8B-eagle3.json
  target_backend: sglang
  embedding_key: model.embed_tokens.weight
  torch_dtype: bfloat16

data:
  train_data_path: ./cache/dataset/sharegpt_train.jsonl
  max_length: 4096
  chat_template: llama3
  build_dataset_num_proc: 32
  cache_dir: ./cache

training:
  strategy: eagle3
  deployment_mode: local_colocated
  num_epochs: 2
  batch_size: 1
  learning_rate: 1.0e-4
  max_grad_norm: 0.5
  ttt_length: 7
  attention_backend: flex_attention
  save_interval: 1000
  log_interval: 50

run_id: llama3-8b-eagle3-online
output_dir: ./outputs/llama3-8b-eagle3-online
```

## 3. Train

```bash
specforge train --config ./llama3-eagle3-online.yaml
```

Colocated training is currently single-rank. For a short smoke run, override
`training.max_steps` to a small value.

To train on Perfect-Blend, prepare it with
`python ./scripts/prepare_data.py --dataset perfectblend`, then point the same
config at the generated file:

```bash
specforge train \
  --config ./llama3-eagle3-online.yaml \
  data.train_data_path=./cache/dataset/perfectblend_train.jsonl
```

## 4. Export and benchmark

Llama 3.1 8B training and benchmarking should use the same system prompt. A
reference draft checkpoint is available at
[zhuyksir/EAGLE3-Llama-3.1-8B-Instruct](https://huggingface.co/zhuyksir/EAGLE3-Llama-3.1-8B-Instruct).

The runtime checkpoint contains training state, but the unified CLI does not
support resuming any online stream in this PR. It is also not directly loadable
by the SGLang speculative decoder. Export the final checkpoint first:

```bash
specforge export --to sglang \
  --checkpoint ./outputs/llama3-8b-eagle3-online/llama3-8b-eagle3-online-latest \
  --draft-config configs/llama3-8B-eagle3.json \
  --output-dir ./exports/llama3-8b-eagle3-sglang
```

From the `benchmarks` directory, point `--speculative-draft-model-path` at the
exported directory:

```bash
config_list=(
    "4,3,1,4"
    "4,7,10,60"
)
python3 bench_eagle3.py \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --speculative-draft-model-path ../exports/llama3-8b-eagle3-sglang \
    --port 30000 \
    --mem-fraction-static 0.8 \
    --tp-size 1 \
    --config-list "${config_list[@]}" \
    --benchmark-list mtbench gsm8k humaneval math500
```
