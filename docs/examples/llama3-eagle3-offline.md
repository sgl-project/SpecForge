# EAGLE3 for Llama 3.1 8B: offline training

Offline training captures target features ahead of time and reads them from
disk while training the draft. It reduces GPU memory pressure during training,
but feature storage can be much larger than the source dataset.

## 1. Prepare ShareGPT

```bash
python ./scripts/prepare_data.py --dataset sharegpt
```

## 2. Capture hidden states

Feature preparation is a data-processing step, not a second training entry
point:

```bash
torchrun --standalone --nproc_per_node 8 \
  scripts/prepare_hidden_states.py \
  --target-model-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./cache/dataset/sharegpt_train.jsonl \
  --output-path ./cache/hidden_states/sharegpt-llama3-8b \
  --chat-template llama3 \
  --max-length 4096 \
  --tp-size 1 \
  --batch-size 32
```

The output directory contains the feature checkpoints consumed by the unified
trainer. Offline EAGLE3 with a compact draft vocabulary also needs a matching
`t2d`/`d2t` mapping generated from the same tokenized training corpus.

## 3. Create the run config

Create `llama3-eagle3-offline.yaml`:

```yaml
model:
  target_model_path: meta-llama/Llama-3.1-8B-Instruct
  draft_model_config: configs/llama3-8B-eagle3.json
  embedding_key: model.embed_tokens.weight
  vocab_mapping_path: ./cache/vocab_mapping/llama3-8b-eagle3.pt
  torch_dtype: bfloat16

data:
  hidden_states_path: ./cache/hidden_states/sharegpt-llama3-8b
  max_length: 4096
  chat_template: llama3
  cache_dir: ./cache

training:
  strategy: eagle3
  deployment_mode: local_colocated
  num_epochs: 10
  batch_size: 1
  learning_rate: 1.0e-4
  max_grad_norm: 0.5
  ttt_length: 7
  attention_backend: flex_attention
  save_interval: 1000
  log_interval: 50

run_id: llama3-8b-eagle3-offline
output_dir: ./outputs/llama3-8b-eagle3-offline
```

The checked-in [Qwen offline
config](../../examples/configs/qwen3-8b-eagle3-offline.yaml) is another complete
reference.

## 4. Train

```bash
specforge train --config ./llama3-eagle3-offline.yaml
```

The same entry supports offline data parallelism:

```bash
torchrun --standalone --nproc_per_node=4 "$(command -v specforge)" \
  train --config ./llama3-eagle3-offline.yaml
```

For long sequences, EAGLE3 offline can instead use USP by setting
`training.attention_backend=usp` and choosing
`training.sp_ulysses_size`/`training.sp_ring_size`. Offline feature training
also supports DFlash and Domino when the feature checkpoints and draft config
use that strategy's contract. `training.compact_teacher: true` enables the
exact lower-memory teacher projection for offline text EAGLE3.
