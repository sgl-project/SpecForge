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
  --draft-model-config configs/llama3.1-8b-eagle3.json \
  --data-path ./cache/dataset/sharegpt_train.jsonl \
  --output-path ./cache/hidden_states/sharegpt_train_Llama-3.1-8B-Instruct \
  --chat-template llama3 \
  --max-length 4096 \
  --tp-size 1 \
  --batch-size 32
```

The output directory now matches the hidden-state path in the checked-in
recipe. It contains the feature checkpoints consumed by the unified trainer
and `vocab_mapping/vocab_mapping.pt`, derived from the same processed corpus.

## 3. Use the checked-in run config

The canonical recipe is
[`examples/configs/llama3.1-8b-eagle3-offline.yaml`](../../examples/configs/llama3.1-8b-eagle3-offline.yaml).
It records the same target, feature directory, draft architecture, and trainer
settings used by this walkthrough; edit the checked-in recipe or use dotted
overrides instead of copying its YAML into another document.

The recipe carries a conventional `model.vocab_mapping_path` for deployments
that prepare and share a mapping artifact. For a local offline run, leaving
that field empty makes SpecForge count effective tokens in the exact feature
corpus, derive `t2d`/`d2t` deterministically, and cache the reusable mapping
under `data.cache_dir/vocab_mapping`. Equal target and draft vocabularies need
no mapping.

## 4. Train

```bash
specforge train \
  --config examples/configs/llama3.1-8b-eagle3-offline.yaml \
  model.vocab_mapping_path=./cache/hidden_states/sharegpt_train_Llama-3.1-8B-Instruct/vocab_mapping/vocab_mapping.pt
```

Omit the final override when the mapping file named by the recipe already
exists. The same recipe supports offline data parallelism through the typed
`deployment.trainer` topology and unified self-launcher.

For long sequences, EAGLE3 offline can instead use USP by setting
`training.attention_backend=usp` and choosing
`training.sp_ulysses_size`/`training.sp_ring_size`. Offline feature training
also supports DFlash and Domino when the feature checkpoints and draft config
use that strategy's contract. `training.compact_teacher: true` enables the
exact lower-memory teacher projection for offline text EAGLE3. See
[Parallel topologies](../basic_usage/training.md#parallel-topologies) for the
multi-process launch contract.
