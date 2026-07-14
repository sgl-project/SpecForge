# EAGLE3 for Llama 3.1 8B: online training

Online training captures the target model's hidden states while the draft model
is training. This walkthrough uses ShareGPT for a small example; a broader,
target-regenerated dataset is recommended for production checkpoints.

## 1. Prepare ShareGPT

Run from the repository root:

```bash
python ./scripts/prepare_data.py --dataset sharegpt
```

## 2. Use the checked-in run config

The canonical recipe is
[`examples/configs/llama3.1-8b-eagle3-online.yaml`](../../examples/configs/llama3.1-8b-eagle3-online.yaml).
It already points at the ShareGPT output from step 1 and records the target,
draft architecture, SGLang backend, optimizer settings, and output directory in
the same typed contract used by every other training method. Edit that file or
use dotted command-line overrides when model and data paths differ; do not
create a second copy of the recipe in documentation.

## 3. Train

```bash
specforge train --config examples/configs/llama3.1-8b-eagle3-online.yaml
```

The recipe points at an external SGLang capture server and starts the
disaggregated producer/consumer roles. Configure target TP on the SGLang
server; `deployment.trainer` controls only consumer data parallelism. See
[Parallel topologies](../basic_usage/training.md#parallel-topologies). For a
short smoke run, append a small `training.max_steps` dotted override.

## 4. Export and benchmark

Llama 3.1 8B training and benchmarking should use the same system prompt. A
reference draft checkpoint is available at
[zhuyksir/EAGLE3-Llama-3.1-8B-Instruct](https://huggingface.co/zhuyksir/EAGLE3-Llama-3.1-8B-Instruct).

The runtime checkpoint contains training state but is not directly loadable by
the SGLang speculative decoder. Online resume is consumer-only and reuses the
retained transport state; export the final checkpoint before serving:

```bash
specforge export --to sglang \
  --checkpoint ./outputs/llama3.1-8b-eagle3-online/llama3.1-8b-eagle3-online-latest \
  --draft-config configs/llama3-8B-eagle3.json \
  --output-dir ./exports/llama3.1-8b-eagle3-sglang
```
