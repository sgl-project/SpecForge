# One-sample overfit validation

This guide contains only two stages:

1. overfit Dspark on one Qwen3-8B training sample with `specforge train`;
2. export the checkpoint, serve it with SGLang, and verify that one complete
   16-token draft block is accepted.

The commands below use GPU 0 for target capture and GPU 1 for training and
serving. We use Qwen3.6-27B-Dspark as an example. You can test other models and speculative algorithms such as Dspark、DFlash and so on.

## Validate the ports

Run this check from the SpecForge repository before preparing data. It checks
only whether the capture, serving, and Mooncake ports are available.

```bash
source scripts/gates/_e2e_common.sh

gate_report_tcp_ports python 127.0.0.1 \
  capture_port 32000 \
  serving_port 32001 \
  mooncake_rpc_port 37551 \
  mooncake_metadata_port 37880 \
  mooncake_metrics_port 37903
```

The check always returns to the shell. If one or more ports are occupied, it
prints every conflict without stopping or killing the owning processes.

## Prepare one sample

### Stage 1: train with SpecForge

```bash
export TRAIN_DATA_PATH=/sgl-workspace/perfectblend/full_1.jsonl
export MODEL_NAME=Qwen3.6-27B
export SPEC_METHOD=Dspark
export DRAFT_MODEL_CONFIG=configs/qwen3.6-27b-dspark.json
export MODEL=Qwen/Qwen3.6-27B

specforge train \
  --config examples/configs/qwen3.6-27b-dspark-disaggregated.yaml \
  model.target_model_path=${MODEL} \
  model.draft_model_config=${DRAFT_MODEL_CONFIG} \
  model.sglang_context_length=4200 \
  data.train_data_path=${TRAIN_DATA_PATH} \
  data.max_prompts=1 \
  data.max_length=4096 \
  data.chat_template=qwen3.5 \
  training.num_epochs=600 \
  training.max_steps=600 \
  training.total_steps=600 \
  training.batch_size=1 \
  training.accumulation_steps=1 \
  training.eval_interval=0 \
  training.save_interval=600 \
  training.log_interval=1 \
  deployment.trainer.nproc_per_node=1 \
  deployment.disaggregated.control_dir=outputs/${MODEL_NAME}-${SPEC_METHOD}-one-sample/control \
  deployment.disaggregated.consumer_state_dir=outputs/${MODEL_NAME}-${SPEC_METHOD}-one-sample/consumer-state \
  'deployment.disaggregated.managed_local.trainer_cuda_visible_devices=["1"]' \
  'deployment.disaggregated.managed_local.capture_servers=[{"port":32000,"cuda_visible_devices":["0"],"tp_size":1,"mem_fraction_static":0.7}]' \
  deployment.disaggregated.managed_local.mooncake.rpc_port=37551 \
  deployment.disaggregated.managed_local.mooncake.metadata_port=37880 \
  deployment.disaggregated.managed_local.mooncake.metrics_port=37903 \
  tracking.report_to=none \
  runtime.in_flight_high_watermark=64 \
  runtime.in_flight_low_watermark=32 \
  runtime.producer_lease=1 \
  run_id=${MODEL_NAME}-${SPEC_METHOD}-one-sample \
  output_dir=outputs/${MODEL_NAME}-${SPEC_METHOD}-one-sample/consumer \
  model.embedding_key="model.language_model.embed_tokens.weight" \
  model.mask_token_id=248070
```
<span style="color: red;"><strong>Note:</strong></span>The training data must be inferenced by the target model use temperature=0 and corresponding thinking mode.</span>
This single command owns Mooncake, the SGLang capture server, the producer, and
the trainer. It also assigns the capture server to GPU 0 and the trainer to GPU
1 from the typed configuration.

After step 600, the checkpoint should be:

```text
outputs/${MODEL_NAME}-${SPEC_METHOD}-one-sample/consumer/${MODEL_NAME}-${SPEC_METHOD}-one-sample-step600
```

The `control` and `consumer-state` directories must be fresh. To repeat the
experiment, use a new suffix consistently in `run_id`, `output_dir`,
`control_dir`, `consumer_state_dir`, and the sample paths.

### Stage 2: serve with SGLang and check accept length

First export the trained draft:

```bash
# Export the draft model to HF format
specforge export \
  --to hf \
  --checkpoint outputs/${MODEL_NAME}-${SPEC_METHOD}-one-sample/consumer/${MODEL_NAME}-${SPEC_METHOD}-one-sample-step600 \
  --draft-config ${DRAFT_MODEL_CONFIG} \
  --output-dir outputs/${MODEL_NAME}-${SPEC_METHOD}-one-sample/draft_hf \
  --embedding-source ${MODEL} \
  --embedding-key model.language_model.embed_tokens.weight

# Normalize the config
python scripts/gates/normalize_dflash_export.py \
  --config outputs/${MODEL_NAME}-${SPEC_METHOD}-one-sample/draft_hf/config.json \
  --block-size 16
```

The normalizer preserves DFlash/Domino exports as ``DFlashDraftModel``. For a
DSpark export it selects SGLang's ``Qwen3DSparkModel`` architecture and promotes
the Markov/confidence settings required by the SGLang loader to the top level.

Start SGLang in the first terminal:

```bash
python -m sglang.launch_server \
  --model-path ${MODEL} \
  --served-model-name ${MODEL_NAME}-${SPEC_METHOD}-overfit \
  --tp-size 1 \
  --dtype bfloat16 \
  --attention-backend triton \
  --context-length 2100 \
  --max-running-requests 1 \
  --max-total-tokens 1024 \
  --chunked-prefill-size -1 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --trust-remote-code \
  --port 32001 \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path outputs/${MODEL_NAME}-${SPEC_METHOD}-one-sample/draft_hf \
  --reasoning-parser qwen3
```

After SGLang is ready, run the validation in a second terminal:

```bash

python scripts/gates/run_dflash_chat_serving_gate.py \
  --server-url http://127.0.0.1:32001 \
  --model-path ${MODEL} \
  --served-model ${MODEL_NAME}-${SPEC_METHOD}-overfit \
  --data-path ${TRAIN_DATA_PATH} \
  --output-path outputs/${MODEL_NAME}-${SPEC_METHOD}-one-sample/serving-gate.json \
  --enable-thinking \
  --system-prompt "" \
  --block-size 16 \
  --max-tokens 16
```

The terminal prints a concise summary. The complete request, SGLang response,
and server information remain in `serving-gate.json`; add `--verbose` only when
the full result is needed in the terminal.

The validation passes only when the result contains:

```json
{
  "passed": true,
  "input_format": "training_jsonl",
  "spec_accept_length": 16.0,
  "target_prefix_match_tokens": 16,
  "generated_tokens": 16,
  "target_tokens": 59,
  "clean_block_tokens": 16,
  "errors": [],
  "result_path": ""
}
```

`spec_accept_length >= 16` and `target_prefix_match_tokens >= 16` mean that the
complete DFlash block was accepted and agrees with the target continuation.
Stop the SGLang server with `Ctrl-C` after validation.
