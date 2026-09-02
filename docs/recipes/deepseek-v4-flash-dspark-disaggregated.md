# DeepSeek-V4-Flash DSpark disaggregated training

Trains a DSpark drafter for `deepseek-ai/DeepSeek-V4-Flash-0731` from scratch:
a five-layer Qwen3-style GQA decoder at the target's width (hidden 4096, vocab
129280) with the DSpark projector over five captured target layers
(`[1, 11, 21, 31, 41]` of 43), a vanilla rank-256 Markov head, and the
confidence head. The drafter ties the target's `embed.weight` /
`head.weight`; mask token `128799` is a reserved placeholder token (the same
id DeepSeek reserves as `dspark_noise_token_id`).

The recipe fits one 8-GPU B200 node: two TP2 capture servers on four GPUs
and a four-rank FSDP trainer on the other four (global batch 128 = 4 ranks x
32 microbatches; ~4.6 s/step measured). Splitting capture and training
across two nodes only changes the endpoints and `CUDA_VISIBLE_DEVICES`.

## Required source revisions

- SpecForge with the `deepseek-v4` chat template and DSpark support.
- SGLang `v0.5.18` patched with the checked-in capture patch; v0.5.18 already
  ships DeepSeek-V4's mHC-aware DSpark capture hook
  (`set_dspark_layers_to_capture`; per-layer post-`hc_post` states averaged
  over the four hyper-connection streams):

  ```bash
  scripts/apply_sglang_spec_capture_patch.sh --target v0.5.18
  ```

The generic DFlash capture method is not equivalent for DeepSeek-V4: the
residual stream between layers is the widened mHC tensor, so the model needs
its own capture hook. `--spec-capture-method dspark` selects it.

## Chat template

DeepSeek-V4 checkpoints ship no Jinja chat template (prompts are rendered by
the repo's reference python encoder), so the registered `deepseek-v4`
template carries its own Jinja mirroring that encoder: ShareGPT-style turns
render the chat-mode scaffold `<｜Assistant｜></think>{content}` and a
populated `reasoning_content` renders the thinking-mode block. Supervision
anchors at `<｜Assistant｜>` and covers through `<｜end▁of▁sentence｜>`.

## Data

```bash
python scripts/prepare_data.py --dataset sharegpt
```

writes `cache/dataset/sharegpt_train.jsonl`, which the recipe points at.

## Capture servers (GPUs 0-3)

Start Mooncake with a global segment sized for at least one optimizer quantum
of features (128 samples x ~0.4 GiB at 8K tokens), then the patched server.
For a two-node run replace `127.0.0.1` with the capture node's routable IP.

```bash
mooncake_master \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=0.0.0.0 \
  --rpc_port=35551 \
  --http_metadata_server_port=35880 \
  --metrics_port=35903 \
  --default_kv_lease_ttl=5m
```

`--default_kv_lease_ttl=5m` is required: captured features wait in the store
while the consumer initializes and fills its optimizer window, and the default
lease is short enough to expire them first (`get_into failed (status -707)`,
LEASE_EXPIRED). Export `MC_TRANSFER_TIMEOUT=300` on both the server and the
trainer so multi-hundred-MB feature transfers don't hit the default transfer
timeout (`status -800`, TRANSFER_FAIL).

In a second process:

```bash
export MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:35551
export MOONCAKE_METADATA_SERVER=http://127.0.0.1:35880/metadata
export MOONCAKE_LOCAL_HOSTNAME=127.0.0.1
export MOONCAKE_PROTOCOL=tcp
export MC_TRANSFER_TIMEOUT=300
export MOONCAKE_GLOBAL_SEGMENT_SIZE=$((200 << 30))
export MOONCAKE_LOCAL_BUFFER_SIZE=$((1 << 30))
# flashinfer's CuTe-DSL rmsnorm/mxfp8-quantize kernels miscompile against some
# nvidia-cutlass-dsl pins (TypeError in GPUModuleOp during warmup); force the
# CUDA backends. The QUANT knob is honored via the checked-in capture patch.
export FLASHINFER_USE_CUDA_NORM=1
export FLASHINFER_USE_CUDA_QUANT=1
# One server per GPU pair; repeat with CUDA_VISIBLE_DEVICES=2,3 --port 30001.
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 30000 \
  --model-path deepseek-ai/DeepSeek-V4-Flash-0731 \
  --trust-remote-code \
  --skip-tokenizer-init \
  --tp-size 2 \
  --mem-fraction-static 0.88 \
  --context-length 8704 \
  --max-running-requests 8 \
  --chunked-prefill-size -1 \
  --max-prefill-tokens 8704 \
  --moe-runner-backend flashinfer_mxfp4 \
  --enable-spec-capture \
  --spec-capture-method dspark \
  --spec-capture-aux-layer-ids 1 11 21 31 41
```

`--moe-runner-backend flashinfer_mxfp4` is required on B200: the checkpoint's
routed experts are fp4 (`expert_dtype: "fp4"`) and the default MoE path cannot
run them. Capture is prefill-only, so `--disable-cuda-graph` may be added to
cut startup time without hurting capture throughput.

## Trainer (GPUs 4-7)

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 specforge train \
  -c examples/configs/online/disaggregated/external/deepseek-v4-flash-dspark-disaggregated.yaml
```

The single supervisor starts the CPU producer and the four-rank consumer.
Do not put Hugging Face or W&B credentials in YAML; supply `HF_TOKEN` and
`WANDB_API_KEY` through the process environment.

## Fresh attempts

A fresh attempt requires fresh `control_dir`/`consumer_state_dir` (delete the
run's `outputs/` directory) **and a fresh Mooncake namespace whenever the
capture server was restarted**: the producer dedups against keys already
registered in the Mooncake master, and keys whose replicas lived in a dead
server's segment poison the consumer with `get_into failed` errors. Either
restart `mooncake_master` together with the server, or override the namespace
per attempt:

```bash
specforge train -c examples/configs/online/disaggregated/external/deepseek-v4-flash-dspark-disaggregated.yaml \
  "deployment.disaggregated.store_id=<unique-attempt-id>"
```
