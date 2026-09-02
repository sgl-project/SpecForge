# DeepSeek-V4-Flash DSpark disaggregated training

Trains the DSpark drafter in `configs/deepseek-v4-flash-dspark.json` (a
five-layer Qwen3-style GQA decoder at the target's width) for
`deepseek-ai/DeepSeek-V4-Flash-0731`, from scratch on ShareGPT. The recipe
fits one 8-GPU B200 node: two TP2 capture servers on GPUs 0-3 and a four-rank
trainer on GPUs 4-7 (global batch 128; ~4.6 s/step measured). A two-node
split only changes the endpoints and `CUDA_VISIBLE_DEVICES`.

The flow is the standard disaggregated setup
(`docs/basic_usage/disaggregated_training.md`); what is DeepSeek-V4-specific:

- **Capture hook**: SGLang `v0.5.18` patched with
  `scripts/apply_sglang_spec_capture_patch.sh --target v0.5.18`, launched
  with `--spec-capture-method dspark`. The generic DFlash capture is not
  equivalent for V4: the inter-layer residual is the widened mHC tensor, so
  capture must use the model's own `set_dspark_layers_to_capture` hook.
- **Chat template**: the checkpoint ships no Jinja chat template, so the
  recipe uses the bundled `deepseek-v4` registry template.
- **B200 MoE**: `--moe-runner-backend flashinfer_mxfp4` (the routed experts
  are fp4; the default MoE path cannot run them), with
  `FLASHINFER_USE_CUDA_NORM=1 FLASHINFER_USE_CUDA_QUANT=1` exported to work
  around CuTe-DSL kernel miscompiles on some nvidia-cutlass-dsl pins.

## Data

`python scripts/prepare_data.py --dataset sharegpt` writes
`cache/dataset/sharegpt_train.jsonl`, which the recipe points at.

## Capture servers (GPUs 0-3)

```bash
mooncake_master \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=0.0.0.0 \
  --rpc_port=35551 \
  --http_metadata_server_port=35880 \
  --metrics_port=35903 \
  --default_kv_lease_ttl=5m
```

`--default_kv_lease_ttl=5m` and `MC_TRANSFER_TIMEOUT=300` (on server and
trainer) are required for multi-hundred-MB feature objects; the Kimi K3
runbook explains the failure modes. Then one server per GPU pair (repeat
with `CUDA_VISIBLE_DEVICES=2,3 --port 30001`; for a two-node run replace
`127.0.0.1` with the capture node's routable IP):

```bash
export MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:35551
export MOONCAKE_METADATA_SERVER=http://127.0.0.1:35880/metadata
export MOONCAKE_LOCAL_HOSTNAME=127.0.0.1
export MOONCAKE_PROTOCOL=tcp
export MC_TRANSFER_TIMEOUT=300
export MOONCAKE_GLOBAL_SEGMENT_SIZE=$((200 << 30))
export MOONCAKE_LOCAL_BUFFER_SIZE=$((1 << 30))
export FLASHINFER_USE_CUDA_NORM=1
export FLASHINFER_USE_CUDA_QUANT=1
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

## Trainer (GPUs 4-7)

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 specforge train \
  -c examples/configs/online/disaggregated/external/deepseek-v4-flash-dspark-disaggregated.yaml
```

Supply `HF_TOKEN` and `WANDB_API_KEY` through the environment, not YAML.

## MoE-FFN arm (ablation)

`examples/configs/online/disaggregated/external/deepseek-v4-flash-dspark-moe-disaggregated.yaml`
is the same recipe with `configs/deepseek-v4-flash-dspark-moe.json`: the
five-layer GQA decoder keeps its attention, and each layer's dense MLP becomes
the target's MoE (`moe_preset: deepseek_v4`: sqrt-softplus scores,
aux-loss-free top-k with the sign-controlled selection bias, combine weights
renormalized and scaled by 1.5, one ungated shared expert, SwiGLU clamped at
10). Sizes are per run: 64 routed experts, top-6, width 2048, so the activated
FFN width (6 x 2048 + 2048 shared) matches the dense 12288 at ~10x the FFN
parameters. Run it against the dense recipe with identical hparams; the two
YAMLs differ only in the draft JSON and run names. The capture servers are
shared by both arms unchanged.

Training-only knobs live under the draft JSON's `dflash_config`:
`moe_bias_update_rate` (0.001, the balancing controller's step) and
`moe_dispatch` (`grouped_mm` runs the experts as grouped GEMMs with no host
sync; `sorted_loop` is the portable fallback). The trainer logs `moe/*` load
metrics (max/min load ratios, unused-expert fraction, balancing-bias
magnitude) alongside the usual scalars. Checkpoints keep the official
per-expert naming (`layers.N.mlp.experts.{i}.w{1,2,3}.weight`,
`layers.N.mlp.gate.bias`, `layers.N.mlp.shared_experts.w{1,2,3}.weight`), so
exports load into SGLang's DeepSeek-V4 MoE unchanged.

## Fresh attempts

Delete the run's `outputs/` directory and, whenever a capture server was
restarted, use a fresh Mooncake namespace (restart `mooncake_master`, or
override `deployment.disaggregated.store_id=<unique-attempt-id>`): the
producer dedups against keys already registered in the master, and keys whose
replicas lived in a dead server's segment poison the consumer with
`get_into failed` errors.
