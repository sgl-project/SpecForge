# DeepSeek-V4-Flash DSpark two-node H200 disaggregated training

Two-node variant of the
[single-node B200 recipe](deepseek-v4-flash-dspark-disaggregated.md), verified
on two 8xH200 nodes (example addresses 192.0.2.10 / 192.0.2.20 below):

- **Capture node (192.0.2.10)**: `mooncake_master` + four TP2 capture
  servers on GPUs 0-7. TP prefill scaling is sublinear, so 4xTP2 out-produces
  2xTP4 or 1xTP8; measured ~10+ samples/s sustained (8K tokens each), which
  keeps the producer pinned at the in-flight high watermark (i.e. capture is
  not the bottleneck).
- **Trainer node (192.0.2.20)**: `specforge train` single supervisor = CPU
  producer + eight-rank FSDP consumer on GPUs 0-7. Global batch stays 128
  (8 ranks x 16 accumulation vs the B200 recipe's 4 x 32).

Config: `examples/configs/online/disaggregated/external/deepseek-v4-flash-dspark-2node-h200.yaml`.
Measured ~4-4.5 s per optimizer step (~800+ steps/hour) — on par with the
verified single-node B200 recipe despite Hopper's slower MoE path, because
capture no longer shares the node with training.

## H200 (SM90) differences vs the B200 recipe

- `--moe-runner-backend marlin`: the checkpoint's fp4 routed experts run
  through the W4A16 Marlin MoE path. The B200 recipe's `flashinfer_mxfp4`
  TRT-LLM path is SM100-only (on SM90 that flag selects a cutlass W4A16
  fallback; the cookbook's command generator picks Marlin for Hopper).
- `--watchdog-timeout 3600` on the capture servers: the first marlin MoE
  call JIT-compiles `moe_wna16_marlin` under a file lock; on a cold
  `~/.cache/sglang` this exceeds the default 300 s watchdog and the server
  kills itself mid-compile. Later starts hit the cache and are fast.
- The `FLASHINFER_USE_CUDA_NORM/QUANT` workarounds from the B200 runbook are
  Blackwell-only (CuTe-DSL kernels never run on SM90) and can be dropped.

## SGLang 0.5.18 capture patch

`patches/sglang/v0.5.18/spec-capture.patch` is the port of the v0.5.14
spec-capture patch to SGLang v0.5.18
(`scripts/apply_sglang_spec_capture_patch.sh --target v0.5.18`). Since 0.5.14,
upstream absorbed part of the original patch (the
`configure_aux_hidden_state_capture` dspark/dflash dispatch and DeepSeek-V4's
mHC-aware `set_dspark_layers_to_capture` hook), so the port carries only the
still-missing pieces:

- the `spec_capture` request field and Mooncake sink (`spec_capture_sink.py`),
- scheduler / batch-result-processor integration — note the sink now fires
  after `update_finish_state()` because 0.5.18 moved the prefill hidden-state
  append *before* finish handling (a request would otherwise never be seen
  as finished and its response would hang),
- the `last_hidden_states` channel and the aux-wins-over-prenorm fix in
  `LogitsProcessor` (V4 always passes `hidden_states_before_norm`, which
  otherwise silently replaces the aux concat),
- `enable-spec-capture` server args + model_runner wiring, including routing
  `is_dspark` from `--spec-capture-method` (a capture-only server has no
  speculative algorithm, so the upstream dispatch would otherwise demand the
  nonexistent `set_dflash_layers_to_capture` on V4),
- the mHC prenorm prewarm fallback for `--chunked-prefill-size -1`.

## Launch

Capture node — mooncake master (note `--default_kv_lease_ttl=5m` and
`MC_TRANSFER_TIMEOUT=300`, same operational contract as the B200 runbook),
then one server per GPU pair (ports 30000-30003):

```bash
mooncake_master --enable_http_metadata_server=true \
  --http_metadata_server_host=0.0.0.0 --rpc_port=35551 \
  --http_metadata_server_port=35880 --metrics_port=35903 \
  --default_kv_lease_ttl=5m
```

Per server instance `i` (GPUs `2i,2i+1`, port `30000+i`): the B200 runbook's
server command with `--moe-runner-backend marlin --watchdog-timeout 3600`, and
`MOONCAKE_LOCAL_HOSTNAME` set to the capture node's routable IP (the trainer
node fetches features out of these servers' Mooncake segments; `127.0.0.1`
would register an unreachable data-plane endpoint).

Trainer node:

```bash
export MC_TRANSFER_TIMEOUT=300
specforge train -c examples/configs/online/disaggregated/external/deepseek-v4-flash-dspark-2node-h200.yaml
```

The config ships `tracking.wandb_offline: true` (the trainer node may not
carry wandb credentials); sync the offline run from any credentialed host —
the wandb dir is on shared storage:

```bash
wandb sync outputs/deepseek-v4-flash-dspark-2node-h200/wandb/wandb/offline-run-*
```

Re-run the sync periodically for a near-live dashboard; the trailing
`transactionlog: unexpected EOF` on a live run is benign.

## Fresh attempts

Same contract as the single-node runbook: wipe the run's `outputs/` directory
(the supervisor refuses a control dir with artifacts from a prior attempt)
and use a fresh `store_id` whenever a capture server restarts.
