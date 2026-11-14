# Training recipes

All bundled recipes use canonical draft configuration. The top-level `draft`
accepts an inline mapping or a path to a YAML/JSON file. Files in
`specforge/config/draft/presets/` are ordinary configurations; their names do
not select model code.

```yaml
draft: specforge/config/draft/presets/qwen3-8b-dflash.yaml
model:
  target_model_path: Qwen/Qwen3-8B
training:
  objective_options:
    num_anchors: 512
  loss:
    fused: true
    terms:
      token:
        kind: ce
    scheduling:
      chunk_size: 128
      checkpoint: true
```

The architecture selects its default objective. Set `training.objective` to
choose another registered objective compatible with that architecture.
Algorithm sampling/rollout options belong in `objective_options`; numerical
losses, coefficients and execution settings belong in `loss`.

Use `initialization.checkpoint` for weight import and `resume` for a canonical
training checkpoint. These sources already record the draft configuration.
Historical checkpoint weights remain importable through the legacy codec;
their optimizer states are not portable across model layout changes.

Offline EAGLE supports `attention_backend: usp` with Ulysses/Ring sequence
parallelism. Online consumers use data parallelism; configure target TP on
the capture server.

## Configuration reference

### Top-level fields

| Field | Default | What to write |
| --- | --- | --- |
| `run_id` | `specforge-run` | Stable identifier for the run. It names checkpoints and is also the default disaggregated store namespace. Use a new value for a fresh attempt. |
| `output_dir` | `./output` | Shared checkpoint, profiler, and tracker output directory. Every trainer rank must resolve it to the same location. |
| `draft` | `null` | Canonical draft mapping or YAML/JSON path, including preset files. Its architecture selects the default objective. |
| `initialization` | `null` | Canonical weights-only loading: `checkpoint`, or `weights_from` (`checkpoint`, `mapping`, `allow_new`) with optional disjoint `from_target` copies. See [canonical training](../../docs/draft_training.md). |
| `resume` | `null` | Canonical full training resume; restores structure from the checkpoint. A canonical checkpoint given in `training.resume_from` resumes the same way; if both fields are set they must agree. Excludes initialization. |

### `model`: target, draft, and capture backend

| Field | Default | What to write |
| --- | --- | --- |
| `model.target_model_path` | required | Local target directory or Hugging Face repository ID. |
| `model.target_backend` | `sglang` | `sglang` is the only accepted value; retired `hf`/`custom` names fail at config load. Offline feature consumers do not instantiate a target inference backend. |
| `model.input_modality` | `text` | The provider modality. The unified runtime supports text only; VLM modalities such as `qwen2_5_vl` are rejected. |
| `model.shard_target_output` | `false` | Retained for config migration; leave it false on the online disaggregated path. |
| `model.trust_remote_code` | `false` | Enable only for model repositories that require custom loading code. |
| `model.use_fused_conv` | `false` | Opt canonical grouped convolution into its existing CUDA/ROCm fused kernel; independent of Liger. |
| `model.use_liger_kernel` | `null` | Use PyTorch kernels by default. Explicit `true` selects Liger RMSNorm and SiLU-gated SwiGLU on CUDA/ROCm; unsupported devices or activations fail before training. The choice is recorded for resume and does not change checkpoint weights. |
| `model.embedding_key` | `model.embed_tokens.weight` | Target checkpoint key copied into or used by the draft embedding. |
| `model.lm_head_key` | `lm_head.weight` | Target checkpoint key used for the frozen output head. |
| `model.vocab_mapping_path` | `""` | Target-to-draft vocabulary mapping. EAGLE3 disaggregated runs require an explicit shared file. |
| `model.load_target_embedding` | `true` | Copy the frozen target embedding into a fresh draft when supported. |
| `model.torch_dtype` | `bfloat16` | `bfloat16`, `float16`, or `float32`. |
| `model.cache_dir` | `null` | Model/tokenizer download cache. This is distinct from `data.cache_dir`. |
| `model.tokenizer_pad_token_id` | `null` | Explicit non-negative tokenizer pad ID. Use it for released tokenizers that omit padding metadata. |
| `model.sglang_attention_backend` | `flashinfer` | SGLang attention implementation for an in-process or managed capture server. |
| `model.sglang_mem_fraction_static` | `0.4` | SGLang static-memory fraction in `(0, 1]`; inherited by managed capture servers unless they override it. |
| `model.sglang_disable_radix_cache` | `true` | Preserve the historical managed-capture behavior. Set `false` for hybrid targets such as Inkling that require the radix tree. Unique per-attempt cache namespaces still force complete capture prefills. |
| `model.sglang_context_length` | `null` | Positive explicit context limit. Managed capture requires at least `data.max_length + 7`; omitting it derives that value. |
| `model.sglang_enable_nccl_nvls` | `false` | Pass the matching SGLang NCCL NVLS optimization flag. |
| `model.sglang_enable_symm_mem` | `false` | Pass the matching SGLang symmetric-memory flag. |
| `model.sglang_enable_torch_compile` | `false` | Enable the SGLang torch-compile path. |
| `model.sglang_enable_dp_attention` | `false` | Enable SGLang DP attention where supported. Managed-local capture currently rejects it. |
| `model.sglang_enable_dp_lm_head` | `false` | Enable SGLang DP LM head where supported. Managed-local capture currently rejects it. |
| `model.sglang_ep_size` | `1` | SGLang expert-parallel size; it must divide and not exceed every managed capture server's `tp_size`. |
| `model.sglang_max_running_requests` | `null` | Positive SGLang request-concurrency limit. |
| `model.sglang_max_total_tokens` | `null` | Positive SGLang token-pool limit. |
| `model.sglang_dp_size` | `null` | Optional SGLang data-parallel size. |
| `model.sglang_moe_a2a_backend` | `null` | Optional SGLang MoE all-to-all backend name. |
| `model.sglang_moe_runner_backend` | `null` | Optional SGLang MoE runner backend name. |
| `model.sglang_page_size` | `null` | Optional positive SGLang KV-cache page size. |
| `model.sglang_quantization` | `null` | Optional SGLang target quantization mode. |
| `model.sglang_fp4_gemm_runner_backend` | `null` | Optional SGLang FP4 GEMM runner backend, passed as `--fp4-gemm-backend`. |
| `model.sglang_mamba_radix_cache_strategy` | `null` | Optional hybrid Mamba/radix cache strategy. |
| `model.sglang_max_mamba_cache_size` | `null` | Optional positive Mamba cache size. |
| `model.sglang_swa_full_tokens_ratio` | `null` | Optional SGLang sliding-window full-token ratio in `(0, 1]`. |
| `model.sglang_mamba_full_memory_ratio` | `null` | Optional SGLang Mamba full-memory ratio in `(0, 1]`. |
| `model.sglang_max_prefill_tokens` | `null` | Optional positive token budget of one SGLang prefill batch (`--max-prefill-tokens`). |
| `model.sglang_linear_attn_prefill_backend` | `null` | Optional linear-attention (GDN/KDA) prefill kernel backend for hybrid targets (`--linear-attn-prefill-backend`). |
| `model.sglang_fp8_gemm_backend` | `null` | Optional blockwise FP8 GEMM runner (`--fp8-gemm-backend`). |
| `model.sglang_disable_cuda_graph` | `false` | Pass `--disable-cuda-graph`, which disables SGLang's decode and prefill CUDA graphs. |
| `model.sglang_extra_args` | `[]` | Further SGLang CLI tokens appended to every managed-local capture server, one flag or value per list item (`["--prefill-max-requests", "8"]`). The list must start with a `--flag`. Flags the launcher owns (`--model-path`, `--port`, `--host`, `--tp-size`, `--dtype`, `--trust-remote-code`, `--skip-tokenizer-init`, `--chunked-prefill-size`, `--context-length`, the auxiliary hidden-state capture flags) and flags another `model.sglang_*` field already renders are rejected, including their SGLang aliases and abbreviations; so are `--api-key` (the capture adapter sends no key), `--enable-dp-attention`/`--enable-dp-lm-head` (unsupported by managed-local capture) and `--config`. External servers take their flags from their own launch command. |

### `data`: choose exactly one training source

| Field | Default | What to write |
| --- | --- | --- |
| `data.train_data_path` | `""` | Raw conversation/preformatted JSON or JSONL sent to the online capture producer. |
| `data.prompts_path` | `""` | Pre-tokenized online JSONL with `input_ids` and `loss_mask`. |
| `data.hidden_states_path` | `""` | Directory of precomputed offline feature `.ckpt` files. Selecting it makes the run offline. |
| `data.eval_data_path` | `""` | Reserved migration field. Online evaluation is unsupported; leave it empty. |
| `data.eval_hidden_states_path` | `""` | Offline evaluation features; configure them together with a positive `training.eval_interval`. |
| `data.max_length` | `2048` | Maximum token length used by preparation, capture, and training. |
| `data.chat_template` | `llama3` | Template name used to format conversations and locate assistant loss spans. |
| `data.is_preformatted` | `false` | Treat each record's text as already formatted by `chat_template`. |
| `data.train_only_last_turn` | `false` | Restrict the loss mask to the final assistant turn. |
| `data.build_dataset_num_proc` | `8` | Positive CPU process count for dataset preprocessing. |
| `data.dataloader_num_workers` | `null` | Ordered feature-loader workers. `null` preserves strategy defaults: EAGLE/P-EAGLE 4, DFlash-family 8; canonical runs use 4. Use 0 for synchronous loading. |
| `data.cache_dir` | `./cache` | Prepared dataset and derived vocabulary-mapping cache. |
| `data.cache_key` | `null` | Optional explicit namespace when multiple preparations share the same source. |
| `data.max_prompts` | `null` | Optional non-negative prompt cap, useful for smoke tests. |

### `training`: optimization, strategy, and parallelism

| Field | Default | What to write |
| --- | --- | --- |
| `training.objective` | `null` | Canonical objective registration; omission uses the architecture's default. |
| `training.objective_options` | `{}` | Options for the selected canonical objective, such as `num_anchors` or `ttt_length`. Unknown options fail validation. |
| `training.frozen` | `[]` | Canonical module or parameter paths excluded from training; persisted in the resume contract. |
| `training.loss` | `{}` | Named loss terms, for example `{terms: {token: {kind: kl}}}`; `{}` keeps the objective's default loss. A loss that needs teacher logits also adds `target_last_hidden_states` to the required features. Persisted in the resume contract. |
| `training.weighting` | `null` | Canonical position weighting, for example `{kind: exponential, gamma: 7.0}`; `null` keeps the objective's default. Persisted in the resume contract. |
| `training.num_epochs` | `1` | Positive passes over a finite source. |
| `training.max_steps` | `null` | Positive hard stop in optimizer steps. If it is set while `total_steps` is omitted, it is also the fallback schedule horizon. |
| `training.total_steps` | `null` | Positive optimizer/loss schedule horizon; it does not itself stop an online stream. A finite online disaggregated run may omit both fields: the producer publishes the exact horizon derived from prepared prompts, epochs, DP size, batch size, and accumulation. |
| `training.batch_size` | `1` | Per-rank microbatch size. P-EAGLE and USP require 1. |
| `training.accumulation_steps` | `1` | Positive microbatches per optimizer update. |
| `training.fsdp_sharding` | `SHARD_GRAD_OP` | Trainer FSDP mode: `SHARD_GRAD_OP`, `FULL_SHARD`, or `NO_SHARD`. |
| `training.ddp_bucket_cap_mb` | unset | Optional DDP bucket target in MiB for `NO_SHARD`; unset keeps PyTorch's default. Benchmark changes because larger buckets trade launch overhead for later overlap. |
| `training.learning_rate` | `1e-4` | Positive peak learning rate. |
| `training.lr_scheduler` | `cosine` | Learning-rate schedule after warmup: `cosine` or `constant`. |
| `training.warmup_ratio` | `0.015` | Fraction in `[0, 1]` used for scheduler warmup. |
| `training.max_grad_norm` | `0.5` | Positive gradient-clipping norm. |
| `training.optimizer_cpu_offload` | `false` | Keep the optimizer's FP32 master parameters and Adam state on CPU. |
| `training.attention_backend` | `flex_attention` | `eager`, `sdpa`, `flex_attention`, `fa`, or `usp`; the selected strategy must support it. Canonical objectives accept `eager` (EAGLE3 also `sdpa`), so a canonical YAML must set it. |
| `training.tp_size` | `1` | Online disaggregated consumers must keep it at 1; configure target TP on capture servers. Offline non-USP ranks consume disjoint data. |
| `training.sp_ulysses_size` | `1` | Ulysses sequence-parallel factor for offline EAGLE3 USP. |
| `training.sp_ring_size` | `1` | Ring sequence-parallel factor for offline EAGLE3 USP. |
| `training.dist_timeout` | `10` | Positive distributed-operation timeout in minutes. |
| `training.save_interval` | `0` | Save every N optimizer steps; 0 disables periodic saves. A final checkpoint is still written. |
| `training.eval_interval` | `0` | Evaluate every N optimizer steps; 0 disables evaluation. |
| `training.log_interval` | `50` | Positive optimizer-step logging interval. |
| `training.max_checkpoints` | `0` | Keep the newest N checkpoints; 0 keeps all. |
| `training.resume_from` | `null` | Alias for `resume`. Restores a canonical full training checkpoint and excludes `initialization`. If both aliases are set they must agree. |
| `training.role` | `all` | Use `all` for offline colocated training; disaggregated entrypoints select `auto`, `producer`, or `consumer`. |
| `training.seed` | `42` | Run and per-rank RNG seed. |
| `training.prompt_seed` | `null` | Optional online prompt-shuffle seed. `null` preserves the historical behavior of using `training.seed`. |
| Strategy | Fields and defaults |
| --- | --- |
| EAGLE3 | `training.ttt_length` (`7`); choose `soft_ce` or an LK formula under `loss.terms.token`. |
| DFlash / DFlash2 / Domino | `training.num_anchors` (`512`); configure CE/LK under `loss.terms`, and exponential/D-PACE weighting under `training.weighting`. Selector warmup/ramp and Domino schedules remain algorithm options. |
| DSpark | `loss.terms.token` defaults to CE/L1 weights 0.1/0.9; `loss.terms.confidence.coefficient` defaults to 1. Sampling stays in `num_anchors`. |
| P-EAGLE | `training.num_depths` (`8`), `training.down_sample_ratio` (`0.8`), `training.down_sample_ratio_min` (`0.2`), `training.norm_before_residual` (`null`). |
| MTP | `training.loss.scheduling.chunk_size` controls projection/loss chunks; `0` disables splitting. |

### `deployment`: process and service topology

| Field | Default | What to write |
| --- | --- | --- |
| `deployment.mode` | `local_colocated` | Set `local_colocated` or `disaggregated` explicitly in every new recipe. |
| `deployment.trainer` | default object | Trainer process topology described by the fields below. |
| `deployment.disaggregated` | `null` | Required object only when `deployment.mode: disaggregated`. |
| `deployment.trainer.nnodes` | `1` | Number of trainer nodes. |
| `deployment.trainer.nproc_per_node` | `1` | Trainer processes per node; the single CLI self-launches local ranks. |
| `deployment.trainer.node_rank` | `null` | Node-local rank. Shared multi-node YAML normally omits it and passes `--node-rank`. |
| `deployment.trainer.master_addr` | `null` | Rendezvous address; required when `nnodes > 1`. |
| `deployment.trainer.master_port` | `29500` | Rendezvous port. |
| Field | Default | What to write |
| --- | --- | --- |
| `deployment.disaggregated.control_dir` | required | Fresh attempt-scoped directory for refs/manifest and lifecycle markers. Shared by default; with `inbox_server_url`, only producer and consumer rank 0 must share it. |
| `deployment.disaggregated.backend` | required | `mooncake` or `shared_dir`. Online disaggregated runs require Mooncake. |
| `deployment.disaggregated.consumer_state_dir` | `null` | Node-local rank-0 SQLite/WAL root. Required for multi-node online consumers; their rank inboxes remain under shared `control_dir`. |
| `deployment.disaggregated.inbox_server_url` | `null` | Optional private `http://host:port` rank-0 relay for tensor-free inbox refs when remote trainer ranks cannot share `control_dir`. Online multi-node only; no credentials, path, query, TLS, or built-in authentication. |
| `deployment.disaggregated.store_root` | `null` | Shared feature directory; required when `backend: shared_dir`. |
| `deployment.disaggregated.store_id` | `null` | Feature-store namespace; defaults to `run_id`. |
| `deployment.disaggregated.server_urls` | `[]` | External SGLang capture endpoints (running the capture plugin). One rollout worker is created per entry. Do not set with `managed_local`. |
| `deployment.disaggregated.server_capture` | `auto` | SGLang capture uses the `specforge-sglang-capture` plugin; allowed values are `auto` and `plugin`. `auto` honors `DISAGG_SERVER_CAPTURE=plugin`, otherwise managed-local planning checks each server interpreter for the plugin and its required SGLang APIs. External capture detects `spec_capture_backend: plugin` in `/server_info`. Missing support fails with installation instructions; source-patch servers are unsupported. See [plugin setup](../../plugins/sglang-spec-capture/README.md). |
| `deployment.disaggregated.mooncake_metadata_server` | `null` | External Mooncake metadata URL. |
| `deployment.disaggregated.mooncake_master_server_addr` | `null` | External Mooncake RPC `host:port`. |
| `deployment.disaggregated.mooncake_local_hostname` | `null` | Node-local Mooncake transfer hostname; usually supplied through the environment. |
| `deployment.disaggregated.mooncake_protocol` | `null` | External transfer protocol, `tcp` or `rdma`. An exported `MOONCAKE_PROTOCOL` overrides this value; with neither set, `tcp`: SpecForge cannot see an external server's protocol, and every server and trainer of one store must use the same one. |
| `deployment.disaggregated.mooncake_rdma_devices` | `null` | External Mooncake RDMA-device selection; an exported `MOONCAKE_RDMA_DEVICES` overrides it. Under `rdma`, a producer or trainer checks every listed device, and unset selects the usable HCAs of its node; external capture servers must list theirs in `MOONCAKE_RDMA_DEVICES`. |
| `deployment.disaggregated.producer_segment_size` | `null` | Positive allocation owned by an offline Mooncake producer. Online capture is server-owned and forces client segments to zero. |
| `deployment.disaggregated.client_buffer_size` | `268435456` | Per-role Mooncake client buffer in bytes. For `receive_buffers: cuda`, size for the largest concurrently fetched tensors; an 8k-token Qwen3.8-27B feature can exceed this default. Use e.g. `2147483648` and increase with fetch concurrency. |
| `deployment.disaggregated.receive_buffers` | `pinned` | Consumer receive buffers for feature reads: `pinned` (pooled page-locked host buffers, side-stream device copy), `pageable` (fresh host tensor per feature), or `cuda` (pooled device buffers; Mooncake 0.3.x stages RDMA reads through its client buffer). |
| `deployment.disaggregated.receive_pool_bytes` | `8589934592` | Retained receive-pool budget per rank (`pinned`/`cuda`), excluding output copies and concurrent overflow. Failed-transfer buffers remain quarantined for the store lifetime. |
| `deployment.disaggregated.idle_timeout_s` | `null` | Positive consumer idle timeout. |
| `deployment.disaggregated.peer_wait_timeout_s` | `null` | Optional positive producer/consumer peer-completion timeout. Unset is unbounded; expiration fails the attempt. |
| `deployment.disaggregated.producer_hold_s` | `null` | Optional positive offline producer retention timeout. Unset is unbounded; expiration fails the attempt. |
| `deployment.disaggregated.async_ack` | `true` | Online consumer: overlap each optimizer boundary's durable ack (ledger commit, feature removes, DP ack collectives on a dedicated Gloo group) with the next step on a background thread. The durable marker lags by at most one optimizer step and is flushed before every checkpoint, eval, and fit end. Exported as `DISAGG_ASYNC_ACK` (`0` disables; an explicit environment value wins). Multi-node trainers need Gloo TCP reachability between ranks (set `GLOO_SOCKET_IFNAME` when the hostname does not resolve to a routable address); disabling it on any rank keeps every rank's acks synchronous on the default process group and creates no Gloo group. |
| `deployment.disaggregated.shutdown_grace_s` | `30.0` | SIGTERM-to-SIGKILL window for a plain supervisor teardown; must cover worker cleanup (Mooncake drains, checkpoint flush, failure sentinels). `managed_local` stacks use `managed_local.shutdown_grace_s`. |
| `deployment.disaggregated.managed_local` | `null` | Optional owned single-node Mooncake + capture-server stack described below. |
| Path | Lifetime and visibility |
| --- | --- |
| `output_dir` | Shared durable checkpoints and run outputs. |
| `deployment.disaggregated.control_dir` | Shared, fresh attempt control state. |
| `deployment.disaggregated.consumer_state_dir` | Fresh node-local online-consumer ledger; required for multi-node runs. |
| `deployment.disaggregated.store_root` | Shared offline feature payloads for `shared_dir`. |
| Field | Default | What to write |
| --- | --- | --- |
| `deployment.disaggregated.managed_local.trainer_cuda_visible_devices` | required | One device token per `nproc_per_node`; trainer and capture devices must not overlap. |
| `deployment.disaggregated.managed_local.mooncake` | default object | Owned loopback Mooncake configuration described by the nested fields below. |
| `deployment.disaggregated.managed_local.capture_servers` | required | One or more owned SGLang server running the capture plugin definitions. |
| `deployment.disaggregated.managed_local.shutdown_grace_s` | `30` | Positive graceful process-group shutdown window. |
| `deployment.disaggregated.managed_local.mooncake.rpc_port` | `35551` | Owned Mooncake RPC port. |
| `deployment.disaggregated.managed_local.mooncake.metadata_port` | `35880` | Owned metadata HTTP port. |
| `deployment.disaggregated.managed_local.mooncake.metrics_port` | `35903` | Owned metrics port. |
| `deployment.disaggregated.managed_local.mooncake.local_hostname` | `127.0.0.1` | Local transfer hostname. |
| `deployment.disaggregated.managed_local.mooncake.protocol` | `null` | `tcp`, `rdma`, or `nvlink`. Unset resolves to `rdma`, or to `tcp` on Ascend hosts, once when the plan is built. `tcp` opts out of RDMA. `nvlink` keeps captures in server GPU arenas and requires `arena_bytes` and `receive_buffers: cuda`. |
| `deployment.disaggregated.managed_local.mooncake.rdma_devices` | `null` | Comma-separated HCAs for `rdma`. Unset selects every usable HCA of one link layer (InfiniBand before RoCE); a listed device that is not usable is an error. |
| `deployment.disaggregated.managed_local.mooncake.global_segment_size_bytes` | `34359738368` | Owned global segment size. |
| `deployment.disaggregated.managed_local.mooncake.arena_bytes` | `null` | Positive HBM capacity in bytes reserved by each capture server for capture objects. Required only with `protocol: nvlink`; omit for other protocols. Leave room for the arena when setting `mem_fraction_static`, and set `runtime.resident_high_watermark_bytes` no higher than the total arena capacity across capture servers. |
| `deployment.disaggregated.managed_local.mooncake.local_buffer_size_bytes` | `1073741824` | Owned local client buffer. |
| `deployment.disaggregated.managed_local.mooncake.startup_timeout_s` | `60` | Positive Mooncake readiness timeout. |
| `deployment.disaggregated.managed_local.mooncake.probe_timeout_s` | `5` | Positive, finite budget for one HTTP + TCP readiness probe, capped by the remaining startup timeout. |
| `deployment.disaggregated.managed_local.mooncake.default_kv_lease_ttl_ms` | `500` | Master key-lease TTL (ms) forwarded to `mooncake_master --default_kv_lease_ttl`. Kept below the consumer's teardown drain window so managed_local shuts down cleanly; set `null` to inherit Mooncake's stock default. |
| `deployment.disaggregated.managed_local.capture_servers[].port` | required | Unique capture HTTP port. |
| `deployment.disaggregated.managed_local.capture_servers[].python_executable` | `null` | Python executable for a separate SGLang environment; defaults to the trainer's Python. |
| `deployment.disaggregated.managed_local.capture_servers[].cuda_visible_devices` | required | Device tokens for this server. Their count must equal its `tp_size`. |
| `deployment.disaggregated.managed_local.capture_servers[].gpu_put` | `null` | Publish from CUDA memory automatically when `mooncake.protocol: rdma` is set explicitly; with the protocol unset, `null` keeps host publication. Set `false` to use host publication or `true` to require GPU publication; `true` requires the stack to resolve to RDMA or NVLink. RDMA GPU publication needs GPUDirect RDMA (`nvidia_peermem` or DMA-BUF support). NVLink requires GPU publication and rejects `false`. |
| `deployment.disaggregated.managed_local.capture_servers[].tp_size` | `1` | Target-model tensor parallelism for this server. |
| `deployment.disaggregated.managed_local.capture_servers[].mem_fraction_static` | `null` | Optional SGLang static-memory override in `(0, 1]`; otherwise inherit `model.sglang_mem_fraction_static`. |
| `deployment.disaggregated.managed_local.capture_servers[].attention_backend` | `null` | Server-specific override; otherwise inherit `model.sglang_attention_backend`. |
| `deployment.disaggregated.managed_local.capture_servers[].startup_timeout_s` | `1800` | Positive server readiness timeout. |
| `deployment.disaggregated.managed_local.capture_servers[].extra_args` | `[]` | SGLang CLI tokens for this server only, appended after `model.sglang_extra_args` under the same rules. A flag may appear once per command line, so set a per-server flag here and not in the global list. |
| `deployment.disaggregated.managed_local.capture_servers[].env` | `{}` | Extra environment for this server process, for example the capture plugin's `SGLANG_SPEC_CAPTURE_TIMING` or `SGLANG_SPEC_CAPTURE_MAX_PENDING_BATCHES`. Keys the launcher sets are rejected: `MOONCAKE_*`, `DISAGG_*`, device-visibility variables, `SGLANG_SPEC_CAPTURE_GPU_PUT` (use `gpu_put`) and `FLASHINFER_DISABLE_VERSION_CHECK`. |
| `deployment.disaggregated.managed_local.capture_servers[].probe_timeout_s` | `5` | Positive, finite HTTP health-probe timeout, capped by the remaining startup timeout. SGLang's generation-based `/health` waits at least one second; allow headroom instead of setting this to one second. |

### `runtime`: streaming backpressure

| Field | Default | What to write |
| --- | --- | --- |
| `runtime.producer_lease` | `8` | Prompts leased to a rollout worker at once. |
| `runtime.consumer_dispatch` | `round_robin` | Use `cost_balanced` within each optimizer window, or `two_window_bucketed` / `two_window_joint_balanced` across two windows. Two-window policies can change sample grouping and require both `in_flight_high_watermark` and `in_flight_low_watermark` to cover two global optimizer windows. |
| `runtime.producer_concurrency` | `1` | Concurrent capture calls maintained by each server's logical producer. Increase to keep ingress full without duplicating producers. |
| `runtime.in_flight_high_watermark` | `256` | Pause production at this many committed, unacknowledged refs. |
| `runtime.in_flight_low_watermark` | `192` | Resume production at or below this count; it cannot exceed the high watermark. |
| `runtime.resident_high_watermark_bytes` | `null` | Optional byte-level pause threshold. |
| `runtime.resident_low_watermark_bytes` | `null` | Optional byte-level resume threshold; requires and cannot exceed the resident high watermark. |
| `runtime.feature_store_max_resident_bytes` | `null` | Optional hard store budget; it cannot be smaller than the resident high watermark. |
| `runtime.feature_store_max_quarantined_bytes` | `8589934592` | Per-process Mooncake receive-buffer quarantine limit; exceeding it fails loudly. |

### `tracking`: experiment logging

| Field | Default | What to write |
| --- | --- | --- |
| `tracking.report_to` | `none` | `none`, `wandb`, `tensorboard`, `swanlab`, or `mlflow`. |
| `tracking.wandb_project` | `null` | W&B project. |
| `tracking.wandb_name` | `null` | W&B run name. |
| `tracking.wandb_key` | `null` | W&B API key; prefer the environment instead of committing it. |
| `tracking.wandb_offline` | `false` | Use W&B offline mode. |
| `tracking.wandb_dir` | `null` | W&B local state directory. |
| `tracking.swanlab_project` | `null` | SwanLab project. |
| `tracking.swanlab_name` | `null` | SwanLab run name. |
| `tracking.swanlab_key` | `null` | SwanLab API key; prefer the environment. |
| `tracking.mlflow_tracking_uri` | `null` | MLflow tracking endpoint. |
| `tracking.mlflow_experiment_name` | `null` | MLflow experiment. |
| `tracking.mlflow_run_name` | `null` | MLflow run name. |

### `profiling`: bounded per-rank traces

| Field | Default | What to write |
| --- | --- | --- |
| `profiling.enabled` | `false` | Enable PyTorch profiler traces on trainer ranks. Capture-only producer roles cannot enable it. |
| `profiling.start_step` | `30` | First completed optimizer step to profile. |
| `profiling.num_steps` | `4` | Positive number of optimizer steps to record. |
| `profiling.record_shapes` | `false` | Include tensor-shape metadata at additional overhead. |
