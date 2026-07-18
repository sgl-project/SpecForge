# Unified training recipe catalog

Every draft model JSON under `configs/` has at least one typed YAML recipe in
this directory. Run any recipe through the one public training entry:

```bash
specforge train --config examples/configs/qwen3-8b-eagle3-disaggregated.yaml
```

`model.draft_model_config` may name a local JSON file, a local model directory,
or a Hugging Face repository. Fresh EAGLE3, P-EAGLE, and DFlash runs may omit it
and derive the draft architecture from the target; see the
[training guide](../../docs/basic_usage/training.md#draft-configuration-and-model-initialization)
for layer/block overrides and the distinction between weights-only
`model.draft_checkpoint_path` and full `training.resume_from`.

Every recipe records its audited process count under `deployment.trainer`.
Multi-process configs self-launch through torch distributed:

```bash
specforge train -c examples/configs/qwen3-30b-a3b-eagle3-online.yaml
```

The filename is the index: `*-online.yaml` performs SGLang server capture while
training, `*-offline.yaml` consumes precomputed features, and
`*-disaggregated.yaml` highlights a producer/consumer topology. Every online
recipe is disaggregated even when its historical filename only says `online`.
VLM training is not supported, so the catalog contains text-only recipes.

The `qwen3-8b-dflash-windowed-fanout.yaml`,
`qwen3-8b-dflash-1server-dp7-disaggregated.yaml`,
`qwen3-8b-domino-1server-dp7-disaggregated.yaml`,
`qwen3-8b-domino-multiserver-disaggregated.yaml`,
`qwen3.6-27b-dflash-1server-dp2-disaggregated.yaml`, and
`qwen3.6-27b-dflash-multiserver-disaggregated.yaml` recipes are opt-in local
full-stack examples. Their typed `managed_local` blocks own Mooncake, one or
two patched SGLang capture servers, and the trainer GPU allocation; the same
`specforge train -c ...` command starts and cleans up each complete stack.
Disaggregated recipes without `managed_local` keep Mooncake and SGLang external
for scheduler- or service-managed deployments.

Before running a recipe, update model/data paths and create any referenced
offline feature or vocabulary-mapping artifacts. Managed-local recipes
intentionally record their GPU allocation and loopback services. External
deployments may record stable endpoints in YAML or override them through the
environment; credentials, tokens, and node-local identity should not be checked
into a portable recipe.

## Writing a recipe

The Pydantic models in `specforge/config/schema.py` are the authoritative
schema. This section explains what belongs in each YAML section and records the
defaults that matter when writing a recipe. Unknown or misspelled fields are
errors; YAML files and dotted CLI overrides go through the same validation.

New checked-in recipes should explicitly set `training.strategy`,
`deployment.mode`, `deployment.trainer.nnodes`,
`deployment.trainer.nproc_per_node`, `run_id`, and `output_dir`, even when the
schema has the same default. A minimal server-only online recipe looks like:

```yaml
model:
  target_model_path: Qwen/Qwen3-8B
  draft_model_config: configs/qwen3-8b-eagle3.json
  target_backend: sglang
  vocab_mapping_path: cache/vocab_mapping/qwen3-8b.pt

data:
  train_data_path: ./cache/dataset/sharegpt_train.jsonl
  max_length: 4096
  chat_template: qwen

training:
  strategy: eagle3
  max_steps: 10000
  batch_size: 1
  learning_rate: 0.0001

run_id: qwen3-8b-eagle3-disaggregated
output_dir: outputs/qwen3-8b-eagle3-disaggregated

deployment:
  mode: disaggregated
  trainer:
    nnodes: 1
    nproc_per_node: 1
  disaggregated:
    control_dir: outputs/qwen3-8b-eagle3-disaggregated/control
    consumer_state_dir: outputs/qwen3-8b-eagle3-disaggregated/consumer-state
    backend: mooncake
    server_urls:
      - http://127.0.0.1:30000
    mooncake_metadata_server: http://127.0.0.1:35880/metadata
    mooncake_master_server_addr: 127.0.0.1:35551
```

Paths are resolved from the current working directory. The checked-in recipes
assume the command runs from the repository root.

### Choose a starting recipe

| Workflow | Canonical starting point |
| --- | --- |
| Colocated offline | `qwen3-8b-eagle3-offline.yaml` |
| External-service online | `qwen3-8b-eagle3-disaggregated.yaml` |
| Managed-local disaggregated online | `qwen3-8b-domino-multiserver-disaggregated.yaml` |
| Independent heterogeneous consumers | `qwen3-8b-dflash-windowed-fanout.yaml` |
| Disaggregated offline | `qwen3-8b-eagle3-offline-disaggregated.yaml` |

The online/offline mode is derived from the selected `data` source, not from
the filename. The filename is a discoverability convention.

### Top-level fields

| Field | Default | What to write |
| --- | --- | --- |
| `run_id` | `specforge-run` | Stable identifier for the run. It names checkpoints and is also the default disaggregated store namespace. Use a new value for a fresh attempt. |
| `output_dir` | `./output` | Shared checkpoint, profiler, and tracker output directory. Every trainer rank must resolve it to the same location. |

The top-level `model` and `data` sections are required. `training`, `tracking`,
`profiling`, `runtime`, and `deployment` have defaults, but checked-in recipes
should make their training strategy and topology explicit.

### `model`: target, draft, and capture backend

| Field | Default | What to write |
| --- | --- | --- |
| `model.target_model_path` | required | Local target directory or Hugging Face repository ID. |
| `model.draft_model_config` | `null` | Draft JSON, model directory containing `config.json`, or Hugging Face repository. EAGLE3, P-EAGLE, and DFlash may omit it and derive a fresh config; Domino and DSpark require one. |
| `model.draft_checkpoint_path` | `null` | Weights-only warm start for a new run. Do not combine it with `training.resume_from`. |
| `model.draft_num_hidden_layers` | `null` | Positive fresh-architecture override where the strategy permits it. EAGLE3 remains one layer; P-EAGLE and DFlash may override their generated defaults. |
| `model.draft_block_size` | `null` | Positive DFlash block-size override; generated DFlash configs default to 16. |
| `model.target_backend` | `sglang` | `sglang` is the only accepted value; retired `hf`/`custom` names fail at config load. Offline feature consumers do not instantiate a target inference backend. |
| `model.input_modality` | `text` | The provider modality. The unified runtime supports text only; VLM modalities such as `qwen2_5_vl` are rejected. |
| `model.shard_target_output` | `false` | Retained for config migration; leave it false on the server-only online path. |
| `model.trust_remote_code` | `false` | Enable only for model repositories that require custom loading code. |
| `model.embedding_key` | `model.embed_tokens.weight` | Target checkpoint key copied into or used by the draft embedding. |
| `model.lm_head_key` | `lm_head.weight` | Target checkpoint key used for the frozen output head. |
| `model.vocab_mapping_path` | `""` | Target-to-draft vocabulary mapping. EAGLE3 disaggregated runs require an explicit shared file. |
| `model.load_target_embedding` | `true` | Copy the frozen target embedding into a fresh draft when supported. |
| `model.aux_hidden_state_layer_ids` | `null` | Optional EAGLE3/P-EAGLE capture override containing exactly three non-negative layer IDs. Other strategies derive layers from the draft config. |
| `model.torch_dtype` | `bfloat16` | `bfloat16`, `float16`, or `float32`. |
| `model.cache_dir` | `null` | Model/tokenizer download cache. This is distinct from `data.cache_dir`. |
| `model.mask_token_id` | `null` | DFlash-family/P-EAGLE mask token override. Otherwise it resolves from the draft config and then the tokenizer. |
| `model.sglang_attention_backend` | `flashinfer` | SGLang attention implementation for an in-process or managed capture server. |
| `model.sglang_mem_fraction_static` | `0.4` | SGLang static-memory fraction in `(0, 1]`; inherited by managed capture servers unless they override it. |
| `model.sglang_context_length` | `null` | Positive explicit context limit. Managed capture requires at least `data.max_length + 7`; omitting it derives that value. |
| `model.sglang_enable_nccl_nvls` | `false` | Pass the matching SGLang NCCL NVLS optimization flag. |
| `model.sglang_enable_symm_mem` | `false` | Pass the matching SGLang symmetric-memory flag. |
| `model.sglang_enable_torch_compile` | `false` | Enable the SGLang torch-compile path. |
| `model.sglang_enable_dp_attention` | `false` | Enable SGLang DP attention where supported. Managed-local capture currently rejects it. |
| `model.sglang_enable_dp_lm_head` | `false` | Enable SGLang DP LM head where supported. Managed-local capture currently rejects it. |
| `model.sglang_ep_size` | `1` | SGLang expert-parallel size; it must divide and not exceed every managed capture server's `tp_size`. |
| `model.sglang_max_running_requests` | `null` | Positive SGLang request-concurrency limit. |
| `model.sglang_max_total_tokens` | `null` | Positive SGLang token-pool limit. |

### `data`: choose exactly one training source

Exactly one of the first three fields must be non-empty:

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
| `data.dataloader_num_workers` | `null` | Ordered feature-loader workers. `null` preserves strategy defaults: EAGLE/P-EAGLE 4, DFlash-family 8; use 0 for synchronous loading. |
| `data.cache_dir` | `./cache` | Prepared dataset and derived vocabulary-mapping cache. |
| `data.cache_key` | `null` | Optional explicit namespace when multiple preparations share the same source. |
| `data.max_prompts` | `null` | Optional non-negative prompt cap, useful for smoke tests. |

Offline evaluation uses `eval_hidden_states_path`; configure it together with
`training.eval_interval`. Online evaluation is unsupported, and setting
`eval_data_path` fails config validation.

### `training`: optimization, strategy, and parallelism

Common fields:

| Field | Default | What to write |
| --- | --- | --- |
| `training.strategy` | `eagle3` | `eagle3`, `peagle`, `dflash`, `domino`, or `dspark`. |
| `training.num_epochs` | `1` | Positive passes over a finite source. |
| `training.max_steps` | `null` | Positive hard stop in optimizer steps. If it is set while `total_steps` is omitted, it is also the fallback schedule horizon. |
| `training.total_steps` | `null` | Positive optimizer/loss schedule horizon; it does not itself stop an online stream. A finite online disaggregated run may omit both fields: the producer publishes the exact horizon derived from prepared prompts, epochs, DP size, batch size, and accumulation. |
| `training.batch_size` | `1` | Per-rank microbatch size. P-EAGLE and USP require 1. |
| `training.accumulation_steps` | `1` | Positive microbatches per optimizer update. |
| `training.learning_rate` | `1e-4` | Positive peak learning rate. |
| `training.warmup_ratio` | `0.015` | Fraction in `[0, 1]` used for scheduler warmup. |
| `training.max_grad_norm` | `0.5` | Positive gradient-clipping norm. |
| `training.attention_backend` | `flex_attention` | `eager`, `sdpa`, `flex_attention`, `fa`, or `usp`; the selected strategy must support it. |
| `training.tp_size` | `1` | Online disaggregated consumers must keep it at 1; configure target TP on capture servers. Offline non-USP ranks consume disjoint data. |
| `training.sp_ulysses_size` | `1` | Ulysses sequence-parallel factor for offline EAGLE3 USP. |
| `training.sp_ring_size` | `1` | Ring sequence-parallel factor for offline EAGLE3 USP. |
| `training.dist_timeout` | `10` | Positive distributed-operation timeout in minutes. |
| `training.save_interval` | `0` | Save every N optimizer steps; 0 disables periodic saves. A final checkpoint is still written. |
| `training.eval_interval` | `0` | Evaluate every N optimizer steps; 0 disables evaluation. |
| `training.log_interval` | `50` | Positive optimizer-step logging interval. |
| `training.max_checkpoints` | `0` | Keep the newest N checkpoints; 0 keeps all. |
| `training.resume_from` | `null` | Full-run checkpoint/run root: draft, optimizer/scheduler, counters, data position, and RNG. Mutually exclusive with `model.draft_checkpoint_path`. |
| `training.compact_teacher` | `false` | Exact lower-peak-memory teacher projection for offline text EAGLE3. |
| `training.compact_teacher_chunk_size` | `null` | Positive vocabulary chunk size; requires `compact_teacher: true`. |
| `training.role` | `all` | Use `all` for local offline training; disaggregated entrypoints select `auto`, `producer`, or `consumer`. |
| `training.seed` | `42` | Run and per-rank RNG seed. |

Strategy-specific fields should be written only when tuning that objective:

| Strategy | Fields and defaults |
| --- | --- |
| EAGLE3 | `training.ttt_length` (`7`), `training.lk_loss_type` (`null`; `lambda` or `alpha`), `training.kl_scale` (`1.0`), `training.kl_decay` (`1.0`) |
| DFlash / Domino / D-PACE | `training.num_anchors` (`512`), `training.loss_decay_gamma` (`null`), `training.loss_type` (`dflash`), `training.dpace_alpha` (`0.5`), `training.lambda_base_start` (`1.0`), `training.lambda_base_decay_ratio` (`0.5`) |
| DSpark | `training.dspark_ce_loss_alpha` (`0.1`), `training.dspark_l1_loss_alpha` (`0.9`), `training.dspark_confidence_head_alpha` (`1.0`) |
| P-EAGLE | `training.num_depths` (`8`), `training.down_sample_ratio` (`0.8`), `training.down_sample_ratio_min` (`0.2`), `training.norm_before_residual` (`null`) |

New recipes must not write the loader-only migration fields
`training.deployment_mode`, `training.server_urls`, or
`training.metadata_db_path`; use the typed `deployment.*` surface below.

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

The trainer world size is `nnodes * nproc_per_node`. `training.tp_size` must
remain 1, and the world size must be divisible by
`training.sp_ulysses_size * training.sp_ring_size`.

For `deployment.mode: disaggregated`, also write:

| Field | Default | What to write |
| --- | --- | --- |
| `deployment.disaggregated.control_dir` | required | Fresh attempt-scoped shared directory for refs/manifest and lifecycle markers. |
| `deployment.disaggregated.backend` | required | `mooncake` or `shared_dir`. Online disaggregated runs require Mooncake. |
| `deployment.disaggregated.consumer_state_dir` | `null` | Node-local rank-0 SQLite/WAL root. Required for multi-node online consumers; their rank inboxes remain under shared `control_dir`. |
| `deployment.disaggregated.store_root` | `null` | Shared feature directory; required when `backend: shared_dir`. |
| `deployment.disaggregated.store_id` | `null` | Feature-store namespace; defaults to `run_id`. |
| `deployment.disaggregated.server_urls` | `[]` | External patched SGLang capture endpoints. One rollout worker is created per entry. Do not set with `managed_local`. |
| `deployment.disaggregated.mooncake_metadata_server` | `null` | External Mooncake metadata URL. |
| `deployment.disaggregated.mooncake_master_server_addr` | `null` | External Mooncake RPC `host:port`. |
| `deployment.disaggregated.mooncake_local_hostname` | `null` | Node-local Mooncake transfer hostname; usually supplied through the environment. |
| `deployment.disaggregated.mooncake_protocol` | `null` | External transfer protocol such as `tcp` or `rdma`. |
| `deployment.disaggregated.mooncake_rdma_devices` | `null` | External Mooncake RDMA-device selection. |
| `deployment.disaggregated.producer_segment_size` | `null` | Positive allocation owned by an offline Mooncake producer. Online capture is server-owned and forces client segments to zero. |
| `deployment.disaggregated.client_buffer_size` | `268435456` | Per-role Mooncake client buffer in bytes. |
| `deployment.disaggregated.idle_timeout_s` | `null` | Positive consumer idle timeout. |
| `deployment.disaggregated.peer_wait_timeout_s` | `null` | Optional positive producer/consumer peer-completion timeout. Unset is unbounded; expiration fails the attempt. |
| `deployment.disaggregated.producer_hold_s` | `null` | Optional positive offline producer retention timeout. Unset is unbounded; expiration fails the attempt. |
| `deployment.disaggregated.shutdown_grace_s` | `30.0` | SIGTERM-to-SIGKILL window for a plain supervisor teardown; must cover worker cleanup (Mooncake drains, checkpoint flush, failure sentinels). `managed_local` stacks use `managed_local.shutdown_grace_s`. |
| `deployment.disaggregated.managed_local` | `null` | Optional owned single-node Mooncake + capture-server stack described below. |

The four path fields have different ownership:

| Path | Lifetime and visibility |
| --- | --- |
| `output_dir` | Shared durable checkpoints and run outputs. |
| `deployment.disaggregated.control_dir` | Shared, fresh attempt control state. |
| `deployment.disaggregated.consumer_state_dir` | Fresh node-local online-consumer ledger; required for multi-node runs. |
| `deployment.disaggregated.store_root` | Shared offline feature payloads for `shared_dir`. |

An external-service online run writes `server_urls` and Mooncake endpoints in
YAML, or injects their environment equivalents. A managed-local run replaces
those fields with one owned stack:

```yaml
deployment:
  mode: disaggregated
  trainer:
    nnodes: 1
    nproc_per_node: 2
  disaggregated:
    control_dir: ./outputs/domino/control
    backend: mooncake
    managed_local:
      trainer_cuda_visible_devices: ["2", "3"]
      mooncake:
        protocol: tcp
      capture_servers:
        - port: 30000
          cuda_visible_devices: ["0"]
          tp_size: 1
        - port: 30001
          cuda_visible_devices: ["1"]
          tp_size: 1
```

Managed-local fields:

| Field | Default | What to write |
| --- | --- | --- |
| `deployment.disaggregated.managed_local.trainer_cuda_visible_devices` | required | One device token per trainer rank, or one per independent `windowed_fanout` consumer; trainer and capture devices must not overlap. |
| `deployment.disaggregated.managed_local.mooncake` | default object | Owned loopback Mooncake configuration described by the nested fields below. |
| `deployment.disaggregated.managed_local.capture_servers` | required | One or more owned patched SGLang server definitions. |
| `deployment.disaggregated.managed_local.shutdown_grace_s` | `30` | Positive graceful process-group shutdown window. |
| `deployment.disaggregated.managed_local.mooncake.rpc_port` | `35551` | Owned Mooncake RPC port. |
| `deployment.disaggregated.managed_local.mooncake.metadata_port` | `35880` | Owned metadata HTTP port. |
| `deployment.disaggregated.managed_local.mooncake.metrics_port` | `35903` | Owned metrics port. |
| `deployment.disaggregated.managed_local.mooncake.local_hostname` | `127.0.0.1` | Local transfer hostname. |
| `deployment.disaggregated.managed_local.mooncake.protocol` | `tcp` | `tcp` or `rdma`. |
| `deployment.disaggregated.managed_local.mooncake.rdma_devices` | `null` | RDMA-device selection when using RDMA. |
| `deployment.disaggregated.managed_local.mooncake.global_segment_size_bytes` | `34359738368` | Owned global segment size. |
| `deployment.disaggregated.managed_local.mooncake.local_buffer_size_bytes` | `1073741824` | Owned local client buffer. |
| `deployment.disaggregated.managed_local.mooncake.startup_timeout_s` | `60` | Positive Mooncake readiness timeout. |
| `deployment.disaggregated.managed_local.capture_servers[].port` | required | Unique capture HTTP port. |
| `deployment.disaggregated.managed_local.capture_servers[].cuda_visible_devices` | required | Device tokens for this server. Their count must equal its `tp_size`. |
| `deployment.disaggregated.managed_local.capture_servers[].tp_size` | `1` | Target-model tensor parallelism for this server. |
| `deployment.disaggregated.managed_local.capture_servers[].mem_fraction_static` | `null` | Optional SGLang static-memory override in `(0, 1]`; otherwise inherit `model.sglang_mem_fraction_static`. |
| `deployment.disaggregated.managed_local.capture_servers[].attention_backend` | `null` | Server-specific override; otherwise inherit `model.sglang_attention_backend`. |
| `deployment.disaggregated.managed_local.capture_servers[].startup_timeout_s` | `1800` | Positive server readiness timeout. |

Managed-local is only for a fresh, single-node, online Mooncake run. It derives
server URLs and Mooncake endpoints, so do not combine it with explicit external
endpoints, `store_root`, or `producer_segment_size`. It does not support resume,
an existing torchrun, or `--node-rank`. All owned ports and GPU assignments must
be disjoint.

### Independent windowed fanout

`deployment.disaggregated.windowed_fanout` runs one producer and multiple
single-process consumers from the same YAML. Each child still enters through
`specforge train`; `launch_plan` selects a consumer with `--consumer-id` and
projects its loss, block size, anchors, optimizer schedule, seed, checkpoint,
and output directory onto the canonical trainer configuration.

This topology is intentionally different from data-parallel training. Consumers
advance independently and may have different step costs. A shared SQLite
registry records each cursor and bounded capture interest, while Mooncake owns
the tensor payloads. The producer captures a sample only when a consumer's
window requests it, reuses a live compatible capture across consumers, and
reclaims entries outside every legal window.

Required constraints are:

- online DFlash with Mooncake and exactly one prompt epoch;
- a positive fixed `data.max_prompts`, divisible by batch size times
  accumulation;
- `deployment.trainer` fixed at 1x1 because each consumer is its own process;
- exactly one capture server, shared by every consumer;
- one managed-local trainer device per consumer, or one explicit
  `cuda_visible_device` on every external consumer;
- `max_live_refs`, `max_live_bytes`, and `max_outstanding_per_consumer` large
  enough for the configured batch and capture reservation.

The main bounds and timing controls are:

| Field | Default | Meaning |
| --- | --- | --- |
| `window_lookbehind` / `window_lookahead` | `2` / `40` | Shared default legal window around each consumer cursor. |
| `max_prefetch_per_consumer` | `8` | Maximum speculative requests inside the lookahead window. |
| `max_outstanding_per_consumer` | `8` | Hard per-consumer acquired-ref bound; must cover one accumulated optimizer step. |
| `max_live_refs` / `max_live_bytes` | required byte bound | Global capture capacity shared by all consumers. |
| `capture_reservation_bytes` | 128 MiB | Bytes reserved transactionally before a capture starts. |
| `capture_max_sample_bytes` | 128 MiB | Server-side rejection bound for one captured sample; it cannot exceed the transactional reservation. |
| `capture_batch_size` / `capture_batch_wait_s` | `8` / `0.002` | Producer request batching. |
| `consumer_prefetch_batches` | `1` | Loader-side prefetch workers for each consumer. |

Each consumer may override `window_lookbehind`, `window_lookahead`, and
`max_prefetch` in addition to its required training fields. External deployments
may resume consumers independently with `resume_from`; managed-local runs are
fresh attempts and reject resume checkpoints.

### `runtime`: streaming backpressure

This section affects disaggregated streaming producers and is normally omitted
unless tuning throughput or memory pressure.

| Field | Default | What to write |
| --- | --- | --- |
| `runtime.producer_lease` | `8` | Prompts leased to a rollout worker at once. |
| `runtime.in_flight_high_watermark` | `256` | Pause production at this many committed, unacknowledged refs. |
| `runtime.in_flight_low_watermark` | `192` | Resume production at or below this count; it cannot exceed the high watermark. |
| `runtime.resident_high_watermark_bytes` | `null` | Optional byte-level pause threshold. |
| `runtime.resident_low_watermark_bytes` | `null` | Optional byte-level resume threshold; requires and cannot exceed the resident high watermark. |
| `runtime.feature_store_max_resident_bytes` | `null` | Optional hard store budget; it cannot be smaller than the resident high watermark. |

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

### Cross-section checks

- Offline feature consumers require `training.tp_size: 1`. Non-USP ranks each
  consume a disjoint data shard; USP peers share a sequence within their SP
  group while draft-DP groups remain disjoint.
- Online disaggregated runs require `model.target_backend: sglang` and
  `backend: mooncake`. When both step fields are omitted, the producer publishes
  the finite prompt plan's optimizer horizon and the consumer trains to EOF;
  `max_steps` remains an optional hard cap. Every trainer rank is data parallel,
  so
  `training.tp_size`, `training.sp_ulysses_size`, and
  `training.sp_ring_size` must remain 1; configure target TP on each capture
  server.
- USP is offline EAGLE3 only, requires `training.batch_size: 1`, and requires
  `sp_ulysses_size * sp_ring_size > 1`. Non-USP runs keep both SP sizes at 1.
- P-EAGLE reuses the EAGLE3 server feature schema, uses `flex_attention`, and
  requires batch size 1. DSpark is also server-only online.
- VLM training, including Qwen2.5-VL, is not supported. Online capture accepts
  text inputs only.
- `training.compact_teacher` is offline text EAGLE3 only.
- Online evaluation is not supported. Offline `data.eval_hidden_states_path`
  and `training.eval_interval` must be configured together.

Validate the complete schema and inspect the resolved processes without
starting a run:

```bash
specforge train -c examples/configs/my-run.yaml --plan
```

Use validated dotted overrides for temporary changes:

```bash
specforge train -c examples/configs/my-run.yaml \
  training.learning_rate=5e-5 \
  'deployment.disaggregated.server_urls=["http://capture-0:30000"]'
```

For deeper lifecycle and recovery semantics, see the
[training guide](../../docs/basic_usage/training.md) and
[disaggregated training guide](../../docs/basic_usage/disaggregated_training.md).

## Capability matrix

| Strategy | SGLang server online | Local offline | Disaggregated offline |
| --- | --- | --- | --- |
| EAGLE3 | consumer DP | DP + USP | consumer DP |
| DFlash | consumer DP | DP | consumer DP |
| Domino | consumer DP | DP | consumer DP |
| DSpark | consumer DP | No | No |
| P-EAGLE | consumer DP, batch size 1 | No | No |

`qwen3-8b-dpace-online.yaml` is the D-PACE recipe. It deliberately uses the
shared DFlash strategy with `training.loss_type: dpace`; D-PACE is an objective
selection inside the unified trainer, not another training entry.

Evaluation is currently offline-only and pairs `training.eval_interval` with
`data.eval_hidden_states_path`. Best checkpoints are
linked as `<run_id>-best`. Offline text EAGLE3 may enable
`training.compact_teacher`; `tracking.report_to` selects `none`, W&B,
TensorBoard, SwanLab, or MLflow.

## Ascend NPU launch

Install vendor-matched PyTorch, `torch_npu`, Mooncake, and an NPU-compatible
SGLang capture server first. The `*-npu.yaml` consumers use SDPA while target
capture remains outside the trainer. For example:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

specforge train -c examples/configs/qwen3.5-4b-dflash-online-npu.yaml
```

The unified launcher provides rank/world/rendezvous variables and the runtime
selects HCCL when `torch_npu` is active. For AMD GPUs, install
`requirements-rocm.txt`; HF + SDPA is the portable ROCm starting point.

Local offline and disaggregated offline resume are supported.
Disaggregated online recovery resumes only the consumer against retained
control/data-plane state; capture producers always start a fresh attempt.

Migration notes:

- The former `run_qwen3_8b_dflash_disagg_1srv_dp7.sh` self-contained topology
  is retained as `qwen3-8b-dflash-1server-dp7-disaggregated.yaml`: one managed
  capture server on GPU 0 and seven DFlash trainer ranks on GPUs 1–7.
- The former `run_qwen3_8b_domino_disagg_1srv_dp7.sh` self-contained topology
  is retained as `qwen3-8b-domino-1server-dp7-disaggregated.yaml`: one managed
  capture server on GPU 0 and seven trainer ranks on GPUs 1–7.
- The former `run_qwen3.6_27b_dflash_disagg.sh` one-server topology is retained
  as `qwen3.6-27b-dflash-1server-dp2-disaggregated.yaml`: one managed capture
  server on GPU 0 and two trainer ranks on GPUs 1–2. The external-service YAML
  remains available for scheduler-managed deployments.
- The latest pre-cleanup `run_qwen3_8b_domino_disagg_multiserver.sh` had been
  reduced to one SGLang server and one URL despite its historical name. The
  managed Qwen3-8B Domino recipe above restores a genuine two-server topology
  without restoring the legacy trainer script; the external-service Domino
  recipe also accepts any number of typed `deployment.disaggregated.server_urls`.
- The former GPT-OSS-120B shell accidentally selected the 20B draft config;
  `gpt-oss-120b-eagle3-online.yaml` points to the matching 120B config.
- The former Qwen3-235B shell accidentally launched Qwen3-Next-80B; the two now
  have separate recipes.
- Qwen3-Next online EAGLE3 retains its batch size of two. P-EAGLE requires
  batch size one.
- The old Qwen3.5-35B offline shell had its training command commented out. The
  YAML records that intended offline trainer configuration after feature
  preparation.
- Qwen3-8B DTA still shares the DFlash trainer, as before. Its specialized
  behavior is encoded by the draft JSON (`training_mode: vp_drafter`); there is
  no second DTA training entry.
