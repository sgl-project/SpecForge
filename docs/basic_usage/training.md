# Training

SpecForge has one public training entry point for every strategy and runtime
topology:

```bash
specforge train --config examples/configs/qwen3-8b-eagle3-online.yaml
```

The YAML file is the run contract. It selects the draft strategy, target model,
data source, optimizer settings, and deployment mode. Method-specific Python
trainers are not part of the public interface.

This is an intentional hard cutover. The old `scripts/train_*.py` commands and
temporary move-only Python import paths were removed rather than deprecated.
Downstream launchers should migrate to a typed run config and `specforge train`;
there is no compatibility dispatch to the previous trainers.

## Launch a run

Use the command directly for every checked-in topology:

```bash
specforge train --config examples/configs/qwen3-8b-eagle3-online.yaml
```

`deployment.trainer.nproc_per_node` records the audited local process count.
When it is greater than one, the CLI starts torch distributed itself:

```bash
specforge train -c examples/configs/qwen3-30b-a3b-eagle3-online.yaml
```

`training.tp_size` defines each frozen-target capture group. SGLang targets and
the supported sharded target backends use that group for weight tensor
parallelism. DFlash and Domino with `model.target_backend: hf` instead load a
complete target replica on every rank; for them, `tp_size` controls the shared
capture batch and local batch partition, not model-weight sharding. When the
world size is larger than `tp_size`, the resulting target-DP groups receive
disjoint prompt shards with deterministic padding so every trainer rank
performs the same number of steps. Offline runs shard fixed feature references
in the same way. See [Parallel topologies](#parallel-topologies) for USP and
accelerator notes.

Paths in a config are resolved from the current working directory. The example
configs assume that the command is run from the repository root.

You can override an existing value without copying the YAML. Overrides use
validated `section.field=value` syntax:

```bash
specforge train \
  --config examples/configs/qwen3-8b-eagle3-online.yaml \
  training.learning_rate=5e-5 \
  training.max_steps=100 \
  output_dir=./outputs/eagle3-smoke
```

Unknown config fields and unknown override paths are errors. This keeps a
misspelled or retired option from being silently ignored.

## Run config

A run config has seven typed sections (`model`, `data`, `training`, `tracking`,
`profiling`, `runtime`, and `deployment`) plus `run_id` and `output_dir`:

```yaml
model:
  target_model_path: Qwen/Qwen3-8B
  draft_model_config: configs/qwen3-8b-eagle3.json
  target_backend: sglang
  torch_dtype: bfloat16

data:
  train_data_path: ./cache/dataset/sharegpt_train.jsonl
  max_length: 4096
  chat_template: qwen
  cache_dir: ./cache

training:
  strategy: eagle3
  num_epochs: 10
  batch_size: 1
  learning_rate: 1.0e-4
  save_interval: 1000

deployment:
  mode: local_colocated
  trainer:
    nnodes: 1
    nproc_per_node: 1

run_id: qwen3-8b-eagle3-online
output_dir: ./outputs/qwen3-8b-eagle3-online
```

### Draft configuration and model initialization

`model.draft_model_config` accepts any of these equivalent config sources:

- a local JSON file;
- a local draft-model directory containing `config.json`;
- a Hugging Face draft-model repository ID.

It may be omitted for EAGLE3, P-EAGLE, and DFlash. In that case SpecForge
derives a registered draft config from the target config. The defaults preserve
the former trainers: one EAGLE3 layer, four P-EAGLE layers, and one DFlash layer
with block size 16. Typed overrides are available when creating a different
fresh architecture:

```yaml
model:
  target_model_path: Qwen/Qwen3-8B
  draft_num_hidden_layers: 2  # P-EAGLE or DFlash; EAGLE3 remains one layer
  draft_block_size: 8         # DFlash only
```

Domino and DSpark need their projector/head metadata, so they require an
explicit draft config (or a pretrained warm-start source that contains
`config.json`). The old Domino parser exposed an optional config flag, but its
no-config branch immediately failed because those required projector fields
had no defaults; the unified schema rejects that unusable combination early.

There are two deliberately separate checkpoint operations:

| Intent | Config field | Restored state |
| --- | --- | --- |
| Continue the same run | `training.resume_from` | draft weights, optimizer/scheduler, epoch/step/data position, and per-rank RNG |
| Initialize a new run from weights | `model.draft_checkpoint_path` | draft weights only |

A weights-only warm start accepts a Hugging Face model directory/repository or
a SpecForge checkpoint directory, `training_state.pt`, or run root. If the warm
source contains `config.json`, it also supplies the draft architecture unless
`model.draft_model_config` is explicit. Warm start never restores optimizer
state, counters, data position, or RNG, and it is mutually exclusive with
`training.resume_from`:

```yaml
model:
  target_model_path: Qwen/Qwen3-8B
  draft_checkpoint_path: ./outputs/base/base-step1000
```

For an online disaggregated run, the producer may receive the same field. It
uses only the warm source's draft configuration to derive the capture contract;
the consumer alone loads the draft weights and optimizer.

Set exactly one data source:

- `data.train_data_path` for raw conversation or preformatted online data;
- `data.prompts_path` for pre-tokenized online JSONL containing `input_ids`
  and `loss_mask`;
- `data.hidden_states_path` for precomputed offline feature checkpoints.

The checked-in examples are the canonical starting points:

| Strategy and mode | Config |
| --- | --- |
| EAGLE3 online | [`qwen3-8b-eagle3-online.yaml`](../../examples/configs/qwen3-8b-eagle3-online.yaml) |
| EAGLE3 offline | [`qwen3-8b-eagle3-offline.yaml`](../../examples/configs/qwen3-8b-eagle3-offline.yaml) |
| DFlash online | [`qwen3-8b-dflash-online.yaml`](../../examples/configs/qwen3-8b-dflash-online.yaml) |
| Domino online | [`qwen3-8b-domino-online.yaml`](../../examples/configs/qwen3-8b-domino-online.yaml) |
| P-EAGLE online | [`qwen3-8b-peagle-online.yaml`](../../examples/configs/qwen3-8b-peagle-online.yaml) |
| DFlash disaggregated | [`qwen3-8b-dflash-disaggregated.yaml`](../../examples/configs/qwen3-8b-dflash-disaggregated.yaml) |
| Domino disaggregated | [`qwen3-8b-domino-disaggregated.yaml`](../../examples/configs/qwen3-8b-domino-disaggregated.yaml) |
| DSpark disaggregated | [`qwen3-4b-dspark-disaggregated.yaml`](../../examples/configs/qwen3-4b-dspark-disaggregated.yaml) |
| EAGLE3 offline disaggregated | [`qwen3-8b-eagle3-offline-disaggregated.yaml`](../../examples/configs/qwen3-8b-eagle3-offline-disaggregated.yaml) |
| Qwen2.5-VL 7B online | [`qwen2.5-vl-7b-eagle3-online.yaml`](../../examples/configs/qwen2.5-vl-7b-eagle3-online.yaml) |
| Qwen2.5-VL 32B online | [`qwen2.5-vl-32b-eagle3-online.yaml`](../../examples/configs/qwen2.5-vl-32b-eagle3-online.yaml) |
| Ascend NPU DFlash online | [`qwen3.5-4b-dflash-online-npu.yaml`](../../examples/configs/qwen3.5-4b-dflash-online-npu.yaml) |
| Ascend NPU Domino online | [`qwen3.5-4b-domino-online-npu.yaml`](../../examples/configs/qwen3.5-4b-domino-online-npu.yaml) |

## Online and offline data

Online training captures target features while the run is active. It uses
little disk space but keeps target inference available during training.
Offline training reads feature checkpoints generated ahead of time, so only
the draft model must fit on the training GPUs at the cost of substantially
more storage.

| Mode | Target during training | Disk use | Data config |
| --- | --- | --- | --- |
| Online | Loaded locally or exposed by capture servers | Low | `train_data_path` or `prompts_path` |
| Offline | Not loaded by the trainer | High | `hidden_states_path` |

Prepare raw datasets and offline features as described in [Data
Preparation](data_preparation.md), then update the matching example YAML before
launching it.

## Supported combinations

The unified runtime supports text and Qwen2.5-VL training in these
combinations:

| Strategy | Colocated online | Local/dataflow offline | Disaggregated online | Disaggregated offline |
| --- | --- | --- | --- | --- |
| EAGLE3 | Yes, target TP + target-DP | Yes, DP + USP | Yes, consumer DP | Yes, consumer DP |
| DFlash | Yes; SGLang target TP + target-DP, or HF replicas + batch partition | Yes, DP | Yes, consumer DP | Yes, consumer DP |
| Domino | Yes; SGLang target TP + target-DP, or HF replicas + batch partition | Yes, DP | Yes, consumer DP | Yes, consumer DP |
| DSpark | No | No | Yes, consumer DP | No |
| P-EAGLE | Yes, batch size 1 | No | No | No |

Unsupported combinations fail explicitly during config validation or run
assembly. In particular:

- Qwen2.5-VL uses `model.input_modality: qwen2_5_vl` in a colocated online
  EAGLE3 run. Its raw records retain `image` and `conversations`; image tensors
  are materialized only inside rollout and are not stored in the control plane.
  Each raw record currently supports one image. Ragged VLM samples are padded
  by the unified online collator, including their three-axis M-RoPE position
  IDs;
- attention backends are strategy-specific: EAGLE3 accepts `sdpa`,
  `flex_attention`, `fa`, or offline `usp`; P-EAGLE requires
  `flex_attention`; DFlash, Domino, and DSpark accept `eager`, `sdpa`, or
  `flex_attention`;
- P-EAGLE requires `training.batch_size=1`; text and Qwen2.5-VL EAGLE3 support
  larger batches, while the checked-in VLM recipes retain their conservative
  batch-size-one defaults;
- DSpark requires disaggregated server capture;
- offline feature training supports EAGLE3, DFlash, and Domino;
- every online disaggregated run uses `model.target_backend=sglang` and sets
  either `training.total_steps` or `training.max_steps`;
- EAGLE3 local offline runs derive and cache a deterministic vocabulary mapping
  from the feature corpus when `model.vocab_mapping_path` is empty. EAGLE3
  disaggregated runs require an explicit shared mapping so producer and
  consumer cannot derive different artifacts.

There is no fallback to a removed training script.

Step limits are global optimizer updates. `training.max_steps` is a stop cap
and becomes the optimizer/loss schedule horizon when `training.total_steps` is
omitted. `training.total_steps` can describe a longer schedule, but does not by
itself stop an online stream.

## Parallel topologies

The launcher creates every process group from the typed run config:

- `training.tp_size` defines the frozen-target capture group. SGLang targets,
  sharded custom targets, and text EAGLE3's supported HF TP path use it for
  model-weight tensor parallelism. DFlash and Domino HF targets do not: every
  rank loads a complete target replica, while the group still captures a
  `tp_size * training.batch_size` batch and keeps one local partition per rank.
  Batch partitioning is therefore not evidence of weight sharding.
- Peers in a target capture group process the same global batch; when
  `world_size > tp_size`, each target-DP group receives a disjoint prompt
  shard. For colocated text EAGLE3 with SGLang,
  `model.shard_target_output: true` can return each TP rank's local batch
  partition directly. Other backends, and VLM, capture the full target batch
  before partitioning outputs locally.
- Offline references use the same target-TP/target-DP partition. Each TP group
  sees the same samples, while different DP groups see disjoint samples.
- EAGLE3 offline can set `training.attention_backend: usp` and choose
  `training.sp_ulysses_size` and `training.sp_ring_size`. Their product must be
  greater than one, and USP currently uses `training.batch_size: 1`.
- An online disaggregated consumer reserves every trainer rank for DP. Keep
  `training.tp_size`, `sp_ulysses_size`, and `sp_ring_size` at one and configure
  target TP on the external SGLang server.

The world size must be divisible by both `training.tp_size` and
`training.sp_ulysses_size * training.sp_ring_size`. Use a shared `output_dir`
for multi-rank checkpoints.

## Loader and profiling controls

`data.dataloader_num_workers` controls ordered background feature
materialization. If omitted, the former trainer defaults are retained:
EAGLE3/P-EAGLE use four workers and DFlash-family strategies use eight. Set it
to zero for fully synchronous loading.

Enable a bounded, per-rank PyTorch trace without a separate profiler entry:

```yaml
profiling:
  enabled: true
  start_step: 30
  num_steps: 4
  record_shapes: false
```

The window is expressed in completed optimizer steps, works across gradient
accumulation and resume, and writes one trace per rank beneath `output_dir`.
An active partial window is finalized when training stops or fails.

## Evaluation and best checkpoints

Evaluation is configured through the same YAML:

```yaml
data:
  train_data_path: ./cache/dataset/train.jsonl
  eval_data_path: ./cache/dataset/eval.jsonl

training:
  eval_interval: 100
```

Use `data.eval_data_path` with an online colocated run, or
`data.eval_hidden_states_path` with a local or disaggregated offline run. The
evaluation source and `training.eval_interval` must be set together and must
match the training mode. Online disaggregated evaluation is not supported.

Online evaluation creates an independent prompt, rollout, and loader stream at
each interval, so it does not consume or rewind the training stream. Offline
evaluation uses the same feature reader and collator as training and retains
the final partial batch. Metrics are emitted under `eval/*`.

The default selection metric is `eval/simulated_acc_len`. An improvement writes
a complete checkpoint and points `<run_id>-best` at it, even when
`training.save_interval` is zero. `<run_id>-latest` continues to identify the
newest complete checkpoint.

## Compact offline teacher

Offline text EAGLE3 can project teacher targets in exact vocabulary chunks
instead of materializing full-vocabulary fp32 logits:

```yaml
training:
  strategy: eagle3
  compact_teacher: true
  compact_teacher_chunk_size: 4096
```

This lowers peak memory without changing the teacher distribution, at the cost
of additional projection passes. The option is intentionally rejected for
online, VLM, and non-EAGLE3 runs.

## Experiment tracking

Console metrics remain available for every run. Select one optional external
backend with the top-level `tracking` section:

```yaml
tracking:
  report_to: tensorboard
```

Accepted values are `none`, `wandb`, `tensorboard`, `swanlab`, and `mlflow`.
W&B, SwanLab, and MLflow have matching project/run fields in the typed schema;
TensorBoard writes beneath `output_dir/runs`, and SwanLab writes beneath
`output_dir/swanlog`. Trainer and evaluator metrics use `train/*` and `eval/*`
names consistently.

## CUDA, ROCm, and Ascend NPU

CUDA and ROCm runs use the same YAML and entry point. For ROCm, install the
checked-in environment before installing SpecForge:

```bash
python -m pip install -r requirements-rocm.txt
python -m pip install -e .
```

Use a model/backend combination supported by that PyTorch ROCm environment;
HF + SDPA recipes are the portable baseline. PyTorch exposes ROCm devices
through its `torch.cuda` API and distributed runs use NCCL.

For Ascend, install the vendor-matched PyTorch and `torch_npu` packages first.
The checked-in NPU recipes use the HF target backend and SDPA. A four-device
launch is:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

specforge train -c examples/configs/qwen3.5-4b-dflash-online-npu.yaml
```

The unified launcher supplies rank, world-size, and rendezvous variables. The
runtime selects the active device dynamically and uses HCCL when `torch_npu`
is active.

## Disaggregated roles

A single-node disaggregated config supervises producer and consumer with one
command. Split deployments use the same config with `--role producer` or
`--role consumer`; multi-node consumers add only `--node-rank` on each host.
The optional `examples/disagg/run_online.sh` and `run_offline.sh` files are thin
delegates, not topology wrappers. See the
[disaggregated training guide](disaggregated_training.md) for external
Mooncake/SGLang prerequisites, freshness rules, and both launch forms.

## Checkpoints and resume

`training.save_interval` controls checkpoint frequency and
`training.max_checkpoints` controls rotation. Checkpoints are written beneath
`output_dir`. A completed trainer run always saves its final runtime checkpoint,
even when `save_interval` is zero or the final step is not an interval boundary.
The `<run_id>-latest` symlink resolves to the newest complete checkpoint.

Colocated online and offline runs restore draft weights, optimizer/scheduler,
epoch/step/data position, and per-rank RNG. Offline disaggregated consumers have
the same checkpoint contract. For a colocated run, override
`training.resume_from`:

```bash
specforge train \
  --config examples/configs/qwen3-8b-eagle3-offline.yaml \
  training.resume_from=./outputs/qwen3-8b-eagle3-offline/qwen3-8b-eagle3-offline-latest
```

For a disaggregated offline resume, reuse the same manifest, feature store, and
run id, then invoke `specforge train -c run.yaml --role consumer` with the
checkpoint override; the producer never accepts a resume checkpoint.

Online disaggregated resume is intentionally consumer-only. Reuse the retained
SQLite metadata DB, original channel/inboxes, Mooncake objects, and matching
checkpoint; rank 0 verifies that the durable optimizer marker equals the
checkpoint step, skips acknowledged refs, and requeues the unacknowledged tail.
The producer itself is not restarted or resumed. Optimizer/FSDP checkpoints
currently require the same trainer world size; control-plane ref redistribution
does not imply optimizer-state resharding.

Training metrics are printed every `training.log_interval` steps and forwarded
to the configured tracking backend.

## Export a trained draft

Runtime checkpoints contain training state and are not serving model
directories. Export the final checkpoint before loading it with SGLang or
Transformers.

For EAGLE3 SGLang serving:

```bash
specforge export --to sglang \
  --checkpoint ./outputs/qwen3-8b-eagle3-online/qwen3-8b-eagle3-online-latest \
  --draft-config configs/qwen3-8b-eagle3.json \
  --output-dir ./exports/qwen3-8b-eagle3-sglang
```

`--to sglang` currently implements the EAGLE3 serving-key contract. Use
`--to hf` for DFlash, Domino, DSpark, and P-EAGLE model directories. For an
EAGLE-family self-contained Hugging Face directory, provide the target model as
the source of the frozen embedding when it is absent from the runtime
checkpoint:

Serving weight names are a fail-silent boundary in SGLang: an unrecognized key
may be skipped while the server still starts. The exporter therefore validates
the required `fc.weight`, `norm.weight`, `lm_head.weight`, `t2d`, and `d2t`
keys and rejects any remaining trainer prefix. `LlamaForCausalLMEagle3` uses an
identity map for its `midlayer.*` and required keys; `embed_tokens.weight` is
deliberately omitted because serving reuses the target model embedding. A new
architecture (including a future MLA draft) must add an explicit, loader-version
matched weight map and required-key contract before SGLang export is enabled.
The export tests cover key structure and tensor round-trip; loading the result
in a real speculative-decoding server and measuring acceptance remains a GPU
serving validation step.

```bash
specforge export --to hf \
  --checkpoint ./outputs/qwen3-8b-eagle3-online/qwen3-8b-eagle3-online-latest \
  --draft-config configs/qwen3-8b-eagle3.json \
  --embedding-source Qwen/Qwen3-8B \
  --output-dir ./exports/qwen3-8b-eagle3-hf
```

Pass `--vocab-mapping /path/to/mapping.pt` when the checkpoint predates the
mapping buffers or when you intentionally need to refresh them.
