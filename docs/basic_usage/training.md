# Training

SpecForge has one public training entry point for every strategy and runtime
topology:

```bash
specforge train --config examples/configs/qwen3-8b-eagle3-disaggregated.yaml
```

The YAML file is the run contract. It selects the draft strategy, target model,
data source, optimizer settings, and deployment mode. Method-specific Python
trainers are not part of the public interface.

This is an intentional hard cutover. The old `scripts/train_*.py` commands and
temporary move-only Python import paths were removed rather than deprecated.
Downstream launchers should migrate to a typed run config and `specforge train`;
there is no compatibility dispatch to the previous trainers.

### Defaults when migrating removed trainers

The typed schema defaults existed before the old trainers were removed, but
they are not identical to defaults embedded in every deleted script. If an old
launch omitted these flags, write the legacy value explicitly in its new YAML
when reproducing that run:

| Removed CLI default | Typed run field and default | Legacy value to preserve |
| --- | --- | --- |
| DFlash/Domino `--num-epochs=6` | `training.num_epochs: 1` | `6` |
| DFlash/Domino `--learning-rate=6e-4` | `training.learning_rate: 1e-4` | `6e-4` |
| DFlash/Domino `--warmup-ratio=0.04` | `training.warmup_ratio: 0.015` | `0.04` |
| DFlash/Domino `--max-grad-norm=1.0` | `training.max_grad_norm: 0.5` | `1.0` |
| DFlash/Domino `--max-length=3072` | `data.max_length: 2048` | `3072` |
| DFlash/Domino `--chat-template=qwen` | `data.chat_template: llama3` | `qwen` |
| DFlash/Domino `--save-interval=1000` | `training.save_interval: 0` | `1000` |
| DFlash/Domino `--dist-timeout=30` | `training.dist_timeout: 10` | `30` |
| EAGLE3 `--kl-decay=3.0` | `training.kl_decay: 1.0` | `3.0` |

The old DFlash/Domino `--eval-interval=1000` did not identify an evaluation
source by itself. In the unified runtime, evaluation is deliberately off by
default and must be paired with `data.eval_hidden_states_path`.

Two numerical lifecycle details are also deliberate. All unified FSDP methods
keep buffers in float32; the removed EAGLE3 and DFlash scripts used bfloat16
buffers, while the removed Domino script already used float32. Consequently,
bit-for-bit comparisons to old EAGLE3/DFlash baselines must account for that
dtype change. Also, `global_step`, LR/loss horizons, logging, saving, and Domino
lambda decay are all expressed in completed optimizer updates. Fixed datasets
are validated before backend/optimizer assembly to contain complete accumulation windows;
finite online plans train only complete global optimizer quanta. The old
scripts mixed micro-batch counters with a ceil-derived optimizer horizon, so
accumulation greater than one did not have the same boundary semantics.

## Launch a run

Use the command directly for every checked-in topology:

```bash
specforge train --config examples/configs/qwen3-8b-eagle3-disaggregated.yaml
```

`deployment.trainer.nproc_per_node` records the audited local process count.
When it is greater than one, the CLI starts torch distributed itself:

```bash
specforge train -c examples/configs/qwen3-30b-a3b-eagle3-online.yaml
```

Online target inference never runs in the trainer. A patched SGLang server owns
target parallelism and publishes captured features through Mooncake; every
consumer rank is data parallel. Offline runs shard fixed feature references
across trainer ranks and may additionally use EAGLE3 USP. See
[Parallel topologies](#parallel-topologies) for the exact constraints.

Paths in a config are resolved from the current working directory. The example
configs assume that the command is run from the repository root.

You can override an existing value without copying the YAML. Overrides use
validated `section.field=value` syntax:

```bash
specforge train \
  --config examples/configs/qwen3-8b-eagle3-disaggregated.yaml \
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
  vocab_mapping_path: cache/vocab_mapping/qwen3-8b.pt
  torch_dtype: bfloat16

data:
  train_data_path: ./cache/dataset/sharegpt_train.jsonl
  max_length: 4096
  chat_template: qwen
  cache_dir: ./cache

training:
  strategy: eagle3
  num_epochs: 10
  max_steps: 10000
  batch_size: 1
  learning_rate: 1.0e-4
  save_interval: 1000

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
| EAGLE3 online | [`qwen3-8b-eagle3-disaggregated.yaml`](../../examples/configs/qwen3-8b-eagle3-disaggregated.yaml) |
| EAGLE3 offline | [`qwen3-8b-eagle3-offline.yaml`](../../examples/configs/qwen3-8b-eagle3-offline.yaml) |
| DFlash online | [`qwen3-8b-dflash-online.yaml`](../../examples/configs/qwen3-8b-dflash-online.yaml) |
| DFlash offline | [`qwen3-8b-dflash-offline.yaml`](../../examples/configs/qwen3-8b-dflash-offline.yaml) |
| Domino online | [`qwen3-8b-domino-online.yaml`](../../examples/configs/qwen3-8b-domino-online.yaml) |
| Domino offline | [`qwen3-8b-domino-offline.yaml`](../../examples/configs/qwen3-8b-domino-offline.yaml) |
| P-EAGLE online | [`qwen3-8b-peagle-disaggregated.yaml`](../../examples/configs/qwen3-8b-peagle-disaggregated.yaml) |
| DFlash disaggregated | [`qwen3-8b-dflash-disaggregated.yaml`](../../examples/configs/qwen3-8b-dflash-disaggregated.yaml) |
| Domino disaggregated | [`qwen3-8b-domino-disaggregated.yaml`](../../examples/configs/qwen3-8b-domino-disaggregated.yaml) |
| DSpark disaggregated | [`qwen3-4b-dspark-disaggregated.yaml`](../../examples/configs/qwen3-4b-dspark-disaggregated.yaml) |
| DSpark offline | [`qwen3-4b-dspark-offline.yaml`](../../examples/configs/qwen3-4b-dspark-offline.yaml) |
| EAGLE3 offline disaggregated | [`qwen3-8b-eagle3-offline-disaggregated.yaml`](../../examples/configs/qwen3-8b-eagle3-offline-disaggregated.yaml) |
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
| Online | External/managed SGLang capture server | Low | `train_data_path` or `prompts_path` |
| Offline | Not loaded by the trainer | High | `hidden_states_path` |

Prepare raw datasets and offline features as described in [Data
Preparation](data_preparation.md), then update the matching example YAML before
launching it.

## Supported combinations

The unified runtime supports text training in these combinations:

| Strategy | SGLang server online | Local/dataflow offline | Disaggregated offline |
| --- | --- | --- | --- |
| EAGLE3 | Yes, consumer DP | Yes, DP + USP | Yes, consumer DP |
| DFlash | Yes, consumer DP | Yes, DP | Yes, consumer DP |
| Domino | Yes, consumer DP | Yes, DP | Yes, consumer DP |
| DSpark | Yes, consumer DP | Yes, DP | Yes, consumer DP |
| P-EAGLE | Yes, consumer DP, batch size 1 | No | No |

Unsupported combinations fail explicitly during config validation or run
assembly. In particular:

- VLM training, including Qwen2.5-VL, is not supported. The unified runtime
  currently accepts text inputs only;
- online evaluation is not supported. Evaluation requires precomputed offline
  features through `data.eval_hidden_states_path`;
- attention backends are strategy-specific: EAGLE3 accepts `sdpa`,
  `flex_attention`, `fa`, or offline `usp`; P-EAGLE requires
  `flex_attention`; DFlash, Domino, and DSpark accept `eager`, `sdpa`, or
  `flex_attention`;
- P-EAGLE requires `training.batch_size=1` and reuses EAGLE3's server capture
  schema;
- offline feature training supports EAGLE3, DFlash, Domino, and DSpark;
- every online run is disaggregated and uses `model.target_backend=sglang`;
  finite runs may omit both step fields so the producer can publish the exact
  optimizer horizon derived from the prepared prompt plan;
- EAGLE3 local offline runs derive and cache a deterministic vocabulary mapping
  from the feature corpus when `model.vocab_mapping_path` is empty. EAGLE3
  disaggregated runs require an explicit shared mapping so producer and
  consumer cannot derive different artifacts.

There is no fallback to a removed training script.

Step limits are global optimizer updates. `training.max_steps` is a stop cap and,
when set without `training.total_steps`, the fallback optimizer/loss schedule
horizon. `training.total_steps` can describe a longer schedule, but does not by
itself stop an online stream. When a finite online run omits both, the producer
publishes the exact schedule horizon and the consumer trains to EOF.

## Parallel topologies

The launcher creates every process group from the typed run config:

- Online target TP/EP belongs to each external SGLang capture server, not the
  trainer. Online consumers keep `training.tp_size` and both SP sizes at 1;
  every trainer rank receives a disjoint feature stream.
- Offline consumers also keep `training.tp_size` at 1. Without USP, every
  trainer rank receives a disjoint reference shard and participates as data
  parallelism.
- EAGLE3 offline can set `training.attention_backend: usp` and choose
  `training.sp_ulysses_size` and `training.sp_ring_size`. Their product must be
  greater than one, USP currently uses `training.batch_size: 1`, and SP peers
  share one sequence while draft-DP groups receive disjoint references.

The world size must be divisible by
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

Offline evaluation is configured through the same YAML:

```yaml
data:
  hidden_states_path: ./cache/hidden_states/train
  eval_hidden_states_path: ./cache/hidden_states/eval

training:
  eval_interval: 100
```

The evaluation source and `training.eval_interval` must be set together.
Online evaluation is not supported; setting `data.eval_data_path` fails config
validation. Offline evaluation uses the same feature reader and collator as
training and retains the final partial batch. Metrics are emitted under
`eval/*`.

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
online and non-EAGLE3 runs.

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
The checked-in NPU recipes use an external NPU-compatible SGLang capture server
and SDPA consumers. A four-device launch is:

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

Local offline runs restore draft weights, optimizer/scheduler, epoch/step/data
position, and per-rank RNG. Offline disaggregated consumers have the same
checkpoint contract. For a local offline run, override
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
  --checkpoint ./outputs/qwen3-8b-eagle3-disaggregated/qwen3-8b-eagle3-disaggregated-latest \
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
  --checkpoint ./outputs/qwen3-8b-eagle3-disaggregated/qwen3-8b-eagle3-disaggregated-latest \
  --draft-config configs/qwen3-8b-eagle3.json \
  --embedding-source Qwen/Qwen3-8B \
  --output-dir ./exports/qwen3-8b-eagle3-hf
```

Pass `--vocab-mapping /path/to/mapping.pt` when the checkpoint predates the
mapping buffers or when you intentionally need to refresh them.
