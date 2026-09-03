# Training

SpecForge has one public training entry point for every strategy and runtime
topology:

```bash
specforge train --config examples/configs/online/disaggregated/external/qwen3-8b-eagle3-disaggregated.yaml
```

The YAML file is the run contract. It selects the draft strategy, target model,
data source, optimizer settings, and deployment topology. Method-specific
Python trainers are not part of the public interface.

This is an intentional hard cutover. The old `scripts/train_*.py` commands and
temporary move-only Python import paths were removed rather than deprecated.
Downstream launchers should migrate to a typed run config and `specforge train`;
there is no compatibility dispatch to the previous trainers.

## Choose a recipe

The config catalog separates three concepts:

| Question | Config source | Catalog level |
| --- | --- | --- |
| Are target features captured during training or prepared earlier? | `data.train_data_path` / `data.prompts_path` versus `data.hidden_states_path` | `online/` versus `offline/` |
| Does the trainer read offline hidden states files directly or consume refs from a producer? | `deployment.mode: local_colocated` versus `deployment.mode: disaggregated` | `colocated/` versus `disaggregated/` |
| Who starts Mooncake and SGLang for online disaggregation? | Presence of `deployment.disaggregated.managed_local` | `external/` versus `managed-local/` |

The supported catalog layout is:

```text
examples/configs/
├── offline/
│   ├── colocated/
│   └── disaggregated/
└── online/
    ├── colocated/                 # reserved; currently unsupported
    └── disaggregated/
        ├── external/
        └── managed-local/
```

`external` is an ownership boundary, not a statement that the services are on
another machine. It is `external` when the user starts
the server. `managed-local` owns those services on one host, while producer and
consumer remain separate SpecForge roles. The runtime reads the YAML fields,
not filename suffixes, to determine these semantics. See the complete
[recipe catalog](../../examples/configs/README.md) for representative configs.

## Launch a run

Use the command directly for every checked-in topology:

```bash
specforge train --config examples/configs/online/disaggregated/external/qwen3-8b-eagle3-disaggregated.yaml
```

`deployment.trainer.nproc_per_node` records the audited local process count.
When it is greater than one, the CLI starts torch distributed itself:

```bash
specforge train -c examples/configs/online/disaggregated/external/qwen3-30b-a3b-eagle3-online.yaml
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
  --config examples/configs/online/disaggregated/external/qwen3-8b-eagle3-disaggregated.yaml \
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
with block size 16. This DFlash default selects `DFlashDraftModel`; DFlash2
always requires an explicit draft config or pretrained config source. Typed
overrides are available when creating a different fresh architecture:

```yaml
model:
  target_model_path: Qwen/Qwen3-8B
  draft_num_hidden_layers: 2  # P-EAGLE or DFlash; EAGLE3 remains one layer
  draft_block_size: 8         # DFlash only
```

For DFlash-family drafts, including DFlash2, configure the attention layout in
the referenced draft JSON. Each entry corresponds to one draft layer; sliding
layers share one positive window:

```json
{
  "num_hidden_layers": 5,
  "layer_types": [
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention"
  ],
  "use_sliding_window": true,
  "sliding_window": 2048
}
```

Use `"full_attention"` for every entry, `"use_sliding_window": false`, and
`"sliding_window": null` for a full-only draft. The layout length must equal
`num_hidden_layers`. A layer-count override may resize a uniform layout, but a
mixed layout must be edited explicitly in the draft JSON.

The `eager`, `sdpa`, and `flex_attention` backends support both layouts.

### DFlash2

DFlash2 is a draft-architecture variant of DFlash, not a separate capture
strategy. Keep `training.strategy: dflash` and select it with a draft config
whose architecture is `DFlash2DraftModel`. The target server still captures
the same selected hidden states with `--spec-capture-method dflash`. Online
capture also retains the target's final hidden state for teacher-alignment
telemetry; existing offline DFlash datasets remain compatible and omit only
the teacher-only metrics.

The DFlash2 config additionally defines `conv_kernel_size` and
`conv_group_size` for the local convolution, plus `selector_rank` and
`selector_top_k` for candidate-path selection. The base draft head uses the
configured CE/LK/TV objective. The selector always uses categorical CE over the
target head's strict unary top-k, exactly as it will be used during inference.
If the gold token is outside that candidate set, the token contributes no
selector loss; `selector_coverage` reports how often the gold token is present.
Both objectives receive the configured fixed-decay or D-PACE position weight,
and their combined numerator is normalized by the sum of valid effective token
weights rather than by batch or anchor count.

Set `training.dflash2_selector_loss_alpha` to scale the selector objective.
`training.dflash2_selector_warmup_ratio` keeps that scale at zero for an initial
fraction of optimizer steps, and `training.dflash2_selector_ramp_ratio` then
ramps it linearly to the configured value. Both schedule ratios default to
zero. A configured alpha of zero disables selector training and freezes its
parameters; a positive alpha keeps the selector in the optimizer and autograd
graph even while its effective warmup weight is zero. The convolution starts
from an identity kernel with a zero-initialized dynamic projection, and the
selector starts as a unary no-op, so both additions initially preserve the
corresponding DFlash computation.

Set `training.dflash2_selector_stop_gradient: true` to keep the selector CE
from updating the unary logits and draft hidden states. The selector parameters
still train, and the primary DFlash/D-PACE/LK objective keeps its normal draft
gradient path. The option defaults to `false`, preserving coupled training.

External tracking reports DFlash2 diagnostics on optimizer steps that reach
`training.log_interval`. Aggregates live under `train/dflash2/*`; the same
families are broken down per predicted block position under their own
`position_1/*` through `position_<block_size-1>/*` sections (the verified
anchor is omitted), so each position renders as one dashboard group:

- `hard_label/*`: unary top-1 accuracy, top-K recall and probability mass,
  gold-token probability, selector loss, selector conditional accuracy (uniform
  over covered slots, so comparable across loss types), and the realized greedy
  serving path's per-slot accuracy. Two aggregate-only per-block accepted-length
  proxies count the anchor plus the leading run of supervised slots that are
  covered by the unary top-K (`unary_topK_oracle_accepted_length`) or matched by
  the serving path (`selector_serving_accepted_length`).
- `position_<k>/objective/loss_weight_share`: the fraction of the effective
  objective weight each block position receives under the configured
  fixed-decay or D-PACE weighting.
- `train/objective/lk_kl_weight`: the CE weight of the `lambda` LK objective.
- `teacher/*` (online capture only): full-vocabulary expected acceptance, unary
  top-1 agreement, unary top-K teacher mass, and serving-path agreement against
  the frozen target head.

The exported computation and parameter names match the public SGLang DFlash2
contract, including optional `output_multiplier` and
`final_logit_softcapping` transforms from `dflash_config`.

The checked-in Qwen3.6-27B recipe owns a two-GPU local stack: one configured
GPU runs the target capture server and another runs the trainer. Update its
model/data paths and `cuda_visible_devices` lists for the local host.

```bash
specforge train \
  -c examples/configs/online/disaggregated/managed-local/qwen3.6-27b-dflash2-disaggregated.yaml \
  model.target_model_path=/path/to/Qwen3.6-27B
```

Export the result with `specforge export --to hf`. Serving requires an SGLang
version that includes DFlash2 support (SGLang PR #35371); the serving algorithm
name remains `DFLASH`, and the exported `DFlash2DraftModel` config enables the
new path automatically.

See the dedicated [DFlash2 guide](../concepts/DFlash2.md) for the complete
model-field constraints, offline feature workflow, selector schedule, and
serving compatibility boundary.

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

| Strategy | Category | Config |
| --- | --- | --- |
| EAGLE3 | Online disaggregated, external | [`qwen3-8b-eagle3-disaggregated.yaml`](../../examples/configs/online/disaggregated/external/qwen3-8b-eagle3-disaggregated.yaml) |
| EAGLE3 | Offline colocated | [`qwen3-8b-eagle3-offline.yaml`](../../examples/configs/offline/colocated/qwen3-8b-eagle3-offline.yaml) |
| EAGLE3 | Offline disaggregated | [`qwen3-8b-eagle3-offline-disaggregated.yaml`](../../examples/configs/offline/disaggregated/qwen3-8b-eagle3-offline-disaggregated.yaml) |
| P-EAGLE | Online disaggregated, external | [`qwen3-8b-peagle-disaggregated.yaml`](../../examples/configs/online/disaggregated/external/qwen3-8b-peagle-disaggregated.yaml) |
| DFlash | Online disaggregated, external | [`qwen3-8b-dflash-online.yaml`](../../examples/configs/online/disaggregated/external/qwen3-8b-dflash-online.yaml) |
| DFlash | Online disaggregated, managed-local | [`qwen3-8b-dflash-1server-dp7-disaggregated.yaml`](../../examples/configs/online/disaggregated/managed-local/qwen3-8b-dflash-1server-dp7-disaggregated.yaml) |
| DFlash | Offline colocated | [`qwen3-8b-dflash-offline.yaml`](../../examples/configs/offline/colocated/qwen3-8b-dflash-offline.yaml) |
| DFlash2 | Online disaggregated, managed-local | [`qwen3.6-27b-dflash2-disaggregated.yaml`](../../examples/configs/online/disaggregated/managed-local/qwen3.6-27b-dflash2-disaggregated.yaml) |
| Domino | Online disaggregated, external | [`qwen3-8b-domino-online.yaml`](../../examples/configs/online/disaggregated/external/qwen3-8b-domino-online.yaml) |
| Domino | Online disaggregated, managed-local | [`qwen3-8b-domino-multiserver-disaggregated.yaml`](../../examples/configs/online/disaggregated/managed-local/qwen3-8b-domino-multiserver-disaggregated.yaml) |
| Domino | Offline colocated | [`qwen3-8b-domino-offline.yaml`](../../examples/configs/offline/colocated/qwen3-8b-domino-offline.yaml) |
| DSpark | Online disaggregated, external | [`qwen3-4b-dspark-disaggregated.yaml`](../../examples/configs/online/disaggregated/external/qwen3-4b-dspark-disaggregated.yaml) |
| DSpark | Offline colocated | [`qwen3-4b-dspark-offline.yaml`](../../examples/configs/offline/colocated/qwen3-4b-dspark-offline.yaml) |
| DFlash (Ascend) | Online disaggregated, external | [`qwen3.5-4b-dflash-online-npu.yaml`](../../examples/configs/online/disaggregated/external/qwen3.5-4b-dflash-online-npu.yaml) |
| MTP (Ascend) | Online disaggregated, managed-local | [`qwen3.5-4b-mtp-disaggregated-npu.yaml`](../../examples/configs/online/disaggregated/managed-local/qwen3.5-4b-mtp-disaggregated-npu.yaml) |

## Online and offline data

Online training captures target features while the run is active. It uses
little disk space but keeps target inference available during training.
Offline training reads feature checkpoints generated ahead of time, so only
the draft model must fit on the training GPUs at the cost of substantially
more storage.

| Mode | Target during training | Disk use | Data config |
| --- | --- | --- | --- |
| Online | External or managed-local SGLang capture server| Low | `train_data_path` or `prompts_path` |
| Offline | Not loaded by the trainer | High | `hidden_states_path` |

Prepare raw datasets and offline features as described in [Data
Preparation](data_preparation.md), then update the matching example YAML before
launching it.

## Supported combinations

The unified runtime supports text training in these combinations:

| Strategy | Online disaggregated | Offline colocated | Offline disaggregated |
| --- | --- | --- | --- |
| EAGLE3 | Yes, consumer DP | Yes, DP + USP | Yes, consumer DP |
| DFlash/DFlash2 | Yes, consumer DP | Yes, DP | Yes, consumer DP |
| Domino | Yes, consumer DP | Yes, DP | Yes, consumer DP |
| DSpark | Yes, consumer DP | Yes, DP | Yes, consumer DP |
| MTP | Yes, consumer DP | Yes, DP | Yes, consumer DP |
| P-EAGLE | Yes, consumer DP, batch size 1 | No | No |

Unsupported combinations fail explicitly during config validation or run
assembly. In particular:

- VLM training, including Qwen2.5-VL, is not supported. The unified runtime
  currently accepts text inputs only;
- online evaluation is not supported. Evaluation requires precomputed offline
  features through `data.eval_hidden_states_path`;
- attention backends are strategy-specific: EAGLE3 accepts `sdpa`,
  `flex_attention`, `fa`, or offline `usp`; P-EAGLE requires
  `flex_attention`; DFlash, DFlash2, Domino, and DSpark accept
  `eager`, `sdpa`, or `flex_attention`; MTP accepts `eager` or `sdpa`;
- P-EAGLE requires `training.batch_size=1` and reuses EAGLE3's server capture
  schema;
- offline feature training supports EAGLE3, DFlash, DFlash2, Domino,
  DSpark, and MTP;
- every online run is disaggregated and uses `model.target_backend=sglang`;
  finite runs may omit both step fields so the producer can publish the exact
  optimizer horizon derived from the prepared prompt plan;
- EAGLE3 offline colocated runs derive and cache a deterministic vocabulary mapping
  from the feature corpus when `model.vocab_mapping_path` is empty. EAGLE3
  disaggregated runs require an explicit shared mapping so producer and
  consumer cannot derive different artifacts.

Step limits, LR/loss horizons, logging, saving, and Domino lambda decay are
expressed in completed optimizer updates. Fixed datasets are validated to
contain complete accumulation windows, and finite online plans do not train an
incomplete final quantum. `training.max_steps` is a stop cap and, when set
without `training.total_steps`, the fallback optimizer/loss schedule horizon.
`training.total_steps` can describe a longer schedule, but does not by itself
stop an online stream. When a finite online run omits both, the producer
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
`output_dir/swanlog`. Optimization and evaluator metrics use `train/*` and
`eval/*`; throughput and timing counters use the top-level `perf/*` namespace.
Historical `ploss*`, `acceptance_rate*`, and `target_probability` names remain
available alongside explicit `kl_loss*` or `lk_loss*`, `ce_loss`, and
`expected_acceptance*` names.

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

specforge train -c examples/configs/online/disaggregated/external/qwen3.5-4b-dflash-online-npu.yaml
```

The unified launcher supplies rank, world-size, and rendezvous variables. The
runtime selects the active device dynamically and uses HCCL when `torch_npu`
is active.

## Disaggregated roles

A single-node disaggregated config supervises the SpecForge producer and
consumer with one command. This does not make the run colocated: the roles and
their data-plane contract remain separate. For an external recipe, the command
also does not start Mooncake or SGLang. Split deployments use the same config
with `--role producer` or `--role consumer`; multi-node consumers add only
`--node-rank` on each host. The optional `examples/disagg/run_online.sh` and
`run_offline.sh` files are thin delegates, not topology wrappers. See the
[disaggregated training guide](disaggregated_training.md) for external-service
prerequisites, managed-local ownership, freshness rules, and both launch forms.

## Checkpoints and resume

`training.save_interval` controls checkpoint frequency and
`training.max_checkpoints` controls rotation. Checkpoints are written beneath
`output_dir`. A completed trainer run always saves its final runtime checkpoint,
even when `save_interval` is zero or the final step is not an interval boundary.
The `<run_id>-latest` symlink resolves to the newest complete checkpoint.

Offline colocated runs restore draft weights, optimizer/scheduler, epoch/step/data
position, and per-rank RNG. Offline disaggregated consumers have the same
checkpoint contract. For an offline colocated run, override
`training.resume_from`:

```bash
specforge train \
  --config examples/configs/offline/colocated/qwen3-8b-eagle3-offline.yaml \
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
`--to hf` for DFlash, DFlash2, Domino, DSpark, and P-EAGLE model directories.
For an EAGLE-family self-contained Hugging Face directory, provide the target model as
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

MTP is deployed by merging its trained native head back into the target model.
The merge command accepts the same runtime checkpoint shapes as the generic
exporter (`training_state.pt`, a step/latest directory, or the run output
directory):

```bash
python scripts/merge_mtp_to_base.py \
  --base-model-path Qwen/Qwen3.5-4B \
  --mtp-checkpoint-path ./outputs/qwen3.5-4b-mtp/qwen3.5-4b-mtp-latest \
  --draft-config configs/qwen3.5-4b-mtp.json \
  --output-path ./exports/Qwen3.5-4B-MTP
```

An already-exported HF MTP draft directory can also be passed as
`--mtp-checkpoint-path`; in that case its own `config.json` is used and
`--draft-config` may be omitted.

## Troubleshooting

### Late OOM or non-finite hidden states on online runs

**Symptoms.** An online job starts cleanly, then fails well into training with a
CUDA OOM that reports a large "reserved but unallocated" pool and only a few
MiB free (for example, unable to serve a 388 MiB request with 58.3 GiB reserved
and 34 MiB free). The same fragmentation can also surface first as non-finite
target hidden states or as an anchor-sampling error that names the data rather
than the allocator.

**Cause.** Online training feeds the draft one micro-batch per step whose rows
vary in length, so the caching allocator is asked for a differently sized block
almost every step. Without expandable segments, reserved-but-unallocated memory
accumulates until a modest allocation fails.

**Fix.** Enable expandable segments before launch:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Set this before chasing data or numeric bugs that match the symptoms above. The
Ascend equivalent (`PYTORCH_NPU_ALLOC_CONF`) is already set in the NPU recipes
earlier in this page.
