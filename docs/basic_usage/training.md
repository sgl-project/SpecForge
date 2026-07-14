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

Colocated and offline inputs currently run in one process because they are not
yet data-parallel sharded. Invoke `specforge` directly:

```bash
specforge train --config examples/configs/qwen3-8b-eagle3-online.yaml
```

The online disaggregated consumer is the supported multi-GPU data-parallel
topology and is launched with `torchrun` in the disaggregation section below.

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

A run config has four top-level sections:

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
  deployment_mode: local_colocated
  num_epochs: 10
  batch_size: 1
  learning_rate: 1.0e-4
  save_interval: 1000

run_id: qwen3-8b-eagle3-online
output_dir: ./outputs/qwen3-8b-eagle3-online
```

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
| DSpark disaggregated | [`qwen3-4b-dspark-disaggregated.yaml`](../../examples/configs/qwen3-4b-dspark-disaggregated.yaml) |
| EAGLE3 offline disaggregated | [`qwen3-8b-eagle3-offline-disaggregated.yaml`](../../examples/configs/qwen3-8b-eagle3-offline-disaggregated.yaml) |

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

The unified runtime currently supports text-only training in these
combinations:

| Strategy | Colocated online | Local/dataflow offline | Disaggregated online | Disaggregated offline |
| --- | --- | --- | --- | --- |
| EAGLE3 | Yes, 1 rank | Yes, 1 rank | Yes, consumer DP | Yes, 1 rank |
| DFlash | Yes, 1 rank | No | Yes, consumer DP | No |
| Domino | Yes, 1 rank | No | Yes, consumer DP | No |
| DSpark | No | No | Yes, consumer DP | No |
| P-EAGLE | Yes, 1 rank and batch size 1 | No | No | No |

Unsupported combinations fail explicitly during config validation or run
assembly. In particular:

- multimodal/VLM input is not accepted by the typed run schema;
- attention backends are strategy-specific: EAGLE3 accepts `sdpa`,
  `flex_attention`, or `fa`; P-EAGLE requires `flex_attention`; DFlash,
  Domino, and DSpark accept `eager`, `sdpa`, or `flex_attention`;
- P-EAGLE and online EAGLE3 require `training.batch_size=1`;
- DSpark requires disaggregated server capture;
- offline features currently use the EAGLE3 feature contract only;
- every online disaggregated run uses `model.target_backend=sglang` and sets
  either `training.total_steps` or `training.max_steps`;
- EAGLE3 offline and EAGLE3 disaggregated runs require
  `model.vocab_mapping_path`.

There is no fallback to a removed training script.

Step limits are global optimizer updates. `training.max_steps` is a stop cap
and becomes the optimizer/loss schedule horizon when `training.total_steps` is
omitted. `training.total_steps` can describe a longer schedule, but does not by
itself stop an online stream.

## Disaggregated roles

A disaggregated run starts the same typed config once per role. The repository
provides one topology-specific wrapper for online server capture and one for
offline feature ingestion: `examples/disagg/run_online.sh` and
`examples/disagg/run_offline.sh`.

These scripts validate the environment and dispatch to the same
`specforge train` entry point; they are not additional trainers. See the
[disaggregated training guide](disaggregated_training.md) for the patched
SGLang command, complete environment contract, fresh-attempt rules, and both
launch procedures.

## Checkpoints and resume

`training.save_interval` controls checkpoint frequency and
`training.max_checkpoints` controls rotation. Checkpoints are written beneath
`output_dir`. A completed trainer run always saves its final runtime checkpoint,
even when `save_interval` is zero or the final step is not an interval boundary.
The `<run_id>-latest` symlink resolves to the newest complete checkpoint.

Resume is supported only by a single-rank offline trainer: either a colocated
offline run or the consumer of a disaggregated offline run. For a colocated
run, override `training.resume_from`:

```bash
specforge train \
  --config examples/configs/qwen3-8b-eagle3-offline.yaml \
  training.resume_from=./outputs/qwen3-8b-eagle3-offline/qwen3-8b-eagle3-offline-latest
```

For a disaggregated offline resume, reuse the same manifest, feature store, and
run id, then pass the override to `run_offline.sh consumer`; the producer never
accepts a resume checkpoint. No online run accepts `training.resume_from` in
this PR, including colocated online and disaggregated online runs. Runtime
checkpoints still contain training state and can be exported, but a new online
attempt must start with a fresh run, channel, and coordination database.

Training metrics are printed every `training.log_interval` steps.

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

```bash
specforge export --to hf \
  --checkpoint ./outputs/qwen3-8b-eagle3-online/qwen3-8b-eagle3-online-latest \
  --draft-config configs/qwen3-8b-eagle3.json \
  --embedding-source Qwen/Qwen3-8B \
  --output-dir ./exports/qwen3-8b-eagle3-hf
```

Pass `--vocab-mapping /path/to/mapping.pt` when the checkpoint predates the
mapping buffers or when you intentionally need to refresh them.
