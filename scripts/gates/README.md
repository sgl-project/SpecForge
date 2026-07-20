# Training and serving gates

These generic gates exercise the unified runtime without adding another
training entry. Training is launched only through the canonical
`examples/disagg/run_online.sh` wrapper, which invokes `specforge train`.
Method-specific Python trainers are not required.

## One-command disaggregated overfit and serving gate

Run the full Domino-compatible DFlash-family gate with one local command:

```bash
CONFIG=examples/configs/qwen3-8b-domino-disaggregated.yaml \
TARGET_MODEL_PATH=Qwen/Qwen3-8B \
DRAFT_CONFIG_PATH=configs/qwen3-8b-domino.json \
SOURCE_DATA_PATH=./cache/dataset/sharegpt_train.jsonl \
REASONING_POLICY=forbidden ENABLE_THINKING=false \
NPROC_PER_NODE=7 \
CAPTURE_GPUS=0 \
CONSUMER_GPUS=1,2,3,4,5,6,7 \
bash scripts/gates/run_disaggregated_overfit_gate.sh
```

The gate selects one untruncated sample, starts Mooncake and a patched SGLang
capture server, launches the canonical producer and consumer roles, checks the
strict loss/accuracy target and exact final checkpoint, releases the capture
stack, exports through `specforge export --to hf`, starts real DFLASH serving,
and validates acceptance metadata plus target-prefix agreement. EXIT, INT, and
TERM traps stop every process owned by the gate; externally supervised services
are never registered for cleanup.

## Model and reasoning contracts

The gate is model-parameterized, but the dataset contract must match the target
chat template and the selected reasoning mode:

| Target | Dataset contract | `CHAT_TEMPLATE` | Reasoning controls |
| --- | --- | --- | --- |
| Qwen3-8B | Non-reasoning responses | `qwen` | `REASONING_POLICY=forbidden`, `ENABLE_THINKING=false` |
| Qwen3.6-27B | Structured reasoning responses | `qwen3.5` | `REASONING_POLICY=required`, `ENABLE_THINKING=true` |

For example, the Qwen3.6 reasoning gate still uses the same canonical training
entry; the selected YAML supplies the disaggregated topology while the gate
passes the model, draft, data, and bounded-run values as typed overrides:

```bash
CONFIG=examples/configs/qwen3-8b-domino-disaggregated.yaml \
TARGET_MODEL_PATH=Qwen/Qwen3.6-27B \
DRAFT_CONFIG_PATH=configs/qwen3.6-27b-domino-full-attention.json \
SOURCE_DATA_PATH=/path/to/validated_qwen36_reasoning.jsonl \
CHAT_TEMPLATE=qwen3.5 \
EMBEDDING_KEY=model.language_model.embed_tokens.weight \
REASONING_POLICY=required ENABLE_THINKING=true MAX_LENGTH=2048 \
CAPTURE_GPUS=0 CAPTURE_TP=1 \
CONSUMER_GPUS=1,2,3,4,5,6,7 NPROC_PER_NODE=7 \
bash scripts/gates/run_disaggregated_overfit_gate.sh
```

Mask token, block size, and auxiliary capture layers are read from the draft
config. The selector rejects rows that violate the requested reasoning policy,
would be truncated, or have fewer than `2 * block_size` trainable tokens after
the real tokenizer and preprocessing path. It writes a prompt artifact with the
exact rendered target suffix; the serving check consumes that artifact rather
than trying to reconstruct structured reasoning from visible content.

The training stage repeats exactly one selected row for the bounded optimizer
run and succeeds only when all of these conditions hold:

- the log reaches exactly `MAX_STEPS`;
- final loss is at most `MAX_LOSS` (`0.0001` by default);
- final token accuracy is at least `MIN_ACCURACY` (`1.0` by default);
- the exact final `training_state.pt` checkpoint exists.

The real-serving stage then exports that checkpoint, launches SGLang with the
`DFLASH` speculative algorithm, and checks one `/v1/chat/completions` response.
Request history must contain no `reasoning_content`, per-choice metadata must
report `spec_accept_length >= block_size`, and generated output (reasoning plus
visible content) must match at least the first `block_size` target tokens. The
auditable response, request, server metadata, and prefix counts are written to
`serving-gate.json`; aggregate server statistics are not accepted in place of
per-request metadata.

The required environment is `CONFIG`, `TARGET_MODEL_PATH`,
`DRAFT_CONFIG_PATH`, and `SOURCE_DATA_PATH`. Common controls are:

| Variable | Purpose | Default |
| --- | --- | --- |
| `WORK_DIR` | Fresh directory for all artifacts; an existing path is rejected | `outputs/<run-id>` |
| `RUN_ID` | Producer/consumer run ID and checkpoint prefix | timestamped |
| `MAX_STEPS` | Exact optimizer-step and checkpoint target | `400` |
| `NPROC_PER_NODE` | Local consumer DP ranks | `1` |
| `CAPTURE_GPUS`, `CAPTURE_TP`, `CAPTURE_PORT` | Capture server placement and topology | `0`, `1`, `30000` |
| `CONSUMER_GPUS` | Consumer CUDA devices | `1` |
| `START_MOONCAKE` | Reuse an external Mooncake endpoint when `false` | `true` |
| `START_CAPTURE_SERVER` | Reuse an external capture server when `false` | `true` |
| `RUN_SERVING_GATE` | Chain export and real serving after overfit | `true` |
| `REASONING_POLICY` | Require, forbid, or allow structured reasoning rows | `allow` |
| `ENABLE_THINKING` | Preserve thinking mode in the prompt artifact and serving request | `false` |
| `MAX_LOSS`, `MIN_ACCURACY` | Strict final training thresholds | `0.0001`, `1.0` |

Mooncake addresses, disaggregated store paths, health polling, serving ports,
GPU placement, and SGLang extra arguments can also be overridden through the
environment; use `--help` for the primary contract and inspect the named
variables in the script for deployment-specific tuning.

Use dry-run mode to validate paths and inspect shell-safe command construction
without creating directories or starting services:

```bash
GATE_DRY_RUN=1 \
CONFIG=examples/configs/qwen3-8b-domino-disaggregated.yaml \
TARGET_MODEL_PATH=Qwen/Qwen3-8B \
DRAFT_CONFIG_PATH=configs/qwen3-8b-domino.json \
SOURCE_DATA_PATH=./cache/dataset/sharegpt_train.jsonl \
bash scripts/gates/run_disaggregated_overfit_gate.sh
```

## Standalone real DFLASH serving gate

An existing DFlash-family checkpoint and prompt artifact can be gated
independently:

```bash
CHECKPOINT_PATH=./outputs/run/run-step400 \
TARGET_MODEL_PATH=Qwen/Qwen3-8B \
DRAFT_CONFIG_PATH=configs/qwen3-8b-domino.json \
PROMPT_ARTIFACT_PATH=./outputs/run/prompt.json \
SERVING_GPUS=0 \
bash scripts/gates/run_dflash_serving_gate.sh
```

This orchestration owns unified export, export-config normalization, SGLang
launch and health checking, strict endpoint validation, and cleanup. Set
`GATE_DRY_RUN=1` to print the export, server, and checker commands without side
effects.

## Reusable leaf checks

`select_overfit_sample.py`, `check_overfit_metrics.py`, and
`run_dflash_chat_serving_gate.py` remain dependency-light leaf helpers for
targeted debugging. The chat checker itself does not manage services; the two
shell orchestrators above own service launch, health checks, and cleanup.
