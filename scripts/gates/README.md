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
