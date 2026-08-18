#!/usr/bin/env bash
# Two-node Inkling DSpark recipe:
#   rank 0: Mooncake + SGLang #31847 TP4 capture + CPU producer
#   rank 1: four-rank FSDP consumer/trainer
#
# Launch this command on both nodes. The cluster launcher supplies
# RCLI_NODE_RANK, RCLI_NUM_NODES, and RCLI_HEAD_IP; both nodes must share the
# fresh DISAGG_RUN_ROOT. Install SGLang #31847 in the active environment; the
# shared launcher applies the checked-in SpecForge capture patch before start.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

export CONFIG="${CONFIG:-$ROOT_DIR/examples/configs/inkling-dspark-disaggregated.yaml}"
export RUN_LABEL="${RUN_LABEL:-inkling-dspark-2node}"
export TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-thinkingmachines/Inkling}"

export SERVER_GPUS="${SERVER_GPUS:-0,1,2,3}"
export SERVER_TP="${SERVER_TP:-4}"
export SERVER_MEM_FRACTION="${SERVER_MEM_FRACTION:-0.85}"
export CAPTURE_LAYER_IDS="${CAPTURE_LAYER_IDS:-5 17 35 47 59}"

export TRAINER_GPUS="${TRAINER_GPUS:-0,1,2,3}"
export TRAINER_NPROC="${TRAINER_NPROC:-4}"
TRAINER_ACCUMULATION_STEPS="${TRAINER_ACCUMULATION_STEPS:-128}"

export APPLY_SGLANG_CAPTURE_PATCH="${APPLY_SGLANG_CAPTURE_PATCH:-1}"
export SGLANG_ENABLE_UNIFIED_RADIX_TREE="${SGLANG_ENABLE_UNIFIED_RADIX_TREE:-1}"
export SGLANG_OPT_USE_INKLING_CUSTOM_AR="${SGLANG_OPT_USE_INKLING_CUSTOM_AR:-1}"

DEFAULT_SERVER_EXTRA_ARGS="--dtype bfloat16 --attention-backend fa4"
DEFAULT_SERVER_EXTRA_ARGS+=" --context-length 4103 --quantization modelopt_fp4"
DEFAULT_SERVER_EXTRA_ARGS+=" --moe-runner-backend flashinfer_trtllm_routed"
DEFAULT_SERVER_EXTRA_ARGS+=" --page-size 128"
DEFAULT_SERVER_EXTRA_ARGS+=" --mamba-radix-cache-strategy extra_buffer"
DEFAULT_SERVER_EXTRA_ARGS+=" --max-mamba-cache-size 64"
DEFAULT_SERVER_EXTRA_ARGS+=" --swa-full-tokens-ratio 0.2"
export SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS:-$DEFAULT_SERVER_EXTRA_ARGS}"

exec "$SCRIPT_DIR/run_qwen3_8b_dflash_disagg_2node.sh" \
    "training.accumulation_steps=$TRAINER_ACCUMULATION_STEPS" \
    "$@"
