#!/usr/bin/env bash
# Two-node Qwen3.8-27B DFlash2 recipe:
#   rank 0: Mooncake + eight SGLang v0.5.18 TP1 capture servers + CPU producer
#   rank 1: eight-rank FSDP consumer/trainer
#
# Launch this command on both nodes. The cluster launcher supplies
# RCLI_NODE_RANK, RCLI_NUM_NODES, and RCLI_HEAD_IP; both nodes must share the
# fresh DISAGG_RUN_ROOT. Install SGLang v0.5.18 in the active environment; the
# shared launcher applies the checked-in SpecForge capture patch before start.
# Throughput, split and flow-control notes:
# docs/recipes/qwen3.8-27b-dflash2-disaggregated.md
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

export CONFIG="${CONFIG:-$ROOT_DIR/examples/configs/online/disaggregated/external/qwen3.8-27b-dflash2-disaggregated.yaml}"
export RUN_LABEL="${RUN_LABEL:-qwen3.8-27b-dflash2-2node}"
export TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-Qwen/Qwen3.8-27B}"

export SERVER_COUNT="${SERVER_COUNT:-8}"
export SERVER_GPUS="${SERVER_GPUS:-0,1,2,3,4,5,6,7}"
export SERVER_TP="${SERVER_TP:-1}"
export SERVER_MEM_FRACTION="${SERVER_MEM_FRACTION:-0.85}"
# Auxiliary capture layers of configs/qwen3.8-27b-dflash2.json.
export CAPTURE_LAYER_IDS="${CAPTURE_LAYER_IDS:-5 19 33 47 61}"

export TRAINER_GPUS="${TRAINER_GPUS:-0,1,2,3,4,5,6,7}"
export TRAINER_NPROC="${TRAINER_NPROC:-8}"

# One 8192-token sample is ~500 MB of BF16 hidden states. The master lease
# must outlive a fetch of that size under eight-rank contention, and every
# capture server contributes a Store segment sized for its share of the
# in-flight payload (512 refs x ~500 MB across eight servers).
export MOONCAKE_DEFAULT_KV_LEASE_TTL="${MOONCAKE_DEFAULT_KV_LEASE_TTL:-10000}"
export MOONCAKE_GLOBAL_SEGMENT_SIZE="${MOONCAKE_GLOBAL_SEGMENT_SIZE:-$((64 << 30))}"
export MOONCAKE_LOCAL_BUFFER_SIZE="${MOONCAKE_LOCAL_BUFFER_SIZE:-$((4 << 30))}"

export APPLY_SGLANG_CAPTURE_PATCH="${APPLY_SGLANG_CAPTURE_PATCH:-1}"

# Same server settings as the recipe's model.sglang_* fields.
DEFAULT_SERVER_EXTRA_ARGS="--dtype bfloat16 --attention-backend triton"
DEFAULT_SERVER_EXTRA_ARGS+=" --disable-radix-cache --context-length 8199"
export SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS:-$DEFAULT_SERVER_EXTRA_ARGS}"

exec "$SCRIPT_DIR/run_qwen3_8b_dflash_disagg_2node.sh" "$@"
