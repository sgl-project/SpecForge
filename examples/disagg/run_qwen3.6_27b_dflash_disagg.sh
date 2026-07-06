#!/bin/bash
# Qwen3.6-27B DFlash, ONLINE disaggregated on a single 8-GPU node:
#   producer (27B target inference) -> GPU 0
#   consumer (DFlash draft trainer) -> GPU 1
# sharing a SharedDirFeatureStore + StreamingRefChannel + SQLite metadata store.
# Producer and consumer are separate processes launched concurrently; the
# consumer's loader blocks on the ref channel and terminates on EOF (producer
# close). This is the disaggregated sibling of run_qwen3.6_27b_dflash_online.sh.
#
# Validated on 8xH200 (sci-h200): producer (GPU0) + consumer (GPU1), 400 prompts
# consume-once -> train loss 12.7 -> 7.1 over 400 steps. Curve:
#   examples/disagg/assets/qwen36-27b-dflash-nemotron-disagg.png
#
# Prerequisites (same as the colocated example):
#   - Qwen/Qwen3.6-27B weights + a chat dataset at $ROOT_DIR/cache/dataset/.
#   - For W&B logging on the consumer: export WANDB_API_KEY=<your key>
#     (never hard-code it), or pass --report-to none.
set -uxo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export FLASHINFER_DISABLE_VERSION_CHECK=1
# the launcher does `from train_dflash import ...`, so scripts/ must be importable
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/scripts:${PYTHONPATH:-}"
cd "$ROOT_DIR"

PRODUCER_GPU=${PRODUCER_GPU:-0}
CONSUMER_GPU=${CONSUMER_GPU:-1}

export DISAGG_STORE_ROOT=${DISAGG_STORE_ROOT:-$ROOT_DIR/outputs/qwen36-disagg-store}
export DISAGG_STORE_ID=${DISAGG_STORE_ID:-qwen36-dflash-disagg}
export DISAGG_DB=${DISAGG_DB:-$DISAGG_STORE_ROOT/$DISAGG_STORE_ID.db}
export DISAGG_MAX_PROMPTS=${DISAGG_MAX_PROMPTS:-400}
export DISAGG_MAX_STEPS=${DISAGG_MAX_STEPS:-0}      # 0 = train on all produced
export DISAGG_LOG_INTERVAL=${DISAGG_LOG_INTERVAL:-1}

rm -rf "$DISAGG_STORE_ROOT/$DISAGG_STORE_ID" "$DISAGG_DB"
mkdir -p "$DISAGG_STORE_ROOT/$DISAGG_STORE_ID"
: > "$DISAGG_STORE_ROOT/$DISAGG_STORE_ID/refs.jsonl"

# recipe matches run_qwen3.6_27b_dflash_online.sh; max-length trimmed to 2048 for a
# bounded single-node demo (online disagg is consume-once / single-DP-rank).
ARGS=(
    --target-model-path Qwen/Qwen3.6-27B
    --target-model-backend hf
    --trust-remote-code
    --draft-config-path "$ROOT_DIR/configs/qwen3.6-27b-dflash.json"
    --embedding-key model.language_model.embed_tokens.weight
    --lm-head-key lm_head.weight
    --mask-token-id 248070
    --train-data-path "$ROOT_DIR/cache/dataset/nemotron_v2_train.jsonl"
    --chat-template qwen3.5
    --max-length 2048
    --batch-size 1
    --learning-rate 6e-4
    --warmup-ratio 0.04
    --max-grad-norm 1.0
    --attention-backend flex_attention
    --block-size 16
    --num-anchors 512
    --loss-decay-gamma 7.0
    --num-epochs 1
    --seed 42
    --save-interval 1000000
)

LAUNCHER=$SCRIPT_DIR/run_disagg_dflash.py

# --- producer: inference pool (GPU $PRODUCER_GPU) ---
CUDA_VISIBLE_DEVICES=$PRODUCER_GPU DISAGG_ROLE=producer \
    torchrun --rdzv-backend c10d --rdzv-endpoint localhost:29701 \
        --nnodes 1 --nproc_per_node 1 "$LAUNCHER" "${ARGS[@]}" \
        --output-dir "$ROOT_DIR/outputs/qwen36-disagg-producer" &
PRODUCER_PID=$!

# --- consumer: trainer pool (GPU $CONSUMER_GPU), logs the curve to W&B ---
CUDA_VISIBLE_DEVICES=$CONSUMER_GPU DISAGG_ROLE=consumer \
    torchrun --rdzv-backend c10d --rdzv-endpoint localhost:29702 \
        --nnodes 1 --nproc_per_node 1 "$LAUNCHER" "${ARGS[@]}" \
        --output-dir "$ROOT_DIR/outputs/qwen36-disagg-consumer" \
        --report-to wandb \
        --wandb-project qwen36-dflash-pr645 \
        --wandb-name qwen36-27b-dflash-nemotron-disagg &
CONSUMER_PID=$!

wait $PRODUCER_PID $CONSUMER_PID
echo "DISAGG36-DONE"
