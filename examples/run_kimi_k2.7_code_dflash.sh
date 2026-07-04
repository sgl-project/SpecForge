#!/bin/bash

# The script assumes a 2-node topology
#
# 1. ssh from your LOCAL machine into EACH node (no node->node ssh needed); the repo lives on a shared FS both nodes mount.
# 2. On EACH node, run once: `bash examples/run_kimi_k2.7_code_dflash.sh setup`
# 3. On rank-0 node, run once: `bash examples/run_kimi_k2.7_code_dflash.sh prepare`
# 4. run on EACH node in its own ssh session, inside tmux so it survives disconnect.
#   * rank 0:  NODE_RANK=0 MASTER_ADDR=<rank0-host> bash examples/run_kimi_k2.7_code_dflash.sh watchdog
#   * rank 1:  NODE_RANK=1 MASTER_ADDR=<rank0-host> bash examples/run_kimi_k2.7_code_dflash.sh watchdog
#
# KEY ENV (all overridable):
#   NNODES=2 
#   NUM_GPUS=8 
#   MASTER_ADDR=<rank0 host> 
#   MASTER_PORT=29500
#   BATCH_SIZE=4 
#   LEARNING_RATE=6e-4 
#   MEM_FRACTION=0.6 
#   REPORT_TO=tensorboard
#   WANDB_API_KEY=<...>  (for `REPORT_TO=wandb`)
#   HF_TOKEN=<...>   (for `prepare`)


if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/$(basename "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
ROOT_DIR=$(dirname "$SCRIPT_DIR")

# ---- knobs ----------------------------------------------------------------
NNODES=${NNODES:-2}
NUM_GPUS=${NUM_GPUS:-8}
WORLD=$((NNODES * NUM_GPUS))               # tp = ep = dp = WORLD (attention dp)
MASTER_ADDR=${MASTER_ADDR:-}
MASTER_PORT=${MASTER_PORT:-29500}
BATCH_SIZE=${BATCH_SIZE:-1}
LEARNING_RATE=${LEARNING_RATE:-6e-4}
NUM_EPOCHS=${NUM_EPOCHS:-6}
LOG_INTERVAL=${LOG_INTERVAL:-50}
SAVE_INTERVAL=${SAVE_INTERVAL:-10000}   # checkpoint every N steps (lower to verify save/resume)
EVAL_INTERVAL=${EVAL_INTERVAL:-10000}   # online eval every N steps (lower to verify eval)
MEM_FRACTION=${MEM_FRACTION:-0.5}

# torchrun in-place worker restarts. Keep 0: on a 2-node job, an in-place restart
# on one node desyncs from the other and cascades into gloo "connection refused"
# storms. With 0, any fault makes torchrun exit cleanly (one readable traceback);
# the watchdog then does a *coordinated whole-job* relaunch. Raise only once stable.
MAX_RESTARTS=${MAX_RESTARTS:-0}

REPORT_TO=${REPORT_TO:-tensorboard}
TARGET_MODEL=${TARGET_MODEL:-moonshotai/Kimi-K2.7-Code}
DATA_DIR=$ROOT_DIR/cache/dataset/nemotron-post-training-v2
OUTPUT_DIR=$ROOT_DIR/outputs/kimi-k2.7-code-dflash-nemotron
DRAFT_CONFIG=$ROOT_DIR/configs/kimi-k2.7-code-dflash.json
MAX_LENGTH=${MAX_LENGTH:-4096}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-kimi-k2.5-instruct}

# Draft attention backend (reference recipe = flex_attention). The draft's short
# query length used to route flex to its decoding kernel, which fails inductor
# autotune on torch 2.x; specforge/modeling/draft/dflash.py now forces the main
# flex kernel (FORCE_USE_FLEX_ATTENTION), so flex_attention works again. `sdpa`
# remains a correct fallback (create_dflash_sdpa_mask) if ever needed.
ATTENTION_BACKEND=${ATTENTION_BACKEND:-flex_attention}

# c10d rendezvous id. MUST be identical on both nodes of one launch. A FIXED id
# lets a stale agent from a previous (killed-but-not-dead) launch silently merge
# into a fresh run with mismatched config (e.g. different BATCH_SIZE) -> the two
# nodes disagree on step/eval counts -> collective deadlock. If you ever suspect
# that, set a FRESH RDZV_ID (same value on both nodes) to force an isolated
# rendezvous, e.g. RDZV_ID=kimi-$(date +%s) on a shared launch.
RDZV_ID=${RDZV_ID:-kimi-k2.7-code-dflash}

export HF_HOME=${HF_HOME:-/cluster-storage/models}

log() { echo "[$(date -u +%FT%TZ)] $*"; }

cmd_setup() {
  cd "$ROOT_DIR"
  pip install --no-deps -e .
  pip install accelerate tensorboard yunchang qwen-vl-utils
  python3 - <<'PY'
import importlib, sys
miss=[m for m in ["torch","sglang","transformers","datasets","accelerate",
                  "yunchang","flash_attn","deep_ep","tensorboard","specforge"]
      if (importlib.util.find_spec(m) is None)]
if miss: print("PREFLIGHT FAILED, missing:", miss); sys.exit(1)
import torch, sglang
print(f"PREFLIGHT OK  torch={torch.__version__}  sglang={sglang.__version__}")
PY
}

cmd_prepare() {
  # Run ONCE on shared storage (visible to both nodes). Idempotent.
  mkdir -p "$DATA_DIR" "$OUTPUT_DIR"
  local train="$DATA_DIR/nemotron_v2_train.jsonl"
  local eval="$DATA_DIR/nemotron_v2_eval.jsonl"
  local eval2k="$DATA_DIR/nemotron_v2_eval_2k.jsonl"

  if [ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]; then
    log "prepare[1/4]: downloading $TARGET_MODEL (~595 GB, idempotent)"
    [ -n "${HF_TOKEN:-}" ] || log "  WARNING: HF_TOKEN unset; gated download may fail."
    python3 - "$TARGET_MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
print("  ->", snapshot_download(repo_id=sys.argv[1]))
PY
  else
    log "prepare[1/4]: skipping model download (SKIP_MODEL_DOWNLOAD=1)"
  fi

  if [ -f "$train" ] && [ -f "$eval" ]; then
    log "prepare[2/4]: jsonl present, skipping"
  else
    log "prepare[2/4]: building Nemotron v2 jsonl"
    python3 "$ROOT_DIR/scripts/prepare_nemotron_post_training_v2.py" --output-dir "$DATA_DIR"
  fi

  if [ -f "$eval2k" ]; then
    log "prepare[3/4]: eval-2k present, skipping"
  else
    log "prepare[3/4]: carving deterministic 2k eval subset"
    head -n "${EVAL2K_SIZE:-2000}" "$eval" > "$eval2k"
  fi

  if [ "${SKIP_TOKENIZE_WARMUP:-0}" != "1" ]; then
    log "prepare[4/4]: warming tokenized train cache (avoids cross-node race)"
    python3 - "$train" "$MAX_LENGTH" "$CHAT_TEMPLATE" "$TARGET_MODEL" "$ROOT_DIR/cache" <<'PY'
import hashlib, os, sys
from transformers import AutoTokenizer
from datasets import load_dataset
from specforge.data import build_eagle3_dataset
train, max_len, tmpl, model, cache = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
key = hashlib.md5(f"{train}-{max_len}-{tmpl}-{model}".encode()).hexdigest()
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
ds = load_dataset("json", data_files=train)["train"]
build_eagle3_dataset(dataset=ds, tokenizer=tok, chat_template=tmpl, max_length=max_len,
                     cache_dir=os.path.join(cache, "processed_dataset"), cache_key=key,
                     num_proc=int(os.environ.get("SPECFORGE_DATA_NUM_PROC", 64)))
print("  tokenized train cache ready:", key)
PY
  else
    log "prepare[4/4]: skipping tokenize warm-up (SKIP_TOKENIZE_WARMUP=1)"
  fi
  log "prepare: done. Now launch per node (in each node's ssh session, inside tmux):"
  log "  rank0:  NODE_RANK=0 MASTER_ADDR=<rank0-host> bash $0 watchdog"
  log "  rank1:  NODE_RANK=1 MASTER_ADDR=<rank0-host> bash $0 watchdog"
}

cmd_train() {
  # Run on ONE node. NODE_RANK / NNODES / MASTER_ADDR come from the env.
  : "${NODE_RANK:?set NODE_RANK=0 (master) or 1 ...}"
  [ -n "$MASTER_ADDR" ] || { echo "ERROR: MASTER_ADDR unset"; exit 1; }

  [ "$REPORT_TO" = "wandb" ] && export WANDB_API_KEY=${WANDB_API_KEY:?set WANDB_API_KEY for wandb reporting}
  export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/cache/compiled_kernels}
  export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
  export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
  export SPECFORGE_DATA_NUM_PROC=${SPECFORGE_DATA_NUM_PROC:-64}
  # Rendezvous/bootstrap over the routable Ethernet; data plane (incl. deepep EP
  # all-to-all) uses the 8 mlx5 InfiniBand HCAs (GPUDirect RDMA), auto-detected.
  export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
  export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond0}
  # NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here — it
  # conflicts with sglang's pynccl cuMem/NVLS allocation and makes
  # ncclCommInitRank fail with "NCCL error: invalid usage". Lowering
  # MEM_FRACTION is the right lever for the MoE-intermediate OOM.

  # export NCCL_DEBUG=INFO
  # export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7

  # Preflight: fail fast on a node where deps aren't installed.
  if ! python3 -c "import specforge, yunchang, accelerate, sglang, deep_ep" 2>/dev/null; then
    echo "ERROR: training deps missing on this node. Run: $0 setup" >&2; exit 1
  fi

  local tracker=(--report-to "$REPORT_TO")
  [ "$REPORT_TO" = "wandb" ] && tracker+=(--wandb-project specforge-dflash --wandb-name kimi-k2.7-code-dflash)

  mkdir -p "$OUTPUT_DIR"
  log "train: node_rank=$NODE_RANK/$NNODES world=$WORLD master=$MASTER_ADDR:$MASTER_PORT \
batch=$BATCH_SIZE lr=$LEARNING_RATE (eff. global batch=$((WORLD * BATCH_SIZE)))"

  torchrun \
    --nnodes "$NNODES" --nproc-per-node "$NUM_GPUS" --node-rank "$NODE_RANK" \
    --rdzv-backend c10d --rdzv-endpoint "$MASTER_ADDR:$MASTER_PORT" \
    --rdzv-id "$RDZV_ID" --max-restarts "$MAX_RESTARTS" \
    "$ROOT_DIR/scripts/train_dflash.py" \
    --target-model-path "$TARGET_MODEL" \
    --target-model-backend sglang --trust-remote-code \
    --tp-size "$WORLD" \
    --sglang-ep-size "$WORLD" \
    --sglang-dp-size "$WORLD" \
    --sglang-enable-dp-attention \
    --sglang-moe-a2a-backend deepep \
    --sglang-attention-backend flashinfer \
    --sglang-mem-fraction-static "$MEM_FRACTION" \
    --draft-config-path "$DRAFT_CONFIG" \
    --embedding-key language_model.model.embed_tokens.weight \
    --lm-head-key language_model.lm_head.weight \
    --mask-token-id 163838 \
    --train-data-path "$DATA_DIR/nemotron_v2_train.jsonl" \
    --eval-data-path "$DATA_DIR/nemotron_v2_eval_2k.jsonl" \
    --output-dir "$OUTPUT_DIR" --cache-dir "$ROOT_DIR/cache" \
    --num-epochs "$NUM_EPOCHS" --batch-size "$BATCH_SIZE" --learning-rate "$LEARNING_RATE" \
    --warmup-ratio 0.04 --max-grad-norm 1.0 --max-length "$MAX_LENGTH" \
    --chat-template "$CHAT_TEMPLATE" --attention-backend "$ATTENTION_BACKEND" \
    --block-size 8 --num-anchors 512 --loss-decay-gamma 4.0 \
    --dataloader-num-workers 0 --log-interval "$LOG_INTERVAL" --save-interval "$SAVE_INTERVAL" --eval-interval "$EVAL_INTERVAL" \
    --dist-timeout 60 "${tracker[@]}" --resume
}

cmd_watchdog() {
  # Per-node supervisor: relaunch `train` (which --resume's) on unexpected death;
  # stop on clean finish (epoch_6_step_*); guard against crash-loops.
  : "${NODE_RANK:?set NODE_RANK for watchdog}"
  local wlog="$OUTPUT_DIR/watchdog.rank${NODE_RANK}.log"
  local tlog="$OUTPUT_DIR/train.rank${NODE_RANK}.log"
  mkdir -p "$OUTPUT_DIR"
  local -a starts=(); local max=8 window=1800
  echo "$(date -u +%FT%TZ) [wd r$NODE_RANK] started" >> "$wlog"
  while true; do
    if pgrep -f "scripts/train_dflash.py" >/dev/null; then sleep 120; continue; fi
    if ls -d "$OUTPUT_DIR"/epoch_6_step_* >/dev/null 2>&1; then
      echo "$(date -u +%FT%TZ) [wd r$NODE_RANK] clean finish, exit" >> "$wlog"; exit 0
    fi
    local now; now=$(date +%s); starts+=("$now"); local recent=0
    for t in "${starts[@]}"; do [ $((now - t)) -le "$window" ] && recent=$((recent+1)); done
    if [ "${#starts[@]}" -gt "$max" ] || [ "$recent" -gt 3 ]; then
      echo "$(date -u +%FT%TZ) [wd r$NODE_RANK] crash-loop, giving up" >> "$wlog"; exit 1
    fi
    echo "$(date -u +%FT%TZ) [wd r$NODE_RANK] relaunch #${#starts[@]}" >> "$wlog"
    sleep 20
    echo "===== WATCHDOG RELAUNCH $(date -u +%FT%TZ) =====" >> "$tlog"
    bash "$SCRIPT_PATH" train >> "$tlog" 2>&1 &
    sleep 120
  done
}

case "${1:-}" in
  setup)    cmd_setup ;;
  prepare)  cmd_prepare ;;
  train)    cmd_train ;;
  watchdog) cmd_watchdog ;;
  *) sed -n '2,55p' "$SCRIPT_PATH"; echo; echo "ERROR: unknown command '${1:-}'"; exit 1 ;;
esac
