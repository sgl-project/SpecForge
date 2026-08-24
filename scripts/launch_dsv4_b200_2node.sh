#!/usr/bin/env bash
# Launch the OFFICIAL-architecture DSV4-Flash DSpark two-node stack on the
# verda-b200-fin-03 pair (8x B200 each).
#
# Capture node (10.13.114.101):
#   ./scripts/launch_dsv4_b200_2node.sh master      # mooncake master (once)
#   ./scripts/launch_dsv4_b200_2node.sh servers32k  # capture servers, 32K ctx
# Trainer node (10.13.114.105):
#   ./scripts/launch_dsv4_b200_2node.sh train32k    # prod 0813-32k trainer
#   ./scripts/launch_dsv4_b200_2node.sh tune32k     # bounded tuning trainer
#
# Tuning knobs (env overrides, all optional):
#   NUM_SERVERS=4 TP=2            capture-server topology (NUM_SERVERS*TP<=8)
#   MOE_BACKEND=flashinfer_mxfp4  sglang --moe-runner-backend
#   MAX_RUNNING=8                 sglang --max-running-requests
#   SINK_CLIENTS=3                SGLANG_SPEC_CAPTURE_SINK_CLIENTS (capture)
#   FETCH_CLIENTS=4               SPECFORGE_MOONCAKE_FETCH_CLIENTS (trainer)
set -euo pipefail

cd /personal/SpecForge-2Nodes

export HF_HUB_CACHE=/cluster-storage/models

case "${1:-}" in
*tune*) RUN=deepseek-v4-flash-dspark-official-0813-32k-b200-2nodes-tune ;;
*)      RUN=deepseek-v4-flash-dspark-official-0813-32k-b200-2nodes ;;
esac
LOGDIR=outputs/$RUN/logs
mkdir -p "$LOGDIR"

CAPTURE_IP=10.13.114.101
TRAINER_IP=10.13.114.105

NUM_SERVERS=${NUM_SERVERS:-4}
TP=${TP:-2}
MOE_BACKEND=${MOE_BACKEND:-flashinfer_mxfp4}
MAX_RUNNING=${MAX_RUNNING:-8}
SINK_CLIENTS=${SINK_CLIENTS:-3}
FETCH_CLIENTS=${FETCH_CLIENTS:-4}

launch_capture_servers() {
  local context_len=$1 segment_bytes=$2
  export MOONCAKE_MASTER_SERVER_ADDR=$CAPTURE_IP:35551
  export MOONCAKE_METADATA_SERVER=http://$CAPTURE_IP:35880/metadata
  # The trainer node fetches features from these servers' Mooncake segments;
  # 127.0.0.1 would register an unreachable data-plane endpoint.
  export MOONCAKE_LOCAL_HOSTNAME=$CAPTURE_IP
  export MOONCAKE_PROTOCOL=tcp
  export MC_TRANSFER_TIMEOUT=300
  export MOONCAKE_GLOBAL_SEGMENT_SIZE=$segment_bytes
  export MOONCAKE_LOCAL_BUFFER_SIZE=$((1 << 30))
  export SGLANG_SPEC_CAPTURE_SINK_CLIENTS=$SINK_CLIENTS

  for ((i = 0; i < NUM_SERVERS; i++)); do
    port=$((30000 + i))
    gpus=$(seq -s, $((i * TP)) $((i * TP + TP - 1)))
    CUDA_VISIBLE_DEVICES=$gpus nohup python3 -m sglang.launch_server \
      --host 0.0.0.0 \
      --port $port \
      --model-path deepseek-ai/DeepSeek-V4-Flash-0731 \
      --trust-remote-code \
      --skip-tokenizer-init \
      --tp-size $TP \
      --mem-fraction-static 0.85 \
      --context-length $context_len \
      --max-running-requests $MAX_RUNNING \
      --chunked-prefill-size -1 \
      --max-prefill-tokens $context_len \
      --moe-runner-backend $MOE_BACKEND \
      --watchdog-timeout 3600 \
      --enable-spec-capture \
      --spec-capture-method dspark \
      --spec-capture-aux-layer-ids 40 41 42 \
      --disable-cuda-graph > "$LOGDIR/sglang-$port.log" 2>&1 &
    echo "capture server :$port (GPUs $gpus, TP$TP, moe=$MOE_BACKEND) pid $!"
  done
  echo "watch: tail -f $LOGDIR/sglang-30000.log"
}

launch_trainer() {
  local config=$1
  export MC_TRANSFER_TIMEOUT=300
  export SPECFORGE_MOONCAKE_FETCH_CLIENTS=$FETCH_CLIENTS
  # ~19.9B drafter under FULL_SHARD; expandable segments avoids
  # fragmentation-driven OOM across the 16 microbatches, and the GC threshold
  # reclaims cached blocks before new segment mappings fail.
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.9
  nohup python3 -m specforge.cli train -c "$config" > "$LOGDIR/train.log" 2>&1 &
  echo "trainer supervisor pid $! -> $LOGDIR/train.log"
}

case "${1:-}" in
master)
  # Fresh master clears metadata whose replicas lived in dead servers.
  nohup mooncake_master \
    --enable_http_metadata_server=true \
    --http_metadata_server_host=0.0.0.0 \
    --rpc_port=35551 \
    --http_metadata_server_port=35880 \
    --metrics_port=35903 \
    --default_kv_lease_ttl=5m > "$LOGDIR/mooncake_master.log" 2>&1 &
  echo "mooncake_master pid $!"
  ;;
servers32k|servers32k-tune)
  # ~0.94 GiB per ~30K median sample; a 128-sample quantum is ~120 GiB.
  # Segments must exceed feature_store_max_resident_bytes (400 GiB) in total.
  launch_capture_servers 33280 $(((512 / NUM_SERVERS) << 30))
  ;;
train32k)
  launch_trainer examples/configs/deepseek-v4-flash-dspark-official-0813-32k-b200-2nodes.yaml
  ;;
tune32k)
  launch_trainer examples/configs/deepseek-v4-flash-dspark-official-0813-32k-b200-2nodes-tune.yaml
  ;;
*)
  echo "usage: $0 master|servers32k|train32k|tune32k" >&2; exit 2 ;;
esac
