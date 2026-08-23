#!/usr/bin/env bash
# Launch the OFFICIAL-architecture DSV4-Flash DSpark two-node stack.
#
# Capture node (10.220.51.50):
#   ./scripts/launch_dsv4_official_2node.sh master    # mooncake master (once)
#   ./scripts/launch_dsv4_official_2node.sh servers   # 4x TP2 capture servers
# Trainer node (10.220.51.52):
#   ./scripts/launch_dsv4_official_2node.sh train     # supervisor (producer + 8-rank consumer)
#
# Differences vs the base 2node recipe: capture layers 40 41 42 (official
# DSpark projector input) and the official-architecture trainer config.
set -euo pipefail

cd /personal/SpecForge
[ -f .venv/bin/activate ] && source .venv/bin/activate

RUN=deepseek-v4-flash-dspark-official-2node-h200
LOGDIR=outputs/$RUN/logs
mkdir -p "$LOGDIR"

CAPTURE_IP=10.220.51.50
TRAINER_IP=10.220.51.52

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
servers)
  export MOONCAKE_MASTER_SERVER_ADDR=$CAPTURE_IP:35551
  export MOONCAKE_METADATA_SERVER=http://$CAPTURE_IP:35880/metadata
  # The trainer node fetches features from these servers' Mooncake segments;
  # 127.0.0.1 would register an unreachable data-plane endpoint.
  export MOONCAKE_LOCAL_HOSTNAME=$CAPTURE_IP
  export MOONCAKE_PROTOCOL=tcp
  export MC_TRANSFER_TIMEOUT=300
  # One optimizer quantum is ~32 GiB (128 samples x ~0.25 GiB at 8K with the
  # 3-layer capture); two quanta in flight, split across four servers.
  export MOONCAKE_GLOBAL_SEGMENT_SIZE=$((24 << 30))
  export MOONCAKE_LOCAL_BUFFER_SIZE=$((1 << 30))
  export SGLANG_SPEC_CAPTURE_SINK_CLIENTS=3

  for i in 0 1 2 3; do
    port=$((30000 + i))
    gpus=$((i * 2)),$((i * 2 + 1))
    CUDA_VISIBLE_DEVICES=$gpus nohup python -m sglang.launch_server \
      --host 0.0.0.0 \
      --port $port \
      --model-path deepseek-ai/DeepSeek-V4-Flash-0731 \
      --trust-remote-code \
      --skip-tokenizer-init \
      --tp-size 2 \
      --mem-fraction-static 0.85 \
      --context-length 8704 \
      --max-running-requests 8 \
      --chunked-prefill-size -1 \
      --max-prefill-tokens 8704 \
      --moe-runner-backend marlin \
      --watchdog-timeout 3600 \
      --enable-spec-capture \
      --spec-capture-method dspark \
      --spec-capture-aux-layer-ids 40 41 42 \
      --disable-cuda-graph > "$LOGDIR/sglang-$port.log" 2>&1 &
    echo "capture server :$port (GPUs $gpus) pid $!"
  done
  echo "watch: tail -f $LOGDIR/sglang-30000.log  (marlin JIT can take a while on cold cache)"
  ;;
train)
  export HF_HOME=/cluster-storage/models
  export MC_TRANSFER_TIMEOUT=300
  export SPECFORGE_MOONCAKE_FETCH_CLIENTS=4
  # ~19.9B drafter under FULL_SHARD runs close to the HBM budget; expandable
  # segments avoids fragmentation-driven OOM across the 16 microbatches.
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  nohup specforge train \
    -c examples/configs/deepseek-v4-flash-dspark-official-2node-h200.yaml \
    > "$LOGDIR/train.log" 2>&1 &
  echo "trainer supervisor pid $! -> $LOGDIR/train.log"
  ;;
*)
  echo "usage: $0 master|servers|train" >&2; exit 2 ;;
esac
