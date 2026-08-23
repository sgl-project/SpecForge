#!/usr/bin/env bash
# Launch the OFFICIAL-architecture DSV4-Flash DSpark two-node stack.
#
# Capture node (10.220.51.50):
#   ./scripts/launch_dsv4_official_2node.sh master      # mooncake master (once)
#   ./scripts/launch_dsv4_official_2node.sh servers     # 4x TP2 capture, 8K (ShareGPT recipe)
#   ./scripts/launch_dsv4_official_2node.sh servers32k  # 4x TP2 capture, 32K (0813-traffic recipe)
# Trainer node (10.220.51.52):
#   ./scripts/launch_dsv4_official_2node.sh train       # ShareGPT recipe trainer
#   ./scripts/launch_dsv4_official_2node.sh train32k    # 0813-traffic recipe trainer
#
# Differences vs the base 2node recipe: capture layers 40 41 42 (official
# DSpark projector input) and the official-architecture trainer config.
set -euo pipefail

cd /personal/SpecForge
[ -f .venv/bin/activate ] && source .venv/bin/activate

case "${1:-}" in
*32k)      RUN=deepseek-v4-flash-dspark-official-0813-32k ;;
*continue) RUN=deepseek-v4-flash-dspark-official-continue-sharegpt ;;
*r2)       RUN=deepseek-v4-flash-dspark-official-2node-h200-r2 ;;
*)         RUN=deepseek-v4-flash-dspark-official-2node-h200 ;;
esac
LOGDIR=outputs/$RUN/logs
mkdir -p "$LOGDIR"

CAPTURE_IP=10.220.51.50
TRAINER_IP=10.220.51.52

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
      --context-length $context_len \
      --max-running-requests 8 \
      --chunked-prefill-size -1 \
      --max-prefill-tokens $context_len \
      --moe-runner-backend marlin \
      --watchdog-timeout 3600 \
      --enable-spec-capture \
      --spec-capture-method dspark \
      --spec-capture-aux-layer-ids 40 41 42 \
      --disable-cuda-graph > "$LOGDIR/sglang-$port.log" 2>&1 &
    echo "capture server :$port (GPUs $gpus) pid $!"
  done
  echo "watch: tail -f $LOGDIR/sglang-30000.log"
}

launch_trainer() {
  local config=$1
  export HF_HOME=/cluster-storage/models
  export MC_TRANSFER_TIMEOUT=300
  export SPECFORGE_MOONCAKE_FETCH_CLIENTS=4
  # ~19.9B drafter under FULL_SHARD runs close to the HBM budget; expandable
  # segments avoids fragmentation-driven OOM across the 16 microbatches.
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  nohup specforge train -c "$config" > "$LOGDIR/train.log" 2>&1 &
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
servers)
  # One optimizer quantum is ~32 GiB (128 samples x ~0.25 GiB at 8K with the
  # 3-layer capture); two quanta in flight, split across four servers.
  launch_capture_servers 8704 $((24 << 30))
  ;;
servers32k)
  # ~0.94 GiB per ~30K median sample; a 128-sample quantum is ~120 GiB.
  # 4 x 128 GiB segments must exceed feature_store_max_resident_bytes.
  launch_capture_servers 33280 $((128 << 30))
  ;;
train)
  launch_trainer examples/configs/deepseek-v4-flash-dspark-official-2node-h200.yaml
  ;;
train32k)
  launch_trainer examples/configs/deepseek-v4-flash-dspark-official-0813-32k.yaml
  ;;
trainr2)
  # ShareGPT r2: weights-only warm start from r1's step1152, lr 3e-4, clip 0.5.
  launch_trainer examples/configs/deepseek-v4-flash-dspark-official-2node-h200-r2.yaml
  ;;
traincontinue)
  # Bounded "meaningful loss" check: continue the OFFICIAL drafter on
  # ShareGPT for max_steps optimizer steps (uses the 8K capture servers).
  launch_trainer examples/configs/deepseek-v4-flash-dspark-official-continue-sharegpt.yaml
  ;;
*)
  echo "usage: $0 master|servers|servers32k|train|train32k|traincontinue" >&2; exit 2 ;;
esac
