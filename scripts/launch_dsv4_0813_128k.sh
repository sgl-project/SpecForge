#!/usr/bin/env bash
# Relaunch the DSV4-Flash DSpark capture stack at 128K context and start the
# 0813-traffic training run. Assumes GPUs are free (old servers stopped).
#
#   ./scripts/launch_dsv4_0813_128k.sh servers   # mooncake master + 2x TP2 capture servers
#   ./scripts/launch_dsv4_0813_128k.sh train     # supervisor (CPU producer + 4-rank consumer)
set -euo pipefail

cd /personal/SpecForge
source .venv/bin/activate

RUN=deepseek-v4-flash-dspark-0813-128k
LOGDIR=outputs/$RUN/logs
mkdir -p "$LOGDIR"

common_env() {
  export MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:35551
  export MOONCAKE_METADATA_SERVER=http://127.0.0.1:35880/metadata
  export MOONCAKE_LOCAL_HOSTNAME=127.0.0.1
  export MOONCAKE_PROTOCOL=tcp
  export MC_TRANSFER_TIMEOUT=300
}

case "${1:-}" in
servers)
  # Fresh master: clears metadata whose replicas lived in the dead 8K servers.
  nohup mooncake_master \
    --enable_http_metadata_server=true \
    --http_metadata_server_host=0.0.0.0 \
    --rpc_port=35551 \
    --http_metadata_server_port=35880 \
    --metrics_port=35903 \
    --default_kv_lease_ttl=5m > "$LOGDIR/mooncake_master.log" 2>&1 &
  echo "mooncake_master pid $!"
  sleep 5

  common_env
  # One optimizer quantum is ~122 GiB (32 samples x ~3.8 GiB at the ~80K
  # median); two quanta stay in flight, so each server's segment holds half.
  export MOONCAKE_GLOBAL_SEGMENT_SIZE=$((256 << 30))
  export MOONCAKE_LOCAL_BUFFER_SIZE=$((1 << 30))
  # flashinfer CuTe-DSL rmsnorm/mxfp8-quantize kernels miscompile against the
  # pinned nvidia-cutlass-dsl; force the CUDA backends (QUANT honored via the
  # v0.5.14 capture patch).
  export FLASHINFER_USE_CUDA_NORM=1
  export FLASHINFER_USE_CUDA_QUANT=1

  for i in 0 1; do
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
      --context-length 131584 \
      --max-running-requests 4 \
      --chunked-prefill-size -1 \
      --max-prefill-tokens 131584 \
      --moe-runner-backend flashinfer_mxfp4 \
      --enable-spec-capture \
      --spec-capture-method dspark \
      --spec-capture-aux-layer-ids 1 11 21 31 41 \
      --disable-cuda-graph > "$LOGDIR/sglang-$port.log" 2>&1 &
    echo "capture server :$port (GPUs $gpus) pid $!"
  done
  echo "watch: tail -f $LOGDIR/sglang-30000.log  (~12 min to /health)"
  ;;
train)
  common_env
  CUDA_VISIBLE_DEVICES=4,5,6,7 nohup specforge train \
    -c examples/configs/deepseek-v4-flash-dspark-0813-128k.yaml \
    > "$LOGDIR/train.log" 2>&1 &
  echo "trainer supervisor pid $! -> $LOGDIR/train.log"
  ;;
*)
  echo "usage: $0 servers|train" >&2; exit 2 ;;
esac
