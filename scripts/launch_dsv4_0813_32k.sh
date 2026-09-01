#!/usr/bin/env bash
# Launch the DSV4-Flash DSpark capture stack at 32K context and the
# 0813-traffic 32K training run. Assumes GPUs are free.
#
#   ./scripts/launch_dsv4_0813_32k.sh servers   # mooncake master + 2x TP2 capture servers
#   ./scripts/launch_dsv4_0813_32k.sh train     # supervisor (CPU producer + 4-rank consumer)
set -euo pipefail

# Run from the repo root with your SpecForge environment active.
cd "$(dirname "$0")/.."

RUN=deepseek-v4-flash-dspark-0813-32k
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
  # Fresh master: clears metadata whose replicas lived in dead servers.
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
  # One optimizer quantum is ~90 GiB (64 samples x ~1.4 GiB at the ~30K
  # median); two quanta stay in flight, split across the two servers.
  export MOONCAKE_GLOBAL_SEGMENT_SIZE=$((192 << 30))
  export MOONCAKE_LOCAL_BUFFER_SIZE=$((1 << 30))
  # flashinfer CuTe-DSL rmsnorm/mxfp8-quantize kernels miscompile against the
  # pinned nvidia-cutlass-dsl; force the CUDA backends (QUANT honored via the
  # spec-capture patch).
  export FLASHINFER_USE_CUDA_NORM=1
  export FLASHINFER_USE_CUDA_QUANT=1

  for i in 0 1 2; do
    port=$((30000 + i))
    gpus=$((i * 2)),$((i * 2 + 1))
    CUDA_VISIBLE_DEVICES=$gpus nohup python -m sglang.launch_server \
      --host 0.0.0.0 \
      --port $port \
      --model-path deepseek-ai/DeepSeek-V4-Flash-0731 \
      --trust-remote-code \
      --skip-tokenizer-init \
      --tp-size 2 \
      --mem-fraction-static 0.88 \
      --context-length 33280 \
      --max-running-requests 8 \
      --chunked-prefill-size -1 \
      --max-prefill-tokens 33280 \
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
  CUDA_VISIBLE_DEVICES=6,7 nohup specforge train \
    -c examples/configs/online/disaggregated/external/deepseek-v4-flash-dspark-0813-32k.yaml \
    > "$LOGDIR/train.log" 2>&1 &
  echo "trainer supervisor pid $! -> $LOGDIR/train.log"
  ;;
*)
  echo "usage: $0 servers|train" >&2; exit 2 ;;
esac
