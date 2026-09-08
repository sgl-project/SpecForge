#!/usr/bin/env bash
# Run ON the trainer node (.102). Step 1 (--master): start mooncake_master bound to all interfaces, then start
# the capture servers on .103 with launch-capture-on-103.sh. Step 2 (--train): producer + 8-rank consumer.
set -euo pipefail
MODE=${2:-}
source "$(dirname "$0")/common.sh"
[ "$(hostname -I | tr ' ' '\n' | grep -c "^$TRAINER_HOST$")" = 1 ] || { echo "run this on $TRAINER_HOST" >&2; exit 1; }
mkdir -p "$LOG" "$CSTATE" "$OUT"
if [ "$MODE" = "--master" ]; then
  pgrep -x mooncake_master > /dev/null && { echo "mooncake_master already running" >&2; exit 1; }
  CUDA_VISIBLE_DEVICES="" setsid nohup mooncake_master --enable_http_metadata_server=true --http_metadata_server_host=0.0.0.0 \
    --rpc_address=0.0.0.0 --rpc_port=35551 --http_metadata_server_port=35880 --metrics_port=35903 --default_kv_lease_ttl=300000 \
    > "$LOG/mooncake.log" 2>&1 < /dev/null &
  sleep 3; echo "[trainer-node] mooncake_master up: $(pgrep -xc mooncake_master)"; exit 0
fi
if [ "$MODE" = "--train" ]; then
  if pgrep -f "specforge.cli tr[a]in" > /dev/null; then echo "a specforge train process is already running" >&2; exit 1; fi
  if nvidia-smi --query-gpu=memory.used --format=csv,noheader | grep -qv '^0 MiB'; then echo "GPUs not idle" >&2; exit 1; fi
  [ -e "$CONTROL/refs.jsonl" ] && { echo "control dir already has a channel: $CONTROL" >&2; exit 1; }
  ok=0; for u in ${SERVER_URLS//,/ }; do curl -sf -m 5 "$u/health" > /dev/null && ok=$((ok+1)); done
  [ "$ok" = "$N_CAPTURE" ] || { echo "only $ok/$N_CAPTURE capture servers healthy at $SERVER_URLS" >&2; exit 1; }
  export MOONCAKE_LOCAL_HOSTNAME=$TRAINER_HOST OMP_NUM_THREADS=${OMP_NUM_THREADS:-24} SPECFORGE_MOONCAKE_FETCH_CLIENTS=${SPECFORGE_MOONCAKE_FETCH_CLIENTS:-8}
  cd "$CK"
  echo "[trainer-node] producer"
  CUDA_VISIBLE_DEVICES="" setsid nohup $PY -m specforge.cli train --config "$RECIPE" --role producer > "$OUT/producer.log" 2>&1 < /dev/null &
  sleep 5
  echo "[trainer-node] consumer torchrun x$NPROC"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 setsid nohup $PY -m torch.distributed.run --standalone --nproc_per_node $NPROC \
    --module specforge.cli train --config "$RECIPE" --role consumer > "$OUT/launch.log" 2>&1 < /dev/null &
  echo "[trainer-node] launched $RUN_ID: consumer $OUT/launch.log, producer $OUT/producer.log"; exit 0
fi
echo "usage: $0 <recipe.yaml> --master | --train" >&2; exit 2
