#!/usr/bin/env bash
# Run ON the capture node (.103): 6 capture servers bound to 0.0.0.0 so the trainer node can reach them.
# Requires the mooncake_master on the trainer node to be up first (segments register with it).
set -euo pipefail
source "$(dirname "$0")/common.sh"
[ "$(hostname -I | tr ' ' '\n' | grep -c "^$CAPTURE_HOST$")" = 1 ] || { echo "run this on $CAPTURE_HOST" >&2; exit 1; }
if pgrep -f "sglang.launch_server.*--port 3000[0-7]" > /dev/null; then echo "capture servers (ports 30000-30007) already running here" >&2; exit 1; fi
for i in $(seq 0 $((N_CAPTURE-1))); do nvidia-smi --query-gpu=memory.used --format=csv,noheader -i $i | grep -q '^0 MiB' || { echo "GPU $i busy" >&2; exit 1; }; done
curl -sf -m 5 "$MOONCAKE_METADATA_SERVER?key=specforge-health-check" > /dev/null 2>&1 || curl -sf -m 5 "http://$TRAINER_HOST:35880/metadata?key=x" -o /dev/null 2>/dev/null || echo "warning: metadata server on $TRAINER_HOST:35880 not answering yet"
mkdir -p "$LOG"
export MOONCAKE_LOCAL_HOSTNAME=$CAPTURE_HOST
for i in $(seq 0 $((N_CAPTURE-1))); do
  echo "[capture] server $i on GPU $i, port $((30000+i))"
  CUDA_VISIBLE_DEVICES=$i MOONCAKE_GLOBAL_SEGMENT_SIZE=107374182400 MOONCAKE_LOCAL_BUFFER_SIZE=1073741824 \
  setsid nohup $PY -m sglang.launch_server --model-path "$MODEL" --dtype bfloat16 --trust-remote-code --skip-tokenizer-init \
    --tp-size 1 --chunked-prefill-size -1 --enable-spec-capture --spec-capture-method dspark --spec-capture-aux-layer-ids 5 19 33 47 61 \
    --host 0.0.0.0 --port $((30000+i)) --attention-backend "$ATTN" --mem-fraction-static "$MEMFRAC" --disable-radix-cache \
    --context-length "$CTX" --ep-size 1 --max-running-requests "$MAXREQ" --max-total-tokens "$MAXTOK" > "$LOG/capture-server-$i.log" 2>&1 < /dev/null &
  [ -n "${SPECFORGE_SERIAL_SERVICE_STARTUP:-}" ] && sleep 90
done
for t in $(seq 1 120); do ok=0; for i in $(seq 0 $((N_CAPTURE-1))); do curl -sf -m 3 http://127.0.0.1:$((30000+i))/health > /dev/null && ok=$((ok+1)); done; [ $ok = $N_CAPTURE ] && break; sleep 10; done
echo "[capture] $ok/$N_CAPTURE healthy after $((t*10))s (logs: $LOG)"; [ "$ok" = "$N_CAPTURE" ]
