#!/usr/bin/env bash
# Manual (non-managed) disaggregated launch on this node for a *resume* run:
# the managed_local supervisor refuses training.resume_from, so start the same
# services it would start (mooncake_master, 4 TP1 capture servers on GPUs 0-3),
# then the producer and a 4-rank consumer on GPUs 4-7, with the env the planner
# renders for managed children. Recipe must be a non-managed disaggregated
# recipe (deployment.disaggregated without managed_local) whose control_dir is
# fresh and whose consumer_state_dir holds the retained consumer.sqlite.
#   usage: launch-moe-disagg-manual-on-103.sh <recipe.yaml>
set -euo pipefail
CK=/personal/SpecForge-qwen38-moe
RECIPE=$1
PY=/opt/sglang/bin/python
export PYTHONPATH=/opt/sglang-0518:$CK MC_TRANSFER_TIMEOUT=300 SGLANG_SPEC_CAPTURE_SINK_CLIENTS=4 OMP_NUM_THREADS=${OMP_NUM_THREADS:-32}
cd "$CK"
eval "$($PY - "$RECIPE" <<'PYEOF'
import sys, yaml, shlex
c = yaml.safe_load(open(sys.argv[1]))
d = c["deployment"]["disaggregated"]
print("RUN_ID=" + shlex.quote(c["run_id"]))
print("OUT=" + shlex.quote(c["output_dir"]))
print("CONTROL=" + shlex.quote(d["control_dir"]))
print("CSTATE=" + shlex.quote(d["consumer_state_dir"]))
print("CLIENT_BUF=" + str(d.get("client_buffer_size", 1073741824)))
print("RESUME=" + shlex.quote(str(c["training"].get("resume_from", ""))))
m = c["model"]
print("MODEL=" + shlex.quote(m["target_model_path"]))
print("ATTN=" + shlex.quote(m.get("sglang_attention_backend", "triton")))
print("MEMFRAC=" + str(m.get("sglang_mem_fraction_static", 0.85)))
print("CTX=" + str(m.get("sglang_context_length", c["data"]["max_length"] + 7)))
print("MAXREQ=" + str(m.get("sglang_max_running_requests", 2)))
print("MAXTOK=" + str(m.get("sglang_max_total_tokens", 32768)))
PYEOF
)"
if pgrep -f "specforge.cli tr[a]in" > /dev/null; then echo "a specforge train process is already running" >&2; exit 1; fi
if nvidia-smi --query-gpu=memory.used --format=csv,noheader | grep -qv '^0 MiB'; then echo "GPUs are not idle" >&2; exit 1; fi
if [ -e "$CONTROL/refs.jsonl" ]; then echo "control dir already has a channel: $CONTROL" >&2; exit 1; fi
if [ -n "$RESUME" ] && [ ! -f "$CSTATE/consumer.sqlite" ]; then echo "resume needs the retained $CSTATE/consumer.sqlite" >&2; exit 1; fi
mkdir -p "$CONTROL/logs" "$CSTATE" "$OUT"
LOG=$CONTROL/logs
export MOONCAKE_METADATA_SERVER=http://127.0.0.1:35880/metadata MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:35551 \
       MOONCAKE_LOCAL_HOSTNAME=127.0.0.1 MOONCAKE_PROTOCOL=tcp \
       DISAGG_SERVER_URLS=http://127.0.0.1:30000,http://127.0.0.1:30001,http://127.0.0.1:30002,http://127.0.0.1:30003 \
       DISAGG_BACKEND=mooncake DISAGG_CLIENT_BUFFER_SIZE=$CLIENT_BUF DISAGG_CLIENT_SEGMENT_SIZE=0 \
       DISAGG_DB=$CSTATE/consumer.sqlite DISAGG_INBOX_DIR=$CSTATE/inboxes DISAGG_REF_CHANNEL=$CONTROL/refs.jsonl \
       DISAGG_STORE_ID=$RUN_ID
echo "[manual] mooncake_master"
CUDA_VISIBLE_DEVICES="" setsid nohup mooncake_master --enable_http_metadata_server=true --http_metadata_server_host=127.0.0.1 \
  --rpc_port=35551 --http_metadata_server_port=35880 --metrics_port=35903 --default_kv_lease_ttl=300000 > "$LOG/mooncake.log" 2>&1 < /dev/null &
sleep 5
for i in 0 1 2 3; do
  echo "[manual] capture-server-$i on GPU $i"
  CUDA_VISIBLE_DEVICES=$i MOONCAKE_GLOBAL_SEGMENT_SIZE=107374182400 MOONCAKE_LOCAL_BUFFER_SIZE=1073741824 \
  setsid nohup $PY -m sglang.launch_server --model-path "$MODEL" --dtype bfloat16 --trust-remote-code --skip-tokenizer-init \
    --tp-size 1 --chunked-prefill-size -1 --enable-spec-capture --spec-capture-method dspark --spec-capture-aux-layer-ids 5 19 33 47 61 \
    --host 127.0.0.1 --port 3000$i --attention-backend "$ATTN" --mem-fraction-static "$MEMFRAC" --disable-radix-cache \
    --context-length "$CTX" --ep-size 1 --max-running-requests "$MAXREQ" --max-total-tokens "$MAXTOK" > "$LOG/capture-server-$i.log" 2>&1 < /dev/null &
  [ -n "${SPECFORGE_SERIAL_SERVICE_STARTUP:-}" ] && sleep 90
done
echo "[manual] waiting for capture servers"
for i in $(seq 1 120); do ok=0; for p in 30000 30001 30002 30003; do curl -sf -m 3 http://127.0.0.1:$p/health > /dev/null && ok=$((ok+1)); done; [ $ok = 4 ] && break; sleep 10; done
echo "[manual] $ok/4 capture servers healthy after $((i*10))s"
[ "$ok" = 4 ] || { echo "capture servers failed to come up; see $LOG" >&2; exit 1; }
echo "[manual] producer"
CUDA_VISIBLE_DEVICES="" setsid nohup $PY -m specforge.cli train --config "$RECIPE" --role producer > "$OUT/producer.log" 2>&1 < /dev/null &
sleep 5
echo "[manual] consumer (torchrun x4 on GPUs 4-7, numactl --preferred=1)"
CUDA_VISIBLE_DEVICES=4,5,6,7 setsid nohup numactl --preferred=1 $PY -m torch.distributed.run --standalone --nproc_per_node 4 \
  --module specforge.cli train --config "$RECIPE" --role consumer > "$OUT/launch.log" 2>&1 < /dev/null &
echo "[manual] launched $RUN_ID on $(hostname): consumer log $OUT/launch.log, producer log $OUT/producer.log, services $LOG"
