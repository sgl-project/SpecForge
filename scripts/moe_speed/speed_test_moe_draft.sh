#!/usr/bin/env bash
# Serving-speed comparison of DSpark drafters at a FIXED accept length.
#
# For each drafter, starts one SGLang server (target + drafter, DSPARK block 7)
# with SGLANG_SIMULATE_ACC_LEN so every verify step commits the same number of
# tokens, then runs sglang.bench_serving at several concurrencies and records
# output tok/s. With the accept length pinned, tok/s differences are step-time
# differences, so random-expert MoE drafters can be compared against the dense one.
#
# Usage:
#   bash scripts/moe_speed/speed_test_moe_draft.sh <target> <label>=<draft_dir>[:<quant>] [...]
# Example:
#   bash scripts/moe_speed/speed_test_moe_draft.sh Qwen/Qwen3.8-27B-FP8 \
#     dense=RadixArk/Qwen3.8-27B-DSpark \
#     moe16x1024=exports/qwen38-dspark-moe-16x1024-rand \
#     moe16x1024_fp8=exports/qwen38-dspark-moe-16x1024-rand:fp8
# Env knobs: GPU (CUDA_VISIBLE_DEVICES, default 0), PORT (30100), ACC_LEN (5.0),
#   CONCS ("1 8 32"), PROMPTS_PER_CONC (8), IN_LEN (512), OUT_LEN (512),
#   RESULTS (results/moe_speed), MOE_RUNNER (triton), EXTRA_SERVER_ARGS,
#   MAMBA_CACHE (160), MAX_RUNNING (32).
set -euo pipefail

TARGET=${1:?target model path}; shift
[ $# -ge 1 ] || { echo "need at least one <label>=<draft_dir>[:<quant>]"; exit 1; }

GPU=${GPU:-0}
PORT=${PORT:-30100}
ACC_LEN=${ACC_LEN:-5.0}
CONCS=${CONCS:-"1 8 32"}
PROMPTS_PER_CONC=${PROMPTS_PER_CONC:-8}
IN_LEN=${IN_LEN:-512}
OUT_LEN=${OUT_LEN:-512}
RESULTS=${RESULTS:-results/moe_speed}
MOE_RUNNER=${MOE_RUNNER:-triton}
MAMBA_CACHE=${MAMBA_CACHE:-160}
MAX_RUNNING=${MAX_RUNNING:-32}
EXTRA_SERVER_ARGS=${EXTRA_SERVER_ARGS:-}
PY=${PY:-python}

mkdir -p "$RESULTS"
TSV="$RESULTS/summary.tsv"
echo -e "label\tquant\tconc\tnum_prompts\toutput_tok_s\tmean_e2e_ms\tmean_tpot_ms\taccept_len" > "$TSV"

wait_health() {
  local tries=0
  until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    sleep 5; tries=$((tries+1))
    if [ $tries -gt 240 ]; then echo "server did not come up in 20 min"; return 1; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "server died, see $LOG"; return 1; fi
  done
}

stop_server() {
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    pkill -P "$SERVER_PID" 2>/dev/null || true
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 3
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
  # anything else still holding the port
  for p in $(lsof -t -i :"$PORT" 2>/dev/null); do kill -9 "$p" 2>/dev/null || true; done
  sleep 5
}
trap stop_server EXIT

for spec in "$@"; do
  label=${spec%%=*}
  rest=${spec#*=}
  draft=${rest%%:*}
  quant=unquant
  if [[ "$rest" == *:* ]]; then quant=${rest##*:}; fi
  LOG="$RESULTS/server_${label}.log"
  echo "================ $label  draft=$draft quant=$quant ================"

  CUDA_VISIBLE_DEVICES=$GPU \
  SGLANG_SIMULATE_ACC_LEN=$ACC_LEN SGLANG_SIMULATE_ACC_METHOD=${SIM_METHOD:-match-expected} \
  SGLANG_RAGGED_VERIFY_MODE=static \
  SGLANG_EXTERNAL_MODEL_PACKAGE=specforge.serving.sglang_models \
  $PY -m sglang.launch_server \
    --model-path "$TARGET" \
    --speculative-algorithm DSPARK --speculative-dspark-block-size 7 \
    --speculative-draft-model-path "$draft" \
    --speculative-draft-model-quantization "$quant" \
    --attention-backend triton --speculative-draft-attention-backend triton \
    --moe-runner-backend "$MOE_RUNNER" \
    --max-running-requests "$MAX_RUNNING" --cuda-graph-max-bs "$MAX_RUNNING" \
    --max-mamba-cache-size "$MAMBA_CACHE" \
    --host 127.0.0.1 --port "$PORT" --log-level warning \
    $EXTRA_SERVER_ARGS > "$LOG" 2>&1 &
  SERVER_PID=$!
  wait_health

  # warm-up (cuda graphs, caches)
  $PY -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port "$PORT" \
    --dataset-name random --random-input-len "$IN_LEN" --random-output-len 64 --random-range-ratio 1.0 \
    --num-prompts 8 --max-concurrency 8 --request-rate inf --disable-tqdm \
    --output-file "$RESULTS/warmup_${label}.jsonl" > /dev/null 2>&1 || true

  for c in $CONCS; do
    np=$((c * PROMPTS_PER_CONC))
    out="$RESULTS/bench_${label}_c${c}.jsonl"
    rm -f "$out"
    $PY -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port "$PORT" \
      --dataset-name random --random-input-len "$IN_LEN" --random-output-len "$OUT_LEN" --random-range-ratio 1.0 \
      --num-prompts "$np" --max-concurrency "$c" --request-rate inf --disable-tqdm \
      --output-file "$out" > "$RESULTS/bench_${label}_c${c}.log" 2>&1
    acc=$(curl -s "http://127.0.0.1:$PORT/server_info" | $PY -c 'import sys,json
d=json.load(sys.stdin)
for k in ("avg_spec_accept_length","spec_accept_length"):
    if k in d: print(d[k]); break
else:
    i=d.get("internal_states",[{}])
    print(i[0].get("avg_spec_accept_length","n/a") if isinstance(i,list) and i else "n/a")' 2>/dev/null || echo n/a)
    $PY - "$out" "$label" "$quant" "$c" "$np" "$acc" "$TSV" <<'EOF'
import json, sys
out, label, quant, c, np_, acc, tsv = sys.argv[1:8]
d = json.loads(open(out).read().strip().splitlines()[-1])
row = [label, quant, c, np_, f"{d.get('output_throughput', float('nan')):.1f}",
       f"{d.get('mean_e2e_latency_ms', float('nan')):.1f}", f"{d.get('mean_tpot_ms', float('nan')):.2f}", acc]
open(tsv, "a").write("\t".join(map(str, row)) + "\n")
print("  ".join(f"{k}={v}" for k, v in zip(["label","quant","c","prompts","tok/s","e2e_ms","tpot_ms","acc"], row)))
EOF
  done
  stop_server
  unset SERVER_PID
done

echo
echo "================ summary ($TSV) ================"
column -t -s $'\t' "$TSV"
