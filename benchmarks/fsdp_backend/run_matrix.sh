#!/usr/bin/env bash
# Run the FSDP1 vs FSDP2 backend matrix on one node.
#   NGPU=4 OUT=/tmp/bench bash benchmarks/fsdp_backend/run_matrix.sh [extra args]
set -uo pipefail
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
OUT=${OUT:-/tmp/fsdp_backend_bench}
ALGOS=${ALGOS:-"eagle3 dflash2 dspark"}
BACKENDS=${BACKENDS:-"fsdp fsdp2"}
mkdir -p "$OUT"
cd "$(dirname "$0")/../.."
for algo in $ALGOS; do
  for backend in $BACKENDS; do
    echo "=== $algo / $backend / ${NGPU} GPUs ==="
    torchrun --standalone --nproc_per_node="$NGPU" \
      benchmarks/fsdp_backend/bench_fsdp_backends.py \
      --algo "$algo" --backend "$backend" --out-dir "$OUT" "$@" \
      2>&1 | tee "$OUT/$algo-$backend.log" | grep -v "^\s*$" | tail -5
  done
done
python3 benchmarks/fsdp_backend/summarize.py "$OUT"
