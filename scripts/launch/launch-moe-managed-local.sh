#!/usr/bin/env bash
# Launch the Qwen3.8-27B DSpark MoE 3-epoch run on this node (intended: 10.13.114.103).
# Lives on /personal so the remote node can execute it. Idempotence: refuses to
# start if a specforge train process or a used GPU is present.
set -euo pipefail
CK=/personal/SpecForge-qwen38-moe
RUN=${1:-qwen3.8-27b-dspark-moe-regen-mixture-v1-3ep}
RECIPE=$CK/examples/configs/online/disaggregated/managed-local/$RUN.yaml
OUT=$CK/outputs/$RUN
if pgrep -f "specforge.cli tr[a]in" > /dev/null; then echo "a specforge train process is already running on $(hostname)" >&2; exit 1; fi
if nvidia-smi --query-gpu=memory.used --format=csv,noheader | grep -qv '^0 MiB'; then echo "GPUs are not idle on $(hostname)" >&2; exit 1; fi
if [ -e "$OUT/control" ]; then echo "control dir exists: $OUT/control (managed_local needs a fresh one)" >&2; exit 1; fi
mkdir -p "$OUT"
cd "$CK"
# OMP_NUM_THREADS: torchrun defaults ranks to 1 thread, which makes the CPU-offloaded AdamW over ~5B fp32 params per rank single-threaded (240 cores on the node).
# expandable_segments: bs4 MoE ranks peaked at 128 GB allocated but ~180 GB reserved with 18 allocator retries in 10 steps (fragmentation).
export PYTHONPATH=/opt/sglang-0518:$CK MC_TRANSFER_TIMEOUT=300 SGLANG_SPEC_CAPTURE_SINK_CLIENTS=4 OMP_NUM_THREADS=${OMP_NUM_THREADS:-32} PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
/opt/sglang/bin/python -c 'import sglang; assert sglang.__version__.startswith("0.5.18"), sglang.__version__; import sglang.srt.spec_capture_sink'
# Host-memory placement: sglang capture schedulers self-bind (MPOL_BIND) to NUMA node 0 (GPUs 0-3),
# and 4 x 100 GiB Mooncake segments + scheduler baseline already fill ~470 GB of node 0's 685 GB.
# Keep the trainer ranks' CPU-offloaded optimizer state (~120 GB/rank) off node 0 by launching the
# whole tree under a node-1 *preferred* policy (falls back to node 0 when node 1 is full; a strict
# membind would OOM the ranks during the checkpoint gather); the schedulers override it for themselves.
# Set TRAINER_MEMNODE="" to disable.
MEMNODE=${TRAINER_MEMNODE-1}
PREFIX=""; [ -n "$MEMNODE" ] && PREFIX="numactl --preferred=$MEMNODE"
setsid nohup $PREFIX /opt/sglang/bin/python -m specforge.cli train -c "$RECIPE" > "$OUT/launch.log" 2>&1 < /dev/null &
echo "launched $RUN on $(hostname), supervisor pid $!, log $OUT/launch.log"
