#!/usr/bin/env bash
# Produce a tuned fused-MoE Triton config for an MoE drafter's expert shape with SGLang's own tuner.
#
#   bash tune_fused_moe_config.sh <path/to/sglang repo> [model config dir] [batch sizes...]
#
# The model config dir only needs a config.json that declares the expert shape as a
# Qwen3MoeForCausalLM (num_experts, num_experts_per_tok, moe_intermediate_size,
# hidden_size); qwen38-dspark-moe/ describes the Qwen3.8-27B DSpark MoE drafter
# (512 experts of width 512, top-10, hidden 5120). Batch sizes are the draft token
# counts per step: DSpark block size x concurrency (7 x {1,2,4,8,16,32,64} by default).
# The tuner uses every visible GPU through ray (~30-75 min for 7 batch sizes); it writes
# E=<experts>,N=<width>,device_name=<GPU>.json into the current directory.
set -euo pipefail
SGLANG_REPO=${1:?path to an sglang checkout (benchmark/kernels/fused_moe_triton/)}
MODEL_DIR=${2:-$(dirname "$0")/qwen38-dspark-moe}
shift $(( $# >= 2 ? 2 : $# ))
BATCH_SIZES=${*:-7 14 28 56 112 224 448}
TUNER=$SGLANG_REPO/benchmark/kernels/fused_moe_triton
# tuning_fused_moe_triton.py imports its sibling common_utils.py, so run it from its own directory.
cd "$TUNER"
python tuning_fused_moe_triton.py --model "$(cd "$MODEL_DIR" && pwd)" --tune --tp-size 1 --dtype auto --batch-sizes $BATCH_SIZES
ls -la E=*.json
