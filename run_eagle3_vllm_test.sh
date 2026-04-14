#!/bin/bash
# 使用 vLLM 测试训练好的 EAGLE3 模型

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# 配置
TARGET_MODEL="/home/pairshoe/ljl/train_eagle3/models/Qwen/Qwen3.5-4B"
DRAFT_MODEL="/home/pairshoe/ljl/train_eagle3/SpecForge/outputs/qwen3.5-4b-eagle3-20k/epoch_2_step_12000"
NUM_PROMPTS=20
GPU_ID=0

echo "=========================================="
echo "EAGLE3 模型测试 (vLLM)"
echo "=========================================="
echo "Target Model: $TARGET_MODEL"
echo "Draft Model: $DRAFT_MODEL"
echo "=========================================="
echo ""

CUDA_VISIBLE_DEVICES=$GPU_ID /home/pairshoe/anaconda3/envs/jimoke/bin/python $SCRIPT_DIR/test_eagle3_vllm.py \
    --model-dir "$TARGET_MODEL" \
    --eagle-dir "$DRAFT_MODEL" \
    --num-prompts $NUM_PROMPTS \
    --num-spec-tokens 4 \
    --output-len 256 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --print-output

echo ""
echo "测试完成！"
