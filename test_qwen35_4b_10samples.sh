#!/bin/bash
# EAGLE3 测试脚本 - 使用 10 条数据验证 Qwen3.5-4B 训练流程

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$SCRIPT_DIR

# GPU 配置
NUM_GPUS=1
GPU_ID=6

# 环境变量
export HF_DATASETS_CACHE=$ROOT_DIR/cache/hf_datasets
export HF_HOME=$ROOT_DIR/cache/huggingface
export PYTHONUNBUFFERED=1

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate jimoke

# 测试数据路径
TEST_DATA=$ROOT_DIR/cache/dataset/ultrachat_test_10.jsonl
HIDDEN_STATES_DIR=$ROOT_DIR/cache/hidden_states/qwen35_4b_test_10
OUTPUT_DIR=$ROOT_DIR/outputs/qwen3.5-4b-eagle3-test-10

echo "======================================"
echo "EAGLE3 测试 - Qwen3.5-4B (10 条数据)"
echo "======================================"
echo "GPU: $GPU_ID"
echo "数据: ultrachat_test_10.jsonl"
echo "======================================"

# 第一步：生成 hidden states
echo ""
echo "步骤 1/2: 生成 hidden states..."
CUDA_VISIBLE_DEVICES=$GPU_ID torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/prepare_hidden_states.py \
    --target-model-path /home/pairshoe/ljl/train_eagle3/models/Qwen/Qwen3.5-4B \
    --enable-aux-hidden-states \
    --data-path $TEST_DATA \
    --output-path $HIDDEN_STATES_DIR \
    --chat-template qwen \
    --max-length 2048 \
    --tp-size 1 \
    --batch-size 4 \
    --sglang-mem-fraction-static 0.7

if [ $? -ne 0 ]; then
    echo "❌ Hidden states 生成失败！"
    exit 1
fi

echo ""
echo "✓ Hidden states 生成完成！"
echo "保存位置: $HIDDEN_STATES_DIR"

# 第二步：训练 draft model
echo ""
echo "步骤 2/2: 训练 draft model..."
CUDA_VISIBLE_DEVICES=$GPU_ID torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path /home/pairshoe/ljl/train_eagle3/models/Qwen/Qwen3.5-4B \
    --draft-model-config $ROOT_DIR/configs/qwen3.5-4b-eagle3.json \
    --train-data-path $TEST_DATA \
    --train-hidden-states-path $HIDDEN_STATES_DIR \
    --output-dir $OUTPUT_DIR \
    --num-epochs 3 \
    --batch-size 1 \
    --tp-size 1 \
    --learning-rate 5e-5 \
    --max-length 1024 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key "model.language_model.embed_tokens.weight" \
    --lm-head-key "model.language_model.embed_tokens.weight" \
    --attention-backend sdpa \
    --save-interval 100 \
    --log-interval 1 \
    --report-to tensorboard

if [ $? -ne 0 ]; then
    echo "❌ 训练失败！"
    exit 1
fi

echo ""
echo "======================================"
echo "✓ 测试完成！"
echo "输出位置: $OUTPUT_DIR"
echo "======================================"
