#!/bin/bash
# EAGLE3 多卡训练脚本 for Qwen3.5-4B
# 使用 SpecForge 框架进行离线训练，支持多 GPU

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$SCRIPT_DIR

# ============= 可修改参数 =============
# GPU 配置
NUM_GPUS=6              # 总 GPU 数量
TP_SIZE=1               # Tensor Parallelism 大小
                        # Qwen3.5-4B 较小，推荐 TP_SIZE=1 (数据并行)
                        # 大模型(>70B)才需要 TP_SIZE>1
GPU_IDS="1,2,3,4,5,6"  # CUDA_VISIBLE_DEVICES (逗号分隔)

# 并行策略说明：
# - TP_SIZE=1 (推荐): 数据并行，每块 GPU 处理不同数据，训练速度更快
# - TP_SIZE>1: Tensor Parallelism，模型分片到多块 GPU，用于大模型
# - 对于 Qwen3.5-4B: 使用 TP_SIZE=1，让 6 卡并行处理不同数据

# 数据路径
TRAIN_DATA=$ROOT_DIR/cache/dataset/ultrachat_train_fixed.jsonl
HIDDEN_STATES_DIR=$ROOT_DIR/cache/hidden_states/qwen35_4b_ultrachat
OUTPUT_DIR=$ROOT_DIR/outputs/qwen3.5-4b-eagle3-ultrachat

# 模型路径
TARGET_MODEL_PATH=/home/pairshoe/ljl/train_eagle3/models/Qwen/Qwen3.5-4B
DRAFT_MODEL_CONFIG=$ROOT_DIR/configs/qwen3.5-4b-eagle3.json

# 训练参数
NUM_EPOCHS=1
BATCH_SIZE=1           # 单卡 batch size
MAX_LENGTH=1024
LEARNING_RATE=5e-5
SAVE_INTERVAL=500
LOG_INTERVAL=10
ATTENTION_BACKEND="sdpa"

# 数据限制（用于测试）
NUM_SAMPLES=None        # 只生成前 N 条数据的 hidden states，设为 None 表示全部

# 其他参数
CHAT_TEMPLATE="qwen"
# ==================================

# 环境变量
export HF_DATASETS_CACHE=$ROOT_DIR/cache/hf_datasets
export HF_HOME=$ROOT_DIR/cache/huggingface
export PYTHONUNBUFFERED=1

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate jimoke

# 计算有效 batch size
EFFECTIVE_BATCH_SIZE=$((NUM_GPUS / TP_SIZE * BATCH_SIZE))

echo "======================================"
echo "EAGLE3 多卡训练 - Qwen3.5-4B"
echo "======================================"
echo "GPU: $GPU_IDS"
echo "总 GPU 数: $NUM_GPUS"
echo "TP Size: $TP_SIZE"
echo "DP Size (自动): $((NUM_GPUS / TP_SIZE))"
echo "有效 Batch Size: $EFFECTIVE_BATCH_SIZE"
echo "数据: ultrachat_train_fixed.jsonl"
echo "样本数: $NUM_SAMPLES (测试用)"
echo "Epochs: $NUM_EPOCHS"
echo "======================================"

# 第一步：生成 hidden states (使用单卡避免多进程写入冲突)
echo ""
echo "步骤 1/2: 生成 hidden states (单卡)..."
# 使用第一个 GPU 进行生成
FIRST_GPU=$(echo $GPU_IDS | cut -d',' -f1)
CUDA_VISIBLE_DEVICES=$FIRST_GPU torchrun \
    --standalone \
    --nproc_per_node 1 \
    $ROOT_DIR/scripts/prepare_hidden_states.py \
    --target-model-path $TARGET_MODEL_PATH \
    --enable-aux-hidden-states \
    --data-path $TRAIN_DATA \
    --output-path $HIDDEN_STATES_DIR \
    --chat-template $CHAT_TEMPLATE \
    --max-length $MAX_LENGTH \
    --tp-size 1 \
    --batch-size 8 \
    --num-samples $NUM_SAMPLES \
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
CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path $TARGET_MODEL_PATH \
    --draft-model-config $DRAFT_MODEL_CONFIG \
    --train-data-path $TRAIN_DATA \
    --train-hidden-states-path $HIDDEN_STATES_DIR \
    --output-dir $OUTPUT_DIR \
    --num-epochs $NUM_EPOCHS \
    --batch-size $BATCH_SIZE \
    --tp-size $TP_SIZE \
    --learning-rate $LEARNING_RATE \
    --max-length $MAX_LENGTH \
    --chat-template $CHAT_TEMPLATE \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key "model.language_model.embed_tokens.weight" \
    --lm-head-key "model.language_model.embed_tokens.weight" \
    --attention-backend $ATTENTION_BACKEND \
    --save-interval $SAVE_INTERVAL \
    --log-interval $LOG_INTERVAL \
    --report-to tensorboard

if [ $? -ne 0 ]; then
    echo "❌ 训练失败！"
    exit 1
fi

echo ""
echo "======================================"
echo "✓ 训练完成！"
echo "输出位置: $OUTPUT_DIR"
echo "======================================"
