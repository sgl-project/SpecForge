#!/bin/bash
# EAGLE3 完整训练脚本 for Qwen3.5-4B
# 使用 SpecForge 框架进行离线训练
# 分两步：先生成 hidden states，再训练 draft model

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$SCRIPT_DIR

# ============= 可修改参数 =============
# GPU 配置
# NUM_GPUS=3
# GPU_ID=3,5,6
NUM_GPUS=1
GPU_ID=6

# 数据路径
# TRAIN_DATA=$ROOT_DIR/cache/dataset/ultrachat_train_fixed.jsonl
TRAIN_DATA=/home/pairshoe/ljl/train_eagle3/regenerated_eagle3_data/common_zh_70k/common_zh_70k_regenerated.jsonl
HIDDEN_STATES_DIR=$ROOT_DIR/cache/sharegpt_hidden_common_zh_70k_states/qwen35_4b_sharegpt
OUTPUT_DIR=$ROOT_DIR/outputs/qwen3.5-4b-eagle3-sharegpt

# 模型路径
TARGET_MODEL_PATH=/home/pairshoe/ljl/train_eagle3/models/Qwen/Qwen3.5-4B
DRAFT_MODEL_CONFIG=$ROOT_DIR/configs/qwen3.5-4b-eagle3.json

# 训练参数
NUM_EPOCHS=5
BATCH_SIZE=1
MAX_LENGTH=4096
LEARNING_RATE=5e-5
SAVE_INTERVAL=500
LOG_INTERVAL=10
ATTENTION_BACKEND="sdpa"  # 可选: sdpa, flex_attention

# 其他参数
TP_SIZE=1
CHAT_TEMPLATE="qwen"
# ==================================

# 环境变量
export HF_DATASETS_CACHE=$ROOT_DIR/cache/hf_datasets
export HF_HOME=$ROOT_DIR/cache/huggingface
export PYTHONUNBUFFERED=1

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate jimoke

echo "======================================"
echo "EAGLE3 完整训练 - Qwen3.5-4B"
echo "======================================"
echo "GPU: $GPU_ID"
echo "数据: common_zh_70k_regenerated.jsonl"
echo "Epochs: $NUM_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Max Length: $MAX_LENGTH"
echo "Learning Rate: $LEARNING_RATE"
echo "Attention Backend: $ATTENTION_BACKEND"
echo "======================================"

# 第一步：生成 hidden states
echo ""
echo "步骤 1/2: 生成 hidden states..."
CUDA_VISIBLE_DEVICES=$GPU_ID torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/prepare_hidden_states.py \
    --target-model-path $TARGET_MODEL_PATH \
    --target-model-backend $TARGET_MODEL_BACKEND \
    --enable-aux-hidden-states \
    --data-path $TRAIN_DATA \
    --output-path $HIDDEN_STATES_DIR \
    --chat-template $CHAT_TEMPLATE \
    --max-length $MAX_LENGTH \
    --tp-size $TP_SIZE \
    --batch-size 8

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
    --target-model-path $TARGET_MODEL_PATH \
    --target-model-backend $TARGET_MODEL_BACKEND \
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
