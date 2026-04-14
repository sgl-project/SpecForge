#!/bin/bash
# EAGLE3 训练脚本 for Qwen3.5-4B (8k 条数据)
# 使用 SpecForge 框架进行离线训练

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$SCRIPT_DIR

# ============= 配置参数 =============
NUM_SAMPLES=20000         # 只使用 2w 条数据
NUM_GPUS=1                # 单卡生成
GPU_ID=1                 # 用于生成 hidden states 的 GPU

# 多卡训练配置
TRAIN_GPUS=5              # 训练时使用 5 卡
TRAIN_GPU_IDS="2,3,4,5,6"
TP_SIZE=1                 # 数据并行

# 数据路径
TRAIN_DATA=$ROOT_DIR/cache/dataset/ultrachat_train_fixed.jsonl
HIDDEN_STATES_DIR=$ROOT_DIR/cache/hidden_states/qwen35_4b_20k
OUTPUT_DIR=$ROOT_DIR/outputs/qwen3.5-4b-eagle3-20k

# 模型路径
TARGET_MODEL_PATH=/home/pairshoe/ljl/train_eagle3/models/Qwen/Qwen3.5-4B
DRAFT_MODEL_CONFIG=$ROOT_DIR/configs/qwen3.5-4b-eagle3.json

# 训练参数
NUM_EPOCHS=3
BATCH_SIZE=1
MAX_LENGTH=1024
LEARNING_RATE=5e-5
SAVE_INTERVAL=500
LOG_INTERVAL=20
ATTENTION_BACKEND="sdpa"
CHAT_TEMPLATE="qwen"
# ==================================

# 环境变量
export HF_DATASETS_CACHE=$ROOT_DIR/cache/hf_datasets
export HF_HOME=$ROOT_DIR/cache/huggingface
export PYTHONUNBUFFERED=1

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate jimoke

echo "=========================================="
echo "EAGLE3 训练 - Qwen3.5-4B (2w 条数据)"
echo "=========================================="
echo "数据量: $NUM_SAMPLES 条"
echo "生成 GPU: $GPU_ID"
echo "训练 GPU: $TRAIN_GPU_IDS ($TRAIN_GPUS 卡)"
echo "数据集: ultrachat_train_fixed.jsonl"
echo "Epochs: $NUM_EPOCHS"
echo "=========================================="

# 第一步：生成 hidden states (单卡)
echo ""
echo "步骤 1/2: 生成 $NUM_SAMPLES 条 hidden states..."
CUDA_VISIBLE_DEVICES=$GPU_ID torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/prepare_hidden_states.py \
    --target-model-path $TARGET_MODEL_PATH \
    --enable-aux-hidden-states \
    --data-path $TRAIN_DATA \
    --output-path $HIDDEN_STATES_DIR \
    --chat-template $CHAT_TEMPLATE \
    --max-length $MAX_LENGTH \
    --num-samples $NUM_SAMPLES \
    --tp-size $TP_SIZE \
    --batch-size 8 \
    --sglang-mem-fraction-static 0.6

if [ $? -ne 0 ]; then
    echo "❌ Hidden states 生成失败！"
    exit 1
fi

echo ""
echo "✓ Hidden states 生成完成！"
echo "保存位置: $HIDDEN_STATES_DIR"
du -sh $HIDDEN_STATES_DIR

# 第二步：训练 draft model (多卡)
echo ""
echo "步骤 2/2: 训练 draft model ($TRAIN_GPUS 卡)..."
CUDA_VISIBLE_DEVICES=$TRAIN_GPU_IDS torchrun \
    --standalone \
    --nproc_per_node $TRAIN_GPUS \
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
echo "=========================================="
echo "✓ 训练完成！"
echo "输出位置: $OUTPUT_DIR"
echo "=========================================="
