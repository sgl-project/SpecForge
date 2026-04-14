#!/bin/bash
# 测试 EAGLE3 模型效果
# 比较 baseline（无 EAGLE）和 EAGLE3 的推理速度和准确率

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# ============= 配置 =============
# 目标模型路径
TARGET_MODEL="/home/pairshoe/ljl/train_eagle3/models/Qwen/Qwen3.5-4B"
# Draft 模型路径（训练好的输出）
# DRAFT_MODEL="$ROOT_DIR/outputs/qwen3.5-4b-eagle3-20k"
DRAFT_MODEL="/home/pairshoe/ljl/train_eagle3/SpecForge/outputs/qwen3.5-4b-eagle3-20k/epoch_2_step_12000"
# 测试参数
PORT=8005
NUM_SAMPLES=20  # 测试样本数量
BATCH_SIZE=1
STEPS=3
TOPK=1
NUM_DRAFT_TOKENS=4

# GPU 配置
CUDA_VISIBLE_DEVICES=2,3,4,5,6

echo "=========================================="
echo "EAGLE3 模型测试"
echo "=========================================="
echo "Target Model: $TARGET_MODEL"
echo "Draft Model: $DRAFT_MODEL"
echo "=========================================="
echo ""

# ============= 测试 1: Baseline（无 EAGLE）============
echo "=========================================="
echo "测试 1: Baseline（无 EAGLE 推测解码）"
echo "=========================================="

$ROOT_DIR/benchmarks/bench_eagle3.py \
    --model-path $TARGET_MODEL \
    --speculative-draft-model-path $DRAFT_MODEL \
    --port $PORT \
    --config-list $BATCH_SIZE,0,0,0 \
    --benchmark-list mtbench:$NUM_SAMPLES \
    --skip-launch-server \
    --dtype bfloat16 \
    --trust-remote-code \
    --name baseline_no_eagle \
    --output-dir $ROOT_DIR/results

echo ""
echo "Baseline 结果已保存到: $ROOT_DIR/results"
echo ""

# ============= 测试 2: EAGLE3 推测解码 =============
echo "=========================================="
echo "测试 2: EAGLE3 推测解码"
echo "=========================================="

# 先启动 SGLang 服务器
echo "启动 SGLang 服务器（启用 EAGLE3）..."
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 -m sglang.launch_server \
    --model-path $TARGET_MODEL \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path $DRAFT_MODEL \
    --speculative-num-steps $STEPS \
    --speculative-eagle-topk $TOPK \
    --speculative-num-draft-tokens $NUM_DRAFT_TOKENS \
    --mem-fraction-static 0.8 \
    --attention-backend torch_native \
    --tp-size 1 \
    --host 0.0.0.0 \
    --port $PORT \
    --dtype bfloat16 \
    --trust-remote-code &

SERVER_PID=$!
echo "服务器 PID: $SERVER_PID"

# 等待服务器启动
echo "等待服务器启动..."
sleep 30

# 检查服务器是否启动成功
if ! curl -s http://localhost:$PORT/health > /dev/null; then
    echo "❌ 服务器启动失败！"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

echo "✓ 服务器启动成功"
echo ""

# 运行基准测试
$ROOT_DIR/benchmarks/bench_eagle3.py \
    --model-path $TARGET_MODEL \
    --speculative-draft-model-path $DRAFT_MODEL \
    --port $PORT \
    --config-list $BATCH_SIZE,$STEPS,$TOPK,$NUM_DRAFT_TOKENS \
    --benchmark-list mtbench:$NUM_SAMPLES \
    --skip-launch-server \
    --dtype bfloat16 \
    --name eagle3_speculative \
    --output-dir $ROOT_DIR/results

# 关闭服务器
echo "关闭服务器..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo ""
echo "结果位置:"
echo "  Baseline: $ROOT_DIR/results/*baseline*"
echo "  EAGLE3:   $ROOT_DIR/results/*eagle3*"
echo ""
echo "对比方法:"
echo "  1. 查看 JSON 文件中的 metrics 字段"
echo "  2. 关注: total_tokens, time, throughput 等指标"
echo ""


