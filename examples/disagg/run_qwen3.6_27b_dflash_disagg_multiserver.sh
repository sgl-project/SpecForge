#!/bin/bash
# Qwen3.6-27B DFlash, ONLINE disaggregated, MULTI-SERVER on one 8-GPU node:
#   mooncake master           -> CPU  (RDMA/TCP object store)
#   patched SGLang server 0   -> GPUs 0,1 (TP=2, frozen 27B target -> mooncake)
#   patched SGLang server 1   -> GPUs 2,3 (TP=2, same model + capture flags)
#   producer (HTTP driver)    -> CPU  (fans prompts out to BOTH servers)
#   consumer (DFlash trainer) -> GPUs 4,5 (DP=2 over per-rank inboxes)
#
# Multi-server sibling of run_qwen3.6_27b_dflash_disagg.sh. The producer builds
# one SGLangServerCaptureAdapter per URL in DISAGG_SERVER_URLS; each adapter is
# driven by its own RolloutWorker on its own thread, leasing DISJOINT prompts
# from the one controller — both servers prefill concurrently. Every server
# registers a segment with the ONE mooncake master, so the trainer fetches any
# sample by key regardless of which server captured it. A server that dies is
# dropped after its in-flight prompts are returned to the pool; the survivor
# finishes the run.
#
# Both servers MUST be launched with identical model + capture flags (the
# producer's FeatureContract check fails loudly per-sample otherwise).
#
# Prereqs: identical to run_qwen3.6_27b_dflash_disagg.sh (patched sglang 0.5.14,
# mooncake_master on PATH, dataset + weights cached, WANDB_API_KEY or
# --report-to none).
set -uxo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/scripts:${PYTHONPATH:-}"
cd "$ROOT_DIR"

# --- 拓扑结构：2 个推理服务器（每个 TP=2）+ 训练器（DP=2），均可通过环境变量覆盖 ---
SERVER0_GPUS=${SERVER0_GPUS:-"0,1"}          # 推理服务器 0 使用的 GPU 编号（TP=2）
SERVER1_GPUS=${SERVER1_GPUS:-"2,3"}          # 推理服务器 1 使用的 GPU 编号（TP=2）
SERVER_TP=${SERVER_TP:-2}                    # 每个 SGLang 服务器的张量并行度
SERVER0_PORT=${SERVER0_PORT:-30000}          # 推理服务器 0 的 HTTP 端口
SERVER1_PORT=${SERVER1_PORT:-30001}          # 推理服务器 1 的 HTTP 端口
TRAIN_DP=${TRAIN_DP:-2}                      # 训练器的数据并行度（进程数 = nproc_per_node）
CONSUMER_GPUS=${CONSUMER_GPUS:-"4,5"}        # 训练器（consumer）使用的 GPU 编号
# DFlash 目标模型的特征抓取层 ID —— 必须与下方 draft config 中 dflash_config.target_layer_ids
# 保持一致，且两个服务器必须传入相同的值，否则 producer 端 FeatureContract 校验会失败。
AUX_LAYER_IDS=${AUX_LAYER_IDS:-"1 16 31 46 61"}

# --- Mooncake 连接配置（推理服务器 sink 端 + producer + consumer 共用） ---
export MOONCAKE_LOCAL_HOSTNAME=${MOONCAKE_LOCAL_HOSTNAME:-127.0.0.1}                            # 本机对外可达的主机名/IP，供 Mooncake 段注册使用
export MOONCAKE_MASTER_SERVER_ADDR=${MOONCAKE_MASTER_SERVER_ADDR:-127.0.0.1:50051}              # Mooncake master 地址（对象存储元数据入口）
export MOONCAKE_METADATA_SERVER=${MOONCAKE_METADATA_SERVER:-http://127.0.0.1:8080/metadata}     # Mooncake HTTP metadata 服务地址
export MOONCAKE_PROTOCOL=${MOONCAKE_PROTOCOL:-tcp}                                              # 传输协议：tcp 或 rdma
# 每个推理服务器都会向唯一的 master 贡献这么大的内存段；对象是硬固定的（段太小时 put 直接失败而不是驱逐）。
# producer 水位线管理的 in-flight 字节数在 N 个段之间分布，但放置并不均衡 ——
# 需为每个服务器预留高于 watermark/N 的余量（防止倾斜）。
export MOONCAKE_GLOBAL_SEGMENT_SIZE=${MOONCAKE_GLOBAL_SEGMENT_SIZE:-$((48 << 30))}              # 每服务器段大小，默认 48 GiB
# Producer/consumer 只是纯客户端（不注册段）：因为对象不能存活在训练结束前就退出的进程中。
export DISAGG_CLIENT_SEGMENT_SIZE=${DISAGG_CLIENT_SEGMENT_SIZE:-0}                              # 客户端段大小，固定为 0（不贡献段）

export DISAGG_STORE_ID=${DISAGG_STORE_ID:-qwen36-dflash-disagg-2srv}                            # 本次运行的存储命名空间标识
export DISAGG_SERVER_URLS="http://127.0.0.1:$SERVER0_PORT,http://127.0.0.1:$SERVER1_PORT"       # 所有推理服务器 URL，producer 会为每个 URL 创建一个 adapter/worker
export DISAGG_REF_CHANNEL=${DISAGG_REF_CHANNEL:-$ROOT_DIR/outputs/$DISAGG_STORE_ID/refs.jsonl}  # 控制面 ref 通道文件（只传 SampleRef 元数据，不含张量）
# 仅训练端使用（rank-0 的 ledger + 每 rank 的 inbox）；producer 不应看到 db —— 下方 launcher 只把它传给 consumer。
DISAGG_DB=${DISAGG_DB:-$ROOT_DIR/outputs/$DISAGG_STORE_ID/run.db}                               # 训练端 SQLite ledger 路径
DISAGG_INBOX_DIR=${DISAGG_INBOX_DIR:-$ROOT_DIR/outputs/$DISAGG_STORE_ID/inboxes}                # 每个 rank 的样本 inbox 目录
export DISAGG_MAX_PROMPTS=${DISAGG_MAX_PROMPTS:-400}                                            # producer 派发的最大 prompt 数（0 表示不限）
export DISAGG_MAX_STEPS=${DISAGG_MAX_STEPS:-0}                                                  # consumer 训练最大步数（0 表示走完一个 epoch）
export DISAGG_LOG_INTERVAL=${DISAGG_LOG_INTERVAL:-1}                                            # 训练日志打印间隔（步）
REPORT_TO=${REPORT_TO:-wandb}  # 训练结果上报后端；设为 none 可禁用 W&B

rm -rf "$(dirname "$DISAGG_REF_CHANNEL")" "$DISAGG_DB"
mkdir -p "$(dirname "$DISAGG_REF_CHANNEL")"
: > "$DISAGG_REF_CHANNEL"

cleanup() { kill "${MASTER_PID:-}" "${SERVER0_PID:-}" "${SERVER1_PID:-}" "${PRODUCER_PID:-}" 2>/dev/null || true; }
trap cleanup EXIT

# --- mooncake master ---
mooncake_master --enable-http-metadata-server=true &
MASTER_PID=$!
sleep 3

# --- 打过 spec-capture 补丁的 SGLang 服务器：冻结的 27B 目标模型，每个 TP=2，开启特征抓取 ---
launch_server() { # $1=gpus $2=port
    CUDA_VISIBLE_DEVICES=$1 MOONCAKE_LOCAL_HOSTNAME=$MOONCAKE_LOCAL_HOSTNAME \
        python -m sglang.launch_server \
            --model-path Qwen/Qwen3.6-27B \
            --trust-remote-code \
            --skip-tokenizer-init \
            --tp-size "$SERVER_TP" \
            --mem-fraction-static 0.85 \
            --chunked-prefill-size -1 \
            --disable-radix-cache \
            --enable-spec-capture \
            --spec-capture-method dflash \
            --spec-capture-aux-layer-ids $AUX_LAYER_IDS \
            --port "$2" &
}
# --model-path                    : 冻结的目标模型（HuggingFace 路径）
# --trust-remote-code             : 允许加载模型仓库中的自定义代码
# --skip-tokenizer-init           : 跳过服务器端 tokenizer 初始化（producer 侧完成分词）
# --tp-size                       : 张量并行度
# --mem-fraction-static           : 静态显存占用比例（KV cache 上限），0.85 = 显存的 85%
# --chunked-prefill-size          : 分块 prefill 大小，-1 表示禁用分块 prefill
# --disable-radix-cache           : 关闭 RadixAttention 前缀缓存（保证捕获与训练特征一一对应）
# --enable-spec-capture           : 启用补丁引入的推测采样特征抓取开关
# --spec-capture-method dflash    : 抓取方式选择 DFlash（上下文隐藏态 + 辅助层）
# --spec-capture-aux-layer-ids    : 需要抓取的辅助层 ID，须与 draft config 中 target_layer_ids 一致
# --port                          : 服务器 HTTP 监听端口
launch_server "$SERVER0_GPUS" "$SERVER0_PORT"
SERVER0_PID=$!
launch_server "$SERVER1_GPUS" "$SERVER1_PORT"
SERVER1_PID=$!

# wait for BOTH servers before driving prompts (die fast if either dies)
for port_pid in "$SERVER0_PORT:$SERVER0_PID" "$SERVER1_PORT:$SERVER1_PID"; do
    port=${port_pid%%:*}; pid=${port_pid##*:}
    until curl -sf "http://127.0.0.1:$port/health" > /dev/null; do
        if ! kill -0 "$pid" 2>/dev/null; then echo "server on :$port died"; exit 1; fi
        sleep 5
    done
done

# 训练配方与 run_qwen3.6_27b_dflash_disagg.sh 一致；单节点 demo 下 max-length 设为 2048。
# 注意 batch-size 是"每 rank"的大小，global batch = batch_size * TRAIN_DP。
ARGS=(
    --target-model-path Qwen/Qwen3.6-27B                                # 目标模型（教师）路径，与推理服务器一致
    --target-model-backend hf                                           # 目标模型加载后端（consumer 侧仅需权重壳做前向对齐）
    --trust-remote-code                                                 # 允许加载模型仓库自定义代码
    --draft-config-path "$ROOT_DIR/configs/qwen3.6-27b-dflash.json"     # DFlash 草稿模型配置（含 target_layer_ids 等）
    --embedding-key model.language_model.embed_tokens.weight            # 从 target 权重复用 embedding 的 state_dict key
    --lm-head-key lm_head.weight                                        # 从 target 权重复用 lm_head 的 state_dict key
    --mask-token-id 248070                                              # 训练时用于对齐/掩码的特殊 token id
    --train-data-path "$ROOT_DIR/cache/dataset/nemotron_v2_train.jsonl" # 训练数据集路径（prompt jsonl）
    --chat-template qwen3.5                                             # 聊天模板名称
    --max-length 2048                                                   # 最大序列长度（prompt+response 上限）
    --batch-size 1                                                      # 每 rank 的 batch size
    --learning-rate 6e-4                                                # 学习率
    --warmup-ratio 0.04                                                 # warmup 比例（占总步数）
    --max-grad-norm 1.0                                                 # 梯度裁剪阈值
    --attention-backend flex_attention                                  # 训练侧 attention 后端（flex_attention）
    --block-size 16                                                     # DFlash 训练的 block 长度
    --num-anchors 512                                                   # DFlash 采样锚点数
    --loss-decay-gamma 7.0                                              # 沿位置的损失衰减系数 γ
    --num-epochs 1                                                      # 训练轮数
    --seed 42                                                           # 随机种子
    --save-interval 1000000                                             # 保存 checkpoint 的步间隔（此处足够大 = 不保存中间态）
)

LAUNCHER=$SCRIPT_DIR/run_disagg_dflash.py  # 统一入口脚本，通过 DISAGG_ROLE 分流 producer / consumer

# --- producer：纯 CPU 的 HTTP 驱动，将 prompt 分发到两个推理服务器 ---
# CUDA_VISIBLE_DEVICES=""    : 强制不使用 GPU（producer 只做 HTTP 调度与 ref 记录）
# DISAGG_ROLE=producer       : 由 launcher 识别为 producer 分支
# --output-dir               : producer 输出目录（日志、ref 通道副本等）
DISAGG_ROLE=producer CUDA_VISIBLE_DEVICES="" \
    python "$LAUNCHER" "${ARGS[@]}" \
        --output-dir "$ROOT_DIR/outputs/qwen36-disagg-2srv-producer" &
PRODUCER_PID=$!

# --- consumer：DP=$TRAIN_DP 的训练进程池；rank 0 兼任样本分发与 ledger 维护 ---
# CUDA_VISIBLE_DEVICES        : 训练器使用的 GPU
# DISAGG_ROLE=consumer        : 由 launcher 识别为 consumer 分支
# DISAGG_DB / DISAGG_INBOX_DIR: 仅训练端可见的 ledger 与每 rank 的 inbox 目录
# torchrun --rdzv-*           : c10d 单节点 rendezvous，端口 29702
# --nnodes / --nproc_per_node : 节点数与每节点进程数（= DP）
# --output-dir                : consumer 输出目录（checkpoint、日志）
# --report-to                 : 训练结果上报（wandb 或 none）
# --wandb-project / --wandb-name : W&B 项目与本次实验名
CUDA_VISIBLE_DEVICES=$CONSUMER_GPUS DISAGG_ROLE=consumer \
    DISAGG_DB=$DISAGG_DB DISAGG_INBOX_DIR=$DISAGG_INBOX_DIR \
    torchrun --rdzv-backend c10d --rdzv-endpoint localhost:29702 \
        --nnodes 1 --nproc_per_node "$TRAIN_DP" "$LAUNCHER" "${ARGS[@]}" \
        --output-dir "$ROOT_DIR/outputs/qwen36-disagg-2srv-consumer" \
        --report-to "$REPORT_TO" \
        --wandb-project qwen36-dflash-disagg \
        --wandb-name qwen36-27b-dflash-2srv-tp2-dp$TRAIN_DP

wait $PRODUCER_PID
echo "DISAGG36-2SRV-DONE"
