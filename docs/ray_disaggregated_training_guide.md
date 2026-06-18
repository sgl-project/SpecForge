# SpecForge Ray 训推分离训练 — 使用文档

## 概述

SpecForge 支持基于 Ray 的训推分离（disaggregated）训练模式，将 target model 推理（rollout）和 draft model 训练分配到不同的 GPU 上，通过 NCCL GPU→GPU 直传或 Ray object store 传输数据。同时支持 Eagle3 和 DFlash 两种 speculative decoding draft model 训练方法。

## 运行模式

### 1. Colocated 模式（训推同卡）

每张 GPU 同时运行 target model 推理和 draft model 训练。适合 GPU 显存充足的场景。

```bash
python scripts/train_eagle3_ray.py \
    --method eagle3 \
    --target-model-path Qwen/Qwen3-8B \
    --draft-model-config configs/qwen3-8b-eagle3.json \
    --train-data-path data/train.jsonl \
    --output-dir outputs/eagle3-colocated \
    --target-model-backend sglang \
    --sglang-mem-fraction-static 0.4 \
    --train-num-gpus 4 \
    --seed 0
```

不需要 `--disaggregate` 标志。每个 TrainWorker 内部加载 target model 并本地生成 rollout 数据，零传输开销。

### 2. Disaggregated 模式（训推分离）

推理和训练在不同 GPU 上运行。适合需要最大化推理吞吐或 GPU 显存不足以同时装两个模型的场景。

#### Ray 传输（默认）

```bash
python scripts/train_eagle3_ray.py \
    --method eagle3 \
    --target-model-path Qwen/Qwen3-8B \
    --draft-model-config configs/qwen3-8b-eagle3.json \
    --train-data-path data/train.jsonl \
    --output-dir outputs/eagle3-disagg \
    --disaggregate \
    --rollout-num-gpus 2 \
    --train-num-gpus 1 \
    --rollout-tp-size 1 \
    --transfer-backend ray \
    --rollout-batch-size 4 \
    --seed 0
```

#### NCCL 传输（推荐）

```bash
python scripts/train_eagle3_ray.py \
    --method eagle3 \
    --target-model-path Qwen/Qwen3-8B \
    --draft-model-config configs/qwen3-8b-eagle3.json \
    --train-data-path data/train.jsonl \
    --output-dir outputs/eagle3-disagg-nccl \
    --disaggregate \
    --rollout-num-gpus 2 \
    --train-num-gpus 1 \
    --rollout-tp-size 1 \
    --transfer-backend nccl \
    --rollout-batch-size 4 \
    --seed 0
```

## 训练方法

### Eagle3

```bash
--method eagle3 \
--draft-model-config configs/qwen3-8b-eagle3.json \
--ttt-length 7 \
--embedding-key model.embed_tokens.weight
```

Eagle3 使用 TTT（Test-Time Training）展开，draft model 通常为 1 层 transformer。需要 target model 的 hidden states + logits。

### DFlash

```bash
--method dflash \
--draft-model-config configs/qwen3-8b-dflash.json \
--block-size 16 \
--num-anchors 512 \
--lm-head-key lm_head.weight \
--embedding-key model.embed_tokens.weight
```

DFlash 使用 block-parallel 生成，draft model 通常为 5 层。只需要 target model 的 hidden states（不需要 logits），传输量比 Eagle3 小约 90%。

## 关键参数

### 通用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--method` | 训练方法：eagle3 或 dflash | eagle3 |
| `--target-model-path` | Target model 的 HF 路径 | 必填 |
| `--draft-model-config` | Draft model 配置文件路径 | 自动生成 |
| `--target-model-backend` | 推理后端：sglang / hf / custom | sglang |
| `--batch-size` | 每个 DP rank 的 batch size | 1 |
| `--max-length` | 最大序列长度 | 2048 |
| `--num-epochs` | 训练轮数 | 10 |
| `--learning-rate` | 学习率 | 1e-4 |

### 分离模式参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--disaggregate` | 启用训推分离模式 | False |
| `--rollout-num-gpus` | 推理 GPU 数量 | 必填 |
| `--train-num-gpus` | 训练 GPU 数量 | 必填 |
| `--rollout-tp-size` | 推理 TP 并行度 | 1 |
| `--train-tp-size` | 训练 TP 并行度（必须为 1） | 1 |
| `--train-sp-ulysses-size` | Ulysses SP 并行度 | 1 |
| `--train-sp-ring-size` | Ring SP 并行度 | 1 |
| `--transfer-backend` | 传输方式：ray 或 nccl | ray |
| `--rollout-batch-size` | 每次 rollout 处理的样本数 | 1 |

### SGLang 后端参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--sglang-mem-fraction-static` | KV cache 显存比例 | 0.4 |
| `--sglang-enable-torch-compile` | 启用 torch.compile 加速 | False |

### 性能调试

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--enable-perf` | 输出每步耗时分解 | False |
| `--log-interval` | 日志打印间隔（步） | 50 |

## 配置示例

### 小规模（3 GPU，开发调试）

```
CUDA_VISIBLE_DEVICES=0,1,2
--rollout-num-gpus 2 --train-num-gpus 1
--rollout-tp-size 1 --rollout-batch-size 4
--transfer-backend nccl
```

GPU 0,1 → 2 个独立 RolloutWorker（round-robin）
GPU 2 → 1 个 TrainWorker（DP=1）

### 中等规模（16 GPU）

```
--rollout-num-gpus 8 --train-num-gpus 8
--rollout-tp-size 8 --rollout-batch-size 4
--train-sp-ulysses-size 8
--transfer-backend nccl
```

GPU 0-7 → 1 个 TP=8 的 RolloutWorker 组
GPU 8-15 → 8 个 TrainWorker（SP=8, DP=1）

### 大规模（80 GPU）

```
--rollout-num-gpus 16 --train-num-gpus 64
--rollout-tp-size 8 --rollout-batch-size 4
--train-sp-ulysses-size 8
--transfer-backend nccl
```

GPU 0-15 → 2 个 TP=8 的 RolloutWorker 组（round-robin）
GPU 16-79 → 64 个 TrainWorker（SP=8, DP=8）
每个 DP group 的 SP leader 从 RolloutWorker 接收数据，broadcast 给组内成员。

## 常见问题

### SGLang OOM

```
RuntimeError: alloc_req_slots runs out of memory
```

降低 `--rollout-batch-size` 或提高 `--sglang-mem-fraction-static`。

### NCCL 超时

```
RuntimeError: NCCL timeout
```

增大 `--dist-timeout`（默认 20 分钟）。首次运行时 SGLang torch.compile 编译可能需要较长时间。

### Placement group 卡住

```
ray.get(pg.ready()) 无响应
```

检查 `ray status` 确认可用 GPU 数量 ≥ `rollout-num-gpus + train-num-gpus`。

### 训练速度慢

1. 使用 `--enable-perf` 查看每步耗时分解
2. 增大 `--rollout-batch-size` 提升推理 GPU 利用率
3. 使用 `--transfer-backend nccl` 替代 ray
4. 确保 rollout GPU 数量足够（prefetch 队列保持满载）
