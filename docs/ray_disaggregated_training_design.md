# SpecForge Ray 训推分离训练 — 设计文档

## 1. 架构概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Driver 进程 (无 GPU)                         │
│                                                                     │
│  RayOrchestrator                                                    │
│    ├── Ray 集群管理                                                  │
│    ├── 数据集预构建 + Driver DataLoader                              │
│    ├── TrainingPipeline (调度核心)                                   │
│    └── 训练循环 (epoch → step → eval → checkpoint)                  │
│                                                                     │
│  TrainingPipeline                                                   │
│    ├── prefetch 队列管理                                             │
│    ├── DP-aware 数据分发                                             │
│    └── 模式分发: colocated / Ray disagg / NCCL disagg               │
└──────────────┬──────────────────────────────┬───────────────────────┘
               │ Ray remote                    │ Ray remote
               ▼                               ▼
┌──────────────────────────────┐  ┌────────────────────────────────────┐
│   RolloutWorkerGroup         │  │      TrainWorkerGroup              │
│                              │  │                                    │
│  每个 Actor 占 1 GPU          │  │  每个 Actor 占 1 GPU               │
│  多个 Actor 组成 TP 组         │  │  多个 Actor 组成 DP/SP 组          │
│                              │  │                                    │
│  例: 16 GPU, TP=8            │  │  例: 64 GPU, SP=8, DP=8           │
│  → 2 个 TP 组 (各 8 Actor)   │  │  → 8 个 SP 组 (各 8 Actor)        │
│  → TP 组内 NCCL 通信          │  │  → SP 组内 broadcast               │
│  → TP 组间 round-robin       │  │  → DP 组间 FSDP gradient sync     │
│                              │  │                                    │
│  RolloutWorker × N           │  │  TrainWorker × M                   │
│  ├── Target Model (SGLang)   │  │  ├── Draft Model (FSDP)            │
│  └── generate_and_send       │  │  ├── Online Wrapper (Eagle3/DFlash)│
│                              │  │  └── run_step / save_checkpoint    │
└──────────────────────────────┘  └────────────────────────────────────┘
```

> **注意：** 每个 Ray Actor 通过 `@ray.remote(num_gpus=1)` 声明，独占 1 张 GPU。
> TP/DP/SP 并行不是单个 Actor 使用多 GPU，而是多个单 GPU Actor 通过
> `torch.distributed` 进程组协调工作。这与 torchrun 模式的原理相同——
> 每个进程 1 GPU，区别在于 Ray 模式由 placement group 管理 GPU 分配，
> 由 `distributed_ray.py` 管理进程组初始化。

## 2. 核心组件

### 2.1 RayOrchestrator (`orchestrator.py`)

顶层协调器，运行在 driver 进程中（不占 GPU）。职责：

1. 初始化 Ray 集群
2. 解析 draft model config，确定 layer capture IDs
3. 预构建数据集缓存（避免 worker 中 fork 死锁）
4. 预算 `total_steps`（LR scheduler 需要）
5. 构建 driver 端 DataLoader（disaggregated 模式）
6. 创建 worker groups
7. 驱动训练循环

```python
orchestrator = RayOrchestrator(args)
orchestrator.run()      # 训练主循环
orchestrator.shutdown()  # 清理
```

### 2.2 TrainingPipeline (`pipeline.py`)

调度核心，管理 rollout 和 training 之间的数据流。支持三种模式：

| 模式 | 数据路径 | 适用场景 |
|------|----------|----------|
| Colocated | 本地 GPU，零传输 | 显存充足 |
| Ray disagg | GPU→CPU→object store→CPU→GPU | 调试/兼容 |
| NCCL disagg | GPU→GPU 直传 | 生产环境 |

#### Prefetch 队列机制

```
_fill_nccl_queue():
  while queue 未满:
    for dp_idx in range(dp_size):
      fetch rollout_batch_size 个 batch (driver DataLoader)
      dispatch 给 RolloutWorker (round-robin TP groups)
      RolloutWorker 异步执行 forward + NCCL send
    记录 (send_refs, src_rank, split_count) 到队列

_run_train_step_nccl():
  1. _ensure_nccl_current() → 从队列取下一个 entry
  2. _fill_nccl_queue()     → 补充队列（瞬间完成）
  3. train_step_nccl_async() → 启动训练（非阻塞）
  4. ray.get(train_refs[0])  → 等待训练完成
```

#### Split 缓存机制

当 `rollout_batch_size=4` 时，一次 rollout 产出 4 个样本的结果。Pipeline 将其拆为 4 个 train step：

```
Entry (split_count=4):
  Step N:   split_index=0, nccl_src_rank=0  → recv + 缓存
  Step N+1: split_index=1, nccl_src_rank=-1 → 用缓存
  Step N+2: split_index=2, nccl_src_rank=-1 → 用缓存
  Step N+3: split_index=3, nccl_src_rank=-1 → 用缓存
```

### 2.3 RolloutWorkerGroup / RolloutWorker (`resource_manager.py`, `rollout_worker.py`)

管理 target model 推理。每个 RolloutWorker 是一个 `@ray.remote(num_gpus=1)` 的 Ray Actor。

#### TP 组结构

```
num_workers=16, tp_size=8 → 2 个 TP 组
  TP group 0: rank 0-7  (tp_rank=0 产出结果)
  TP group 1: rank 8-15 (tp_rank=0 产出结果)
```

Pipeline round-robin 在 TP 组之间轮换，实现流水线并行。

#### 方法分发

```python
if method == "eagle3":
    target_model = get_eagle3_target_model(...)
    target_model.set_aux_hidden_states_layers(capture_layer_ids)
    # 输出: hidden_states + target logits
elif method == "dflash":
    target_model = get_dflash_target_model(...)
    target_model.set_capture_layers(capture_layer_ids)
    # 输出: hidden_states only (无 logits，传输量降 ~90%)
```

RolloutWorker 不感知 DP — 每次固定处理 `rollout_batch_size` 个样本，发给单个目标。

### 2.4 TrainWorkerGroup / TrainWorker (`resource_manager.py`, `train_worker.py`)

管理 draft model 训练。支持 DP + SP 并行。

#### 进程组结构（64 GPU, SP=8, DP=8 示例）

```
全局组: rank 0-79 (rollout + train)
Train 子组: rank 16-79
  DP group 0: rank 16,24,32,40,48,56,64,72 (跨 SP 组的同位置 rank)
  SP group 0: rank 16-23 (DP rank 0 的 8 个 SP worker)
  SP group 1: rank 24-31 (DP rank 1 的 8 个 SP worker)
  ...
```

#### 数据接收路径（NCCL 模式）

```
SP leader (sp_rank=0):
  NCCL recv from RolloutWorker → broadcast to SP group

Non-leader (sp_rank>0):
  recv broadcast from SP leader

所有 worker:
  _forward() → backward() → FSDP gradient sync → optimizer.step()
```

#### 方法分发

```python
if method == "eagle3":
    online_model = OnlineEagle3Model(draft_model, length=ttt_length)
    # forward: (input_ids, attention_mask, target, loss_mask, hidden_states)
    # returns: [loss_0, ..., loss_6], [acc_0, ..., acc_6]
elif method == "dflash":
    online_model = OnlineDFlashModel(draft_model, target_lm_head, target_embed_tokens, ...)
    # forward: (input_ids, hidden_states, loss_mask)
    # returns: [loss], [acc]  (包装为列表，统一接口)
```

### 2.5 RolloutBatch (`worker_utils.py`)

跨 worker 传输的数据容器：

```python
@dataclass
class RolloutBatch:
    input_ids: torch.Tensor          # (B, seq_len)
    attention_mask: torch.Tensor     # (B, seq_len)
    loss_mask: torch.Tensor          # (B, seq_len)
    hidden_states: torch.Tensor      # (B, seq_len, N*H)
    target: Optional[torch.Tensor]   # (B, seq_len, V) — Eagle3 only, None for DFlash
```

## 3. 数据传输协议

### 3.1 NCCL 点对点传输

```
Sender (RolloutWorker):
  1. send header (1, 6) int64    → [num_present_fields, ...]
  2. send metadata (N, 6) int64  → [field_idx, dtype, ndim, dim0, dim1, dim2] × N
  3. send tensor × N             → 实际数据

Receiver (TrainWorker):
  1. recv header → 知道有几个 field
  2. recv metadata → 知道每个 field 的 shape 和 dtype
  3. 分配 GPU buffer + recv tensor × N
```

Optional field（如 `target`）通过 presence mask 处理：DFlash 不发送 `target`，receiver 端该字段为 None。

### 3.2 SP 组内广播

```
SP leader (sp_rank=0):
  1. broadcast header
  2. broadcast metadata
  3. broadcast tensor × N

Non-leader:
  1. recv broadcast header → 分配 metadata buffer
  2. recv broadcast metadata → 分配 tensor buffers
  3. recv broadcast tensor × N
```

### 3.3 Ray Object Store 传输（备选）

```
RolloutWorker:
  tensors.cpu() → Ray 序列化 → object store (共享内存)

TrainWorker:
  ray.get(ref) → 反序列化 → pin_memory() → .cuda()
```

## 4. 分布式初始化

### 4.1 NCCL 模式

```
所有 actor 共享一个全局 NCCL 进程组:
  init_global_distributed(global_rank, global_world_size, master_addr, master_port)
    → dist.init_process_group("nccl")

各组创建本地子组 (use_local_synchronization=True):
  RolloutWorker: init_rollout_subgroup(rollout_ranks, tp_size)
    → dist.new_group(rollout_ranks) → TP/DP device mesh
  TrainWorker: init_train_subgroup(train_ranks, tp_size, sp_ulysses, sp_ring)
    → dist.new_group(train_ranks) → TP/DP/SP device mesh

use_local_synchronization=True 确保子组创建不需要全局同步。
```

### 4.2 Ray 模式

```
各组独立初始化:
  RolloutWorker: init_rollout_distributed(rank, world_size, addr, port)
    → 独立的 dist.init_process_group()
  TrainWorker: init_train_distributed(rank, world_size, addr, port)
    → 独立的 dist.init_process_group()

无全局进程组，跨组通信走 Ray object store。
```

### 4.3 死锁避免

NCCL 模式下 `dist.init_process_group(world_size=N)` 需要所有 N 个 rank 同时加入。解决方案：

```python
# resource_manager.py
rollout_group = RolloutWorkerGroup(...)  # 创建 actor，不等 ready
train_group = TrainWorkerGroup(...)      # 创建 actor，不等 ready
# 所有 actor 并行执行 init_process_group
rollout_group.wait_ready()               # 统一等待
train_group.wait_ready()
```

## 5. GPU 资源管理

### 5.1 Placement Group

```python
bundles = [{"GPU": 1, "CPU": 1} for _ in range(total_gpus)]
pg = placement_group(bundles, strategy="PACK")
```

- `PACK` 策略优先将所有 bundle 放在同一节点
- RolloutWorker 绑定 bundle 0 ~ rollout_num_gpus-1
- TrainWorker 绑定 bundle rollout_num_gpus ~ total-1
- 确保两组 GPU 不交叉

### 5.2 SGLang 资源配置

```python
sglang_max_running_requests = target_batch_size * rollout_batch_size
sglang_max_total_tokens = max_running_requests * max_length
```

`rollout_batch_size` 决定每次推理的并发请求数，直接影响 KV cache 需求。

## 6. DP-Aware 数据分发

### 6.1 问题

DP=8 意味着 8 个 DP rank 需要不同的训练数据。RolloutWorker 不感知 DP。

### 6.2 解决方案

Pipeline 为每个 DP group 独立调度 rollout：

```
每个逻辑训练步:
  for dp_idx in range(dp_size):
    data = driver_dataloader.next() × rollout_batch_size
    rollout_group.generate_and_send_single(tp_idx, data, [sp_leader[dp_idx]])

RolloutWorker 每次处理 rollout_batch_size 个样本，发给 1 个 SP leader。
SP leader broadcast 给组内成员。
```

### 6.3 Round-Robin

```
Prefetch entry 0: TP group 0 处理 dp_size 个 rollout
Prefetch entry 1: TP group 1 处理 dp_size 个 rollout
Prefetch entry 2: TP group 0 ...
```

多个 TP group 的 prefetch entry 并行排队，实现流水线重叠。

## 7. 统一 Eagle3 / DFlash 接口

### 7.1 抽象层

| 组件 | Eagle3 | DFlash | 统一接口 |
|------|--------|--------|----------|
| Target model | `get_eagle3_target_model()` | `get_dflash_target_model()` | `method` 参数分发 |
| Layer capture | `set_aux_hidden_states_layers()` | `set_capture_layers()` | `capture_layer_ids` |
| Draft model | `AutoEagle3DraftModel` | `DFlashDraftModel` | `method` 参数分发 |
| Online wrapper | `OnlineEagle3Model` | `OnlineDFlashModel` | `_online_model` |
| Forward 返回 | `[loss×7], [acc×7]` | `loss, acc` | DFlash 包装为 `[loss], [acc]` |
| RolloutBatch | `target` 有值 | `target=None` | Optional field |

### 7.2 统一 loss 计算

```python
# 两种方法共用同一段代码:
ploss_weight = [0.8**i for i in range(len(plosses))]
ploss = sum(weight[i] * plosses[i]) / accumulation_steps
ploss.backward()

# Eagle3: len(plosses)=7, 加权衰减
# DFlash: len(plosses)=1, weight=1.0, 等价于 ploss = loss / accumulation_steps
```

## 8. 文件结构

```
scripts/
  train_eagle3_ray.py          # 入口脚本，参数解析

specforge/ray_trainer/
  orchestrator.py              # 顶层协调器
  pipeline.py                  # 调度核心（prefetch + 模式分发）
  resource_manager.py          # Worker group 管理 + GPU 分配

specforge/ray_workers/
  rollout_worker.py            # RolloutWorker Ray Actor
  train_worker.py              # TrainWorker Ray Actor
  worker_utils.py              # RolloutBatch + NCCL 传输 + 工具函数

specforge/
  distributed_ray.py           # Ray 环境下的 torch.distributed 初始化
  distributed.py               # 进程组管理（TP/DP/SP）
  args.py                      # 参数定义（SGLang/Disaggregate/Tracker/Ray）

examples/
  run_qwen3_8b_eagle3_ray_colocated.sh       # Eagle3 colocated 示例
  run_qwen3_8b_eagle3_ray_disaggregated.sh   # Eagle3 disaggregated 示例
  run_qwen3_8b_dflash_ray_disaggregated.sh   # DFlash disaggregated 示例
```

## 9. 性能优化总结

| 优化 | 效果 |
|------|------|
| NCCL GPU→GPU 直传 | 消除 4 次 CPU 内存拷贝 (3-9s → <0.2s/step) |
| Driver 端 DataLoader | 消除 fetch_batch 的 Ray RPC 开销 |
| rollout_batch_size | 提升 RolloutWorker GPU 利用率 |
| Double-buffered prefetch | 多 TP group round-robin，保持流水线满载 |
| 跳过非日志步 all_reduce | 减少每步同步开销 |
| SP broadcast | 8 次 send → 1 次 send + 1 次 broadcast |
| Colocated 零拷贝 | tensor 全程留在 GPU，不经过 CPU |
| Pinned memory + non_blocking | CPU→GPU 传输并行化（Ray 模式） |
| Placement group | 确定性 GPU 分配，避免资源冲突 |
