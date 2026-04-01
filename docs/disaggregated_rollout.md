# 基于 Ray 的训推分离/合并架构设计

## 概览

本文档描述将 SpecForge 的 online 训练从纯 `torch.distributed` 迁移到 **Ray** 管理的分布式架构，并新增对**训推分离（disaggregated rollout）**的支持。

通过 `--disaggregate` 开关控制两种模式：

| 模式 | 含义 | GPU 分配 |
|------|------|---------|
| **合并（colocated，默认）** | 推理与训练在同一批 GPU 上串行执行 | 所有 GPU 同时承载 rollout worker 和 train worker |
| **分离（disaggregated）** | 推理 GPU 和训练 GPU 完全隔离 | `rollout_num_gpus` 专用于目标模型推理；`train_num_gpus` 专用于 draft 模型训练 |

---

## 动机与收益

现有实现将目标模型（`SGLangEagle3TargetModel`）与 draft 模型（`Eagle3DraftModel`）放在同一进程、同一 GPU 上，主要痛点：

1. **显存竞争**：大型目标模型（如 70B）占用大量 KV cache 显存，与 FSDP shards 竞争。
2. **计算串行**：rollout 阶段 GPU 全部用于推理，training 阶段全部用于反向传播，两阶段无法重叠。
3. **扩展瓶颈**：若目标模型需要 TP=8，剩余给训练的 GPU 数量就被固定绑定。

引入 Ray 后：

- **分离模式**可让更多/更好的 GPU 专注推理，另一批 GPU 专注训练。
- **合并模式**下 Ray 依然提供统一的资源调度、容错与监控接口，与现有 torch.distributed 逻辑兼容。
- 未来可在 rollout 与训练之间插入 **async pipeline**，使两者时间重叠（本文档预留接口但不强制实现）。

---

## 目录结构

```
SpecForge/
├── specforge/
│   ├── args.py                         # 扩展：新增 RayArgs、DisaggregateArgs
│   ├── distributed.py                  # 保持不变（torch.distributed 工具）
│   ├── distributed_ray.py              # 新增：Ray 下的 distributed group 初始化工具
│   │
│   ├── ray_workers/                    # 新增目录
│   │   ├── __init__.py
│   │   ├── rollout_worker.py           # RolloutWorker Ray Actor
│   │   ├── train_worker.py             # TrainWorker Ray Actor
│   │   └── worker_utils.py             # 两类 worker 共享的工具函数
│   │
│   └── ray_trainer/                    # 新增目录
│       ├── __init__.py
│       ├── orchestrator.py             # Eagle3RayOrchestrator：顶层协调器
│       ├── pipeline.py                 # 一个训练 step 的完整流水线逻辑
│       └── resource_manager.py         # Ray placement group 与 GPU 资源管理
│
└── scripts/
    └── train_eagle3_ray.py             # 新增：基于 Ray 的训练入口脚本
```

---

## 各文件详细说明

---

### `specforge/args.py`（扩展现有文件）

在现有 `SGLangBackendArgs`、`TrackerArgs` 基础上新增两个 dataclass：

#### `RayArgs`

管理 Ray cluster 初始化相关参数。

```python
@dataclass
class RayArgs:
    ray_address: str = None          # Ray cluster 地址，None 表示本地启动
    ray_num_cpus: int = None         # 本地启动时分配的 CPU 数，None 表示自动
    ray_num_gpus: int = None         # 本地启动时分配的 GPU 总数，None 表示自动检测
    ray_namespace: str = "specforge" # Ray namespace，用于 actor 命名隔离
```

**需要实现：**
- `add_args(parser)` 静态方法：向 `argparse.ArgumentParser` 注册以上参数。
- `from_args(args)` 静态方法：从 `argparse.Namespace` 构造 `RayArgs`。

#### `DisaggregateArgs`

控制是否训推分离及 GPU 分配策略。

```python
@dataclass
class DisaggregateArgs:
    disaggregate: bool = False           # 是否开启训推分离
    rollout_num_gpus: int = None         # 推理侧总 GPU 数（分离模式必填）
    train_num_gpus: int = None           # 训练侧总 GPU 数（分离模式必填；= dp_size * train_tp_size）
    rollout_tp_size: int = 1             # 推理侧目标模型 TP size
    train_tp_size: int = 1               # 训练侧 draft 模型 TP size
    train_sp_ulysses_size: int = 1       # 训练侧 Ulysses SP size
    train_sp_ring_size: int = 1          # 训练侧 Ring SP size
    transfer_backend: str = "ray"        # rollout→train 数据传输方式："ray"（object store）或 "nccl"
    rollout_async: bool = False          # 是否异步 rollout（推理与训练时间重叠，高级特性）
```

**需要实现：**
- `add_args(parser)` 静态方法。
- `from_args(args)` 静态方法。
- `validate()` 方法，执行以下检查：
  - 分离模式下 `rollout_num_gpus` 与 `train_num_gpus` 不为空。
  - `rollout_num_gpus % rollout_tp_size == 0`。
  - `train_num_gpus % train_tp_size == 0`。
  - `train_num_gpus % (train_sp_ulysses_size * train_sp_ring_size) == 0`。
  - SP > 1 时 `attention_backend` 必须为 `"usp"`，且 `batch_size == 1`（与现有 `sp_sanity_check` 逻辑一致）。

---

### `specforge/distributed_ray.py`（新增文件）

Ray 下各 worker 内部的 `torch.distributed` 初始化工具。与现有 `distributed.py` 的区别在于：

1. **不依赖 `torchrun` / `MASTER_ADDR` 环境变量**，而是通过 Ray 的 actor placement 获取 rank/world_size 信息后显式调用 `dist.init_process_group`。
2. 提供两套进程组：一套给 rollout workers（TP group），一套给 train workers（DP+SP+TP group）。

**需要实现的函数：**

```python
def init_rollout_distributed(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    tp_size: int,
    timeout_minutes: int = 20,
) -> None:
    """
    在 RolloutWorker 内部调用。
    初始化 rollout 侧的 torch.distributed process group（仅 TP group），
    设置 CUDA device，并将 TP group 写入 distributed.py 中的 _TP_GROUP 全局变量。
    """

def init_train_distributed(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    tp_size: int,
    sp_ulysses_size: int = 1,
    sp_ring_size: int = 1,
    timeout_minutes: int = 20,
) -> None:
    """
    在 TrainWorker 内部调用。复用现有 distributed.py 的 init_distributed() 全部逻辑，
    唯一区别是通过显式设置 os.environ["MASTER_ADDR"] / os.environ["MASTER_PORT"]
    后再调用 dist.init_process_group，而不依赖 torchrun 注入的环境变量。

    调用后，distributed.py 中的全部全局进程组变量
    （_TP_GROUP、_DP_GROUP、_DRAFT_DP_GROUP、_DRAFT_SP_GROUP、
      _SP_ULYSSES_GROUP、_SP_RING_GROUP 等）均被正确设置，
    后续代码可直接复用 get_tp_group()、get_dp_group() 等工具函数。

    约束（与现有 init_distributed 相同）：
    - world_size == dp_size * tp_size
    - world_size % (sp_ulysses_size * sp_ring_size) == 0
    - draft_dp_size = world_size // (sp_ulysses_size * sp_ring_size)
    """

def get_free_port() -> int:
    """
    在 orchestrator 进程中调用，找到一个未被占用的 TCP 端口，
    用于 torch.distributed 的 rendezvous。
    """
```

---

### `specforge/ray_workers/__init__.py`

导出 `RolloutWorker`、`TrainWorker`。

---

### `specforge/ray_workers/worker_utils.py`（新增文件）

两类 worker 共享的轻量工具，**不引入 Ray 依赖**（便于单元测试）。

**需要实现：**

```python
@dataclass
class RolloutBatch:
    """
    RolloutWorker 产出、传递给 TrainWorker 的数据容器。
    字段均为 CPU tensor（通过 Ray object store 传递时避免 CUDA IPC 问题）。

    batch 大小为 target_batch_size = tp_size * per_dp_batch_size。
    每个 TrainWorker 收到同一份完整 RolloutBatch，
    在 train_step 内部按 tp_rank 切片得到属于自己的 sub-batch。
    """
    input_ids: torch.Tensor           # (target_batch_size, seq_len)
    attention_mask: torch.Tensor      # (target_batch_size, seq_len)
    loss_mask: torch.Tensor           # (target_batch_size, seq_len, 1)
    hidden_states: torch.Tensor       # (target_batch_size, seq_len, 3 * hidden_size)
    target: torch.Tensor              # (target_batch_size, seq_len, vocab_size)
    # VLM 可选字段
    pixel_values: Optional[torch.Tensor] = None
    image_grid_thw: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None


def batch_to_device(batch: RolloutBatch, device: torch.device) -> RolloutBatch:
    """将 RolloutBatch 中所有非 None tensor 移到指定 device。"""

def batch_shard_by_tp(batch: RolloutBatch, tp_size: int, tp_rank: int) -> RolloutBatch:
    """
    按 TP rank 对 batch 的第 0 维（batch 维）切片，等价于现有的 get_dp_data_shard_from_tp()。

    用途：RolloutBatch 包含 target_batch_size = tp_size * per_dp_batch_size 条样本。
    同一 TP group 内的 tp_size 个 worker 分别取自己对应的 per_dp_batch_size 条，
    互不重叠地完成 FSDP 前向计算。

    注意：SP（序列并行）不在此处切片，而是在 UspAdapter 内部由 yunchang 库
    按序列维度自动分发。
    """
```

---

### `specforge/ray_workers/rollout_worker.py`（新增文件）

封装目标模型推理逻辑的 Ray Actor。

**核心设计：**

```python
@ray.remote(num_gpus=1)
class RolloutWorker:
    """
    负责运行目标模型（SGLang/HF/Custom backend）进行前向推理，
    产出 Eagle3 所需的 hidden states 和 logits。

    一个 RolloutWorker 占用一块 GPU。
    若 rollout_tp_size > 1，则 rollout_tp_size 个 RolloutWorker 构成一个 TP group，
    共同推理同一个目标模型分片。
    """
```

**需要实现的方法：**

```python
def __init__(
    self,
    rank: int,
    world_size: int,
    tp_size: int,
    tp_rank: int,
    master_addr: str,
    master_port: int,
    target_model_path: str,
    backend: str,                      # "sglang" | "hf" | "custom"
    sglang_backend_args: dict,         # SGLangBackendArgs.to_kwargs() 的结果
    draft_model_config: dict,          # 序列化后的 AutoDraftModelConfig
    torch_dtype: str = "bfloat16",
    trust_remote_code: bool = False,
    model_download_dir: str = None,
    is_vlm: bool = False,
    timeout_minutes: int = 20,
) -> None:
    """
    初始化步骤：
    1. 调用 distributed_ray.init_rollout_distributed() 初始化 rollout 侧 TP group。
    2. 调用 get_eagle3_target_model() 加载目标模型。
    3. 调用 target_model.set_aux_hidden_states_layers() 设置 aux hidden states 层。
    注意：__init__ 在 Ray Actor 内异步执行，调用方需用 ray.get(worker.is_ready.remote()) 确认初始化完成。
    """

def is_ready(self) -> bool:
    """健康检查，返回 True 表示模型加载完成。"""

def generate_rollout_batch(
    self,
    input_ids_ref,          # ray.ObjectRef 或直接 bytes，指向 (batch, seq_len) CPU tensor
    attention_mask_ref,
    loss_mask_ref,
    pixel_values_ref=None,
    image_grid_thw_ref=None,
) -> "ray.ObjectRef":
    """
    执行一次前向推理：
    1. 从 ray.ObjectRef（或直接 tensor）中取出输入数据并移到 GPU。
    2. 调用 target_model.generate_eagle3_data()。
    3. 将结果（Eagle3TargetOutput）的所有 tensor 移到 CPU，
       封装为 RolloutBatch，放入 Ray object store，返回 ObjectRef。

    在 colocated 模式下，此方法由 orchestrator 在 train step 之前同步调用。
    在 disaggregated + async 模式下，此方法被提前异步触发。
    """

def get_model_config(self) -> dict:
    """返回目标模型的 hf_config（序列化为 dict），供 orchestrator 传递给 TrainWorker。"""

def shutdown(self) -> None:
    """释放 GPU 资源，销毁 process group。"""
```

**TP 协调：**

当 `rollout_tp_size > 1` 时，`RolloutWorkerGroup`（在 `resource_manager.py` 中实现）负责创建 `rollout_tp_size` 个 `RolloutWorker` Actor，并将它们的 `rank`/`tp_rank` 正确赋值，以及共享同一 `master_addr:master_port`。只有 `tp_rank == 0` 的 worker 需要实际返回 `RolloutBatch`（其他 rank 返回 `None`），由 orchestrator 只取 `tp_rank=0` 的结果。

---

### `specforge/ray_workers/train_worker.py`（新增文件）

封装 draft 模型训练逻辑的 Ray Actor，支持 **DP + SP + TP** 三维并行。

#### 并行拓扑说明

`TrainWorker` 的并行方式完全对齐现有 `distributed.py` 中的 `init_distributed()`：

```
train_num_gpus = world_size = dp_size * tp_size

目标模型 device mesh（用于 rollout 侧 TP，不在 train worker 内）：
  (dp_size, tp_size)

draft 模型 device mesh（SP 将现有 world 内的 rank 重新分组）：
  draft_dp_size = world_size // (sp_ulysses_size * sp_ring_size)
  (draft_dp_size, sp_ulysses_size * sp_ring_size)
```

各进程组的作用：

| 进程组 | 作用 |
|--------|------|
| `_TP_GROUP` | draft 模型 TP（通常 tp_size=1，即无 TP；保留以兼容多 TP 扩展） |
| `_DP_GROUP` | FSDP sharding group，FSDP `process_group=dist.group.WORLD` 在 world 内 shard |
| `_DRAFT_DP_GROUP` | USP 模式下的 draft 数据并行组（DataLoader DistributedSampler 使用） |
| `_DRAFT_SP_GROUP` | USP 序列并行组（`sp_ulysses_size * sp_ring_size` 个 rank 共享同一条样本的不同序列分段） |
| `_SP_ULYSSES_GROUP` | yunchang Ulysses SP 内部使用 |
| `_SP_RING_GROUP` | yunchang Ring SP 内部使用 |

**rank 分配公式**（与 `init_device_mesh` 行为一致）：

```
# 目标模型 mesh (dp_size, tp_size)：
tp_rank  = rank % tp_size
dp_rank  = rank // tp_size

# draft mesh (draft_dp_size, sp_size)，sp_size = sp_ulysses * sp_ring：
sp_rank       = rank % sp_size
draft_dp_rank = rank // sp_size
```

**数据流向**（单个训练 step，batch_size per DP rank = B）：

```
DataLoader（按 draft_dp_group 采样，每个 draft_dp_rank 不同样本）
    每个 draft_dp_rank 取 B 条样本，通过 train_step 传入 RolloutBatch
    ↓
batch_shard_by_tp(batch, tp_size, tp_rank)   # 按 tp_rank 切 batch 维，得到 B//tp_size 条
    ↓ (SP 开启时)
UspAdapter.step_view()                       # 在 sequence 维按 sp_rank 切片（yunchang 内部）
    ↓
FSDP forward / backward（world 级别 all-reduce gradient）
    ↓
BF16Optimizer.step()
```

**约束（SP 模式）：**
- `attention_backend` 必须为 `"usp"`
- `batch_size == 1`（per dp rank）
- `draft_accumulation_steps` 自动乘以 `sp_size`（与现有 `sp_sanity_check` 逻辑相同）

**核心设计：**

```python
@ray.remote(num_gpus=1)
class TrainWorker:
    """
    负责维护 draft 模型（FSDP 分片）、执行前向/反向传播与参数更新。
    支持 DP + Ulysses/Ring SP + TP 三维并行。

    一个 TrainWorker 占用一块 GPU。
    多个 TrainWorker 通过 torch.distributed（由 orchestrator 建立的 process group）协同。
    """
```

**需要实现的方法：**

```python
def __init__(
    self,
    rank: int,
    world_size: int,          # = dp_size * tp_size（train 侧总 GPU 数）
    tp_size: int,             # draft 模型 TP size（通常为 1）
    sp_ulysses_size: int,     # Ulysses SP size（>1 时需 attention_backend="usp"）
    sp_ring_size: int,        # Ring SP size
    master_addr: str,
    master_port: int,
    # draft model 参数
    draft_model_config_path: str,
    target_model_path: str,
    attention_backend: str,   # "sdpa" | "fa" | "flex_attention" | "usp"
    embedding_key: str,
    ckpt_dir: str,
    output_dir: str,
    # 训练超参
    learning_rate: float,
    max_grad_norm: float,
    warmup_ratio: float,
    total_steps: int,
    ttt_length: int,
    draft_accumulation_steps: int,
    torch_dtype: str = "bfloat16",
    trust_remote_code: bool = False,
    model_download_dir: str = None,
    timeout_minutes: int = 20,
) -> None:
    """
    初始化步骤：
    1. 调用 distributed_ray.init_train_distributed(rank, world_size, master_addr,
       master_port, tp_size, sp_ulysses_size, sp_ring_size) 初始化训练侧所有进程组。
       执行完成后 distributed.py 中的全部全局变量（_TP_GROUP、_DP_GROUP、
       _DRAFT_DP_GROUP、_DRAFT_SP_GROUP、_SP_ULYSSES_GROUP、_SP_RING_GROUP）均就绪。
    2. 若 sp_ulysses_size * sp_ring_size > 1，校验 attention_backend == "usp"
       且 batch_size == 1（与现有 sp_sanity_check 逻辑一致）。
    3. 调用现有 build_draft_model() 逻辑加载 draft 模型（embed + freeze）。
    4. 构造 OnlineEagle3Model(draft_model=draft_model, length=ttt_length,
       attention_backend=attention_backend)，target_model=None。
    5. 用 FSDP 包裹 eagle3_model：
         FSDP(eagle3_model,
              use_orig_params=True,
              mixed_precision=MixedPrecision(param_dtype=bf16, buffer_dtype=bf16),
              sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
              process_group=dist.group.WORLD)
       与现有 train_eagle3.py 中的 FSDP 初始化完全一致。
    6. 初始化 BF16Optimizer(draft_model, lr, max_grad_norm, warmup_ratio, total_steps)。
    7. 若存在 resume checkpoint，加载 optimizer state。
    注意：TrainWorker 中不加载目标模型，target_model 字段为 None。
    """

def is_ready(self) -> bool:
    """健康检查。"""

def train_step(
    self,
    rollout_batch_ref,          # ray.ObjectRef 指向 RolloutBatch（CPU tensor）
    global_step: int,
) -> dict:
    """
    执行一个训练 step，数据流：
    1. ray.get(rollout_batch_ref) 取出 RolloutBatch（CPU）。
    2. batch_to_device(batch, cuda)。
    3. batch_shard_by_tp(batch, tp_size, tp_rank)：
         按 tp_rank 切 batch 的第 0 维，得到本 worker 负责的 sub-batch。
         （SP 切片由 UspAdapter.step_view 在 forward 内部完成，此处无需处理）
    4. 调用 eagle3_model.forward(
           input_ids=batch.input_ids,
           attention_mask=batch.attention_mask,
           loss_mask=batch.loss_mask,
           hidden_states=batch.hidden_states,   # 预计算好，跳过 target_model 调用
           target=batch.target,
           position_ids=batch.position_ids,
       )
    5. 调用 run_backward_and_update(plosses, optimizer, global_step)。
    6. 仅 rank==0 的 worker 返回有效 metrics dict（loss, acc, lr）；
       其余 rank 返回 None，由 TrainWorkerGroup 过滤。
    """

def eval_step(
    self,
    rollout_batch_ref,
) -> dict:
    """
    与 train_step 相同，但在 torch.no_grad() 下执行，不调用 backward 和 optimizer.step()。
    """

def save_checkpoint(
    self,
    epoch: int,
    step: int,
) -> str:
    """
    保存 checkpoint，逻辑复用现有 save_checkpoints()：
    - 使用 FSDP.state_dict_type(FULL_STATE_DICT) 收集完整权重。
    - 只有 rank==0 实际写磁盘（training_state.pt + draft model 权重）。
    - 所有 rank 调用 dist.barrier() 同步后返回。
    返回保存路径字符串（rank != 0 时返回空字符串）。
    """

def load_vocab_mapping(self, vocab_mapping_path: str) -> None:
    """加载 vocab mapping 文件到 draft_model（所有 rank 都需执行）。"""

def get_training_state(self) -> dict:
    """返回当前 epoch、global_step、lr 等训练状态，用于 orchestrator 维护进度。"""

def shutdown(self) -> None:
    """调用 destroy_distributed() 销毁所有 process group，释放 GPU 资源。"""
```

**关键约束（`OnlineEagle3Model.forward()` 的修改）：**

现有 `OnlineEagle3Model.forward()` 不直接调用 target_model，而是由 `train_eagle3.py` 的 `run_forward()` 在外部调用 `target_model.generate_eagle3_data()`，再将结果传入 forward。因此 TrainWorker 中的调用方式（直接传入 `hidden_states` 和 `target`）与现有 `is_online=True` 路径**完全兼容，无需修改** `OnlineEagle3Model.forward()`。

---

### `specforge/ray_trainer/resource_manager.py`（新增文件）

管理 Ray placement group 与 worker 的创建/销毁。

**需要实现的类和函数：**

```python
class RolloutWorkerGroup:
    """
    管理一组 RolloutWorker Actor，负责 TP 组的创建与通信协调。
    """
    def __init__(
        self,
        num_workers: int,           # == rollout_num_gpus
        tp_size: int,
        target_model_path: str,
        backend: str,
        sglang_backend_kwargs: dict,
        draft_model_config: dict,
        is_vlm: bool,
        **model_kwargs,
    ) -> None:
        """
        步骤：
        1. 通过 get_free_port() 为每个 TP group 分配 rendezvous 端口。
           （num_workers // tp_size 个 TP group，每组一个端口）
        2. 用 ray.remote 创建 num_workers 个 RolloutWorker Actor，
           通过 ray placement group 将同一 TP group 内的 worker 尽量放在同一 node。
        3. 等待所有 worker is_ready()。
        """

    def generate_rollout_batch(
        self,
        data_batch: dict,           # DataLoader 返回的一个 batch（CPU tensor dict）
    ) -> List["ray.ObjectRef"]:
        """
        将 data_batch 拆分并分发给各 TP group 的 tp_rank=0 worker，
        并发调用 worker.generate_rollout_batch.remote()。
        返回 ObjectRef 列表，每个 ObjectRef 对应一个 TP group 的输出。
        通常只有 1 个 TP group，len == 1。
        """

    def shutdown(self) -> None:
        """向所有 worker 发送 shutdown，销毁 placement group。"""


class TrainWorkerGroup:
    """
    管理一组 TrainWorker Actor，负责 DP+SP+TP group 的创建与同步。

    GPU 分配：
      num_workers = world_size = dp_size * tp_size
      其中 SP 不新增 GPU，而是在 world 内重新划分序列并行组。

    rank 分配（与 init_device_mesh 行为对齐）：
      对于 workers[rank]：
        tp_rank  = rank % tp_size
        dp_rank  = rank // tp_size
        sp_size  = sp_ulysses_size * sp_ring_size
        sp_rank       = rank % sp_size
        draft_dp_rank = rank // sp_size
    """
    def __init__(
        self,
        num_workers: int,           # = dp_size * tp_size
        tp_size: int,
        sp_ulysses_size: int,
        sp_ring_size: int,
        draft_model_args: dict,     # 包含 draft_model_config_path、target_model_path 等
        train_hparams: dict,        # 包含 lr、max_grad_norm、attention_backend 等超参
        output_dir: str,
        ckpt_dir: str,
    ) -> None:
        """
        步骤：
        1. 验证：num_workers % tp_size == 0，
                 num_workers % (sp_ulysses_size * sp_ring_size) == 0。
        2. 为训练侧 rendezvous 分配一个端口（所有 num_workers 个 worker 共用一个端口）。
        3. 创建 num_workers 个 TrainWorker Actor，每个 actor 传入对应的 rank（0..num_workers-1）。
        4. 等待所有 worker is_ready()。
        """

    def train_step(
        self,
        rollout_batch_refs: List["ray.ObjectRef"],
        global_step: int,
    ) -> dict:
        """
        向所有 TrainWorker 广播同一份 rollout_batch_ref，并发调用 train_step.remote()。
        等待全部完成，仅取 rank==0 的结果作为 metrics（loss, acc, lr）。

        注意：所有 worker 收到同一份 RolloutBatch（完整 target_batch_size 条样本），
        各 worker 在 train_step 内部用 batch_shard_by_tp 按 tp_rank 切片。
        SP 进一步在 UspAdapter 内按序列维切片（yunchang 库内部处理）。
        """

    def eval_step(self, rollout_batch_refs: List["ray.ObjectRef"]) -> dict:
        """并发调用所有 TrainWorker.eval_step.remote()，取 rank==0 的 eval metrics。"""

    def save_checkpoint(self, epoch: int, step: int) -> str:
        """并发调用，等待所有 rank dist.barrier() 完成后，取 rank-0 返回的路径。"""

    def load_vocab_mapping(self, vocab_mapping_path: str) -> None:
        """向所有 worker 并发发送加载指令，等待所有完成。"""

    def shutdown(self) -> None:
        """向所有 worker 发送 shutdown，等待完成后销毁 placement group。"""


def build_worker_groups(
    args,                           # argparse.Namespace，含 DisaggregateArgs 字段
    sglang_backend_args,
    draft_model_config,
) -> Tuple[RolloutWorkerGroup, TrainWorkerGroup]:
    """
    根据 args.disaggregate 决定 GPU 分配策略，创建并返回两个 worker group。

    合并模式（disaggregate=False）：
    - rollout_num_gpus = train_num_gpus = args.tp_size * (world_size // args.tp_size)
    - 两个 group 共享同一批物理 GPU（RolloutWorker 和 TrainWorker 各占 0.5 GPU）
    - 使用 ray placement group PACK 策略确保同 GPU 上配对

    分离模式（disaggregate=True）：
    - rollout_num_gpus 个 GPU 专用于 RolloutWorkerGroup
    - train_num_gpus 个 GPU 专用于 TrainWorkerGroup
    - 通过 Ray placement group bundle 的 SPREAD 策略分配到不同 GPU

    返回：(rollout_group, train_group)
    """
```

---

### `specforge/ray_trainer/pipeline.py`（新增文件）

定义单个训练 step 的完整执行逻辑，将 rollout 与 train 解耦。

**需要实现：**

```python
class TrainingPipeline:
    """
    控制一个全局 step 的执行流程。
    在合并模式和分离模式下行为相同，差异由 RolloutWorkerGroup 和 TrainWorkerGroup 内部屏蔽。
    """

    def __init__(
        self,
        rollout_group: RolloutWorkerGroup,
        train_group: TrainWorkerGroup,
        eval_dataloader,            # 可选，用于 eval step
        args,
    ) -> None: ...

    def run_train_step(
        self,
        data_batch: dict,
        global_step: int,
        tracker,
    ) -> dict:
        """
        完整的一个训练 step，流程：
        1. rollout_group.generate_rollout_batch(data_batch)
           → 返回 rollout_batch_refs（异步 ObjectRef 列表）
        2. train_group.train_step(rollout_batch_refs, global_step)
           → 阻塞等待训练完成，得到 metrics dict
        3. 若 global_step % log_interval == 0，通过 tracker 记录 metrics。
        4. 返回 metrics dict。
        """

    def run_eval(
        self,
        global_step: int,
        tracker,
    ) -> dict:
        """
        遍历 eval_dataloader，对每个 batch 调用：
        1. rollout_group.generate_rollout_batch(batch)
        2. train_group.eval_step(rollout_batch_refs)
        汇总所有 batch 的 metrics 后记录到 tracker。
        """

    def save_checkpoint(self, epoch: int, step: int) -> str:
        """委托给 train_group.save_checkpoint()。"""
```

---

### `specforge/ray_trainer/orchestrator.py`（新增文件）

顶层协调器，对应现有 `train_eagle3.py` 中的 `main()` 函数，但以 Ray 方式驱动。

**需要实现：**

```python
class Eagle3RayOrchestrator:
    """
    在 driver 进程（非 Ray Actor）中运行，负责：
    - 初始化 Ray cluster
    - 通过 resource_manager 创建 worker groups
    - 驱动 TrainingPipeline 完成所有 epoch 的训练循环
    - 管理 dataloader、tracker、checkpoint 逻辑

    不持有任何模型参数，仅协调控制流。
    """

    def __init__(self, args) -> None:
        """
        步骤：
        1. ray.init(address=args.ray_address, namespace=args.ray_namespace, ...)
        2. 构建 draft_model_config（复用现有逻辑）。
        3. 构建 train_dataloader 和 eval_dataloader（复用现有 build_dataloaders() 逻辑）。
        4. 调用 build_worker_groups() 创建 rollout_group 和 train_group。
        5. 调用 train_group.load_vocab_mapping() 加载 vocab mapping。
        6. 初始化 tracker。
        """

    def run(self) -> None:
        """
        主训练循环，逻辑与现有 main() 中 epoch/step 循环基本一致：

        for epoch in range(start_epoch, num_epochs):
            for step_in_epoch, data in enumerate(train_dataloader):
                # skip already-done steps for resume
                global_step += 1
                metrics = pipeline.run_train_step(data, global_step, tracker)
                if should_eval:
                    pipeline.run_eval(global_step, tracker)
                if should_save:
                    pipeline.save_checkpoint(epoch, global_step)
                if max_num_steps and global_step >= max_num_steps:
                    break

        # 保存最终 checkpoint
        # 关闭 tracker
        # shutdown worker groups
        # ray.shutdown()
        """

    def _compute_total_steps(self) -> int:
        """复用现有逻辑，计算 total_steps。"""

    def shutdown(self) -> None:
        """优雅关闭，依次 shutdown train_group、rollout_group，最后 ray.shutdown()。"""
```

---

### `scripts/train_eagle3_ray.py`（新增文件）

基于 Ray 的训练入口脚本，取代 `scripts/train_eagle3.py`（原脚本保持不变）。

**需要实现：**

```python
def parse_args():
    """
    在现有 train_eagle3.py 的 parse_args() 基础上，额外注册：
    - RayArgs.add_args(parser)
    - DisaggregateArgs.add_args(parser)
    并移除 torch.distributed 相关参数（如 --dist-timeout），
    因为 worker 内部由 orchestrator 管理。
    """

def main():
    """
    1. 解析参数。
    2. set_seed(args.seed)。
    3. 实例化 Eagle3RayOrchestrator(args)。
    4. orchestrator.run()。
    5. orchestrator.shutdown()。

    注意：此脚本用 python train_eagle3_ray.py ... 直接启动，
    不需要 torchrun 或 deepspeed launcher。
    Ray 自动处理多节点/多 GPU 调度。
    """

if __name__ == "__main__":
    main()
```

**启动示例：**

```bash
# 合并模式：8 GPU，rollout TP=4，train DP=2 TP=1 SP=1
python scripts/train_eagle3_ray.py \
    --target-model-path /path/to/llama3-70b \
    --train-data-path /path/to/data.jsonl \
    --output-dir /path/to/output \
    --rollout-tp-size 4 \
    --train-tp-size 1 \
    --train-sp-ulysses-size 1 \
    --batch-size 2

# 合并模式：4 GPU，train DP=2 TP=1 SP_ulysses=2（序列并行）
python scripts/train_eagle3_ray.py \
    --target-model-path /path/to/llama3-70b \
    --train-data-path /path/to/data.jsonl \
    --output-dir /path/to/output \
    --rollout-tp-size 1 \
    --train-tp-size 1 \
    --train-sp-ulysses-size 2 \
    --attention-backend usp \
    --batch-size 1

# 分离模式：4 GPU 推理（TP=4）+ 8 GPU 训练（DP=4 SP_ulysses=2）
python scripts/train_eagle3_ray.py \
    --target-model-path /path/to/llama3-70b \
    --train-data-path /path/to/data.jsonl \
    --output-dir /path/to/output \
    --disaggregate \
    --rollout-num-gpus 4 \
    --train-num-gpus 8 \
    --rollout-tp-size 4 \
    --train-tp-size 1 \
    --train-sp-ulysses-size 2 \
    --attention-backend usp \
    --batch-size 1
```

---

## 数据流

### 合并模式（colocated）

```
DataLoader (driver)
    │  data_batch (CPU)
    ▼
RolloutWorkerGroup.generate_rollout_batch()
    │  内部：放入 Ray ObjectRef
    │  [GPU 0~3: 目标模型 TP 推理]
    │  输出：RolloutBatch (CPU) → ObjectRef
    ▼
TrainWorkerGroup.train_step(rollout_batch_ref)
    │  内部：从 ObjectRef 取出 → .cuda()
    │  [GPU 0~3: draft 模型 FSDP 前向/反向]
    ▼
metrics dict (driver)
    │
    ▼
Tracker.log()
```

### 分离模式（disaggregated）

```
DataLoader (driver)
    │  data_batch (CPU)
    ▼
RolloutWorkerGroup.generate_rollout_batch()
    │  [GPU 0~3 (rollout 专用): 目标模型 TP 推理]
    │  输出：RolloutBatch (CPU) → Ray ObjectRef
    │       （经由 Ray object store 跨 node 传输，无需 NCCL）
    ▼
TrainWorkerGroup.train_step(rollout_batch_ref)
    │  [GPU 4~7 (train 专用): draft 模型 FSDP 前向/反向]
    ▼
metrics dict (driver)
```

---

## 训练侧 DP+SP+TP 并行拓扑

本节通过具体数值示例说明三维并行的 rank 分配，与 `distributed.py` 的 `init_distributed()` 完全一致。

### 示例一：DP=4, TP=1, SP=1（纯数据并行，4 GPU）

```
world_size = 4, dp_size = 4, tp_size = 1, sp = 1
draft_dp_size = 4

rank 0: dp=0, tp=0, draft_dp=0, sp=0   → TrainWorker[0]
rank 1: dp=1, tp=0, draft_dp=1, sp=0   → TrainWorker[1]
rank 2: dp=2, tp=0, draft_dp=2, sp=0   → TrainWorker[2]
rank 3: dp=3, tp=0, draft_dp=3, sp=0   → TrainWorker[3]

FSDP group: [0,1,2,3]（world 级别）
DataLoader sampler: get_dp_group() = [0,1,2,3]
RolloutBatch size: target_batch_size = tp_size * batch_size = 4
  → 每个 worker 通过 batch_shard_by_tp 取 batch[rank:rank+1]（1 条）
```

### 示例二：DP=2, TP=2, SP=1（数据+张量并行，4 GPU）

```
world_size = 4, dp_size = 2, tp_size = 2, sp = 1
draft_dp_size = 4

rank 0: dp=0, tp=0, draft_dp=0, sp=0
rank 1: dp=0, tp=1, draft_dp=1, sp=0
rank 2: dp=1, tp=0, draft_dp=2, sp=0
rank 3: dp=1, tp=1, draft_dp=3, sp=0

TP group[dp=0]: [rank 0, rank 1]
TP group[dp=1]: [rank 2, rank 3]
FSDP group: [0,1,2,3]（world 级别）
DataLoader sampler: get_dp_group() = [0,2]（每个 TP group 的代表 rank）
RolloutBatch size: target_batch_size = tp_size * batch_size = 2 * B
  → rank 0 和 rank 1 用 batch_shard_by_tp 各取一半 batch（B 条）
  → rank 2 和 rank 3 同理
```

### 示例三：DP=2, TP=1, SP_ulysses=2（数据+序列并行，4 GPU）

```
world_size = 4, dp_size = 4, tp_size = 1
sp_ulysses_size = 2, sp_ring_size = 1, sp_size = 2
draft_dp_size = 4 // 2 = 2

rank 0: dp=0, tp=0, draft_dp=0, sp=0（ulysses=0）
rank 1: dp=1, tp=0, draft_dp=0, sp=1（ulysses=1）
rank 2: dp=2, tp=0, draft_dp=1, sp=0（ulysses=0）
rank 3: dp=3, tp=0, draft_dp=1, sp=1（ulysses=1）

SP(Ulysses) group[draft_dp=0]: [rank 0, rank 1]（同一样本，序列分两段）
SP(Ulysses) group[draft_dp=1]: [rank 2, rank 3]
FSDP group: [0,1,2,3]
DataLoader sampler: get_draft_dp_group() = [0,2]（每个 draft_dp 组的代表）
attention_backend: 必须为 "usp"，batch_size = 1（per draft_dp rank）
RolloutBatch size: target_batch_size = tp_size * batch_size = 1 * 1 = 1
  → 每个 draft_dp group（2 个 worker）共享同一条样本，序列维在 UspAdapter 内切分
```

### 示例四：DP=1, TP=2, SP_ulysses=2（全三维并行，4 GPU）

```
world_size = 4, dp_size = 2, tp_size = 2
sp_ulysses_size = 2, sp_ring_size = 1, sp_size = 2
draft_dp_size = 4 // 2 = 2

rank 0: dp=0, tp=0, draft_dp=0, sp=0
rank 1: dp=0, tp=1, draft_dp=0, sp=1
rank 2: dp=1, tp=0, draft_dp=1, sp=0
rank 3: dp=1, tp=1, draft_dp=1, sp=1

TP group[dp=0]:    [rank 0, rank 1]
TP group[dp=1]:    [rank 2, rank 3]
SP group[draft_dp=0]: [rank 0, rank 1]（与 TP group 重合，SP 和 TP 共享相同 rank 集合）
SP group[draft_dp=1]: [rank 2, rank 3]
FSDP group: [0,1,2,3]
attention_backend: "usp"，batch_size = 1
RolloutBatch size: target_batch_size = tp_size * 1 = 2
  → rank 0/rank 1 通过 batch_shard_by_tp 各取 1 条，再由 UspAdapter 切序列维
```

---

## 进程组布局

### 合并模式（8 GPU：rollout TP=4，train DP=2 TP=2 SP=2）

```
GPU 0: RolloutWorker(tp_rank=0) + TrainWorker(rank=0, dp=0, tp=0, sp=0)
GPU 1: RolloutWorker(tp_rank=1) + TrainWorker(rank=1, dp=0, tp=1, sp=1)
GPU 2: RolloutWorker(tp_rank=2) + TrainWorker(rank=2, dp=1, tp=0, sp=0)
GPU 3: RolloutWorker(tp_rank=3) + TrainWorker(rank=3, dp=1, tp=1, sp=1)
GPU 4: RolloutWorker(tp_rank=0) + TrainWorker(rank=4, dp=2, tp=0, sp=0)
GPU 5: RolloutWorker(tp_rank=1) + TrainWorker(rank=5, dp=2, tp=1, sp=1)
GPU 6: RolloutWorker(tp_rank=2) + TrainWorker(rank=6, dp=3, tp=0, sp=0)
GPU 7: RolloutWorker(tp_rank=3) + TrainWorker(rank=7, dp=3, tp=1, sp=1)

rollout TP group 0: [GPU 0,1,2,3]
rollout TP group 1: [GPU 4,5,6,7]
train FSDP group:   [GPU 0..7]（world 级别）
train SP group 0:   [GPU 0,1]（draft_dp=0，sp=2）
train SP group 1:   [GPU 2,3]（draft_dp=1，sp=2）
...
```

### 分离模式（4 GPU rollout + 8 GPU train，train: DP=4, TP=1, SP_ulysses=2）

```
GPU 0~3:   RolloutWorker[0..3]（rollout TP group: [GPU 0,1,2,3]）
GPU 4~11:  TrainWorker[0..7]

TrainWorker rank layout（tp=1, sp=2）：
  rank 0: draft_dp=0, sp=0  → GPU 4
  rank 1: draft_dp=0, sp=1  → GPU 5
  rank 2: draft_dp=1, sp=0  → GPU 6
  rank 3: draft_dp=1, sp=1  → GPU 7
  rank 4: draft_dp=2, sp=0  → GPU 8
  rank 5: draft_dp=2, sp=1  → GPU 9
  rank 6: draft_dp=3, sp=0  → GPU 10
  rank 7: draft_dp=3, sp=1  → GPU 11

train FSDP group:   [GPU 4..11]
train SP group[0]:  [GPU 4,5]（draft_dp=0）
train SP group[1]:  [GPU 6,7]（draft_dp=1）
...

两侧各自独立的 torch.distributed process group，互不干扰。
数据通过 Ray object store 传递（CPU tensor，无需 NCCL 跨组通信）。
```

---

## 与现有代码的兼容性

| 现有组件 | 改动 |
|----------|------|
| `specforge/distributed.py` | **不改动**，`distributed_ray.py` 内通过设置环境变量后调用其 `init_distributed()` 逻辑，复用全部全局变量与工具函数 |
| `specforge/core/eagle3.py`（`OnlineEagle3Model`） | **不改动**，现有 `forward()` 签名已接受外部传入的 `hidden_states` 和 `target`，无需额外修改 |
| `specforge/core/eagle3_adapters.py`（`UspAdapter`） | **不改动**，SP 切片逻辑由 yunchang 库在 `UspAdapter.step_view()` 内自动完成 |
| `specforge/modeling/target/` | **不改动** |
| `specforge/optimizer.py` | **不改动** |
| `scripts/train_eagle3.py` | **不改动**，保持原有 torchrun 方式可用 |
| `specforge/args.py` | **追加** `RayArgs`、`DisaggregateArgs`，不修改现有类 |

---

## 依赖

新增依赖（需加入 `pyproject.toml` / `requirements.txt`）：

```
ray[default]>=2.10.0
```

`ray[default]` 包含 Dashboard、GCS 等组件，可监控 Actor 状态。若只需核心功能，可用 `ray>=2.10.0`。

---

## 实现优先级建议

**第一阶段（核心功能，验证 Ray 管理合并模式正确性）**

1. `distributed_ray.py`：`init_train_distributed`、`get_free_port`
2. `ray_workers/worker_utils.py`：`RolloutBatch`、`batch_to_device`、`batch_shard_by_tp`
3. `ray_workers/train_worker.py`：完整实现
4. `ray_trainer/resource_manager.py`：仅 `TrainWorkerGroup`
5. `ray_trainer/orchestrator.py`：合并模式，RolloutWorker 由 TrainWorker 内部调用（与现有等价）
6. `scripts/train_eagle3_ray.py`：入口脚本

目标：使用 Ray 管理的 `TrainWorker` 跑通合并模式训练，loss 曲线与原 torchrun 脚本一致。

**第二阶段（训推分离）**

1. `distributed_ray.py`：`init_rollout_distributed`
2. `ray_workers/rollout_worker.py`：完整实现
3. `ray_trainer/resource_manager.py`：`RolloutWorkerGroup`、`build_worker_groups`
4. `ray_trainer/pipeline.py`：`TrainingPipeline`
5. `specforge/args.py`：`RayArgs`、`DisaggregateArgs`

目标：`--disaggregate` 模式下，rollout 与 train 运行在不同 GPU 上，loss 曲线与合并模式一致。

**第三阶段（异步 pipeline，可选）**

在 `pipeline.py` 中，当 `rollout_async=True` 时，对下一个 step 的 `generate_rollout_batch` 在当前 step train 期间提前发起（流水线并行），隐藏 rollout 延迟。需在 `TrainingPipeline` 内维护一个 prefetch buffer。
