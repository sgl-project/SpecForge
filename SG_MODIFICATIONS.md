# SpecForge 适配 Qwen3.5-4B 修改记录

## 修改概述

为了使 SpecForge 框架能够训练 Qwen3.5-4B EAGLE3 模型，需要进行以下修改：

---

## 1. transformers/utils/generic.py

**文件路径**: `/home/pairshoe/anaconda3/envs/jimoke/lib/python3.11/site-packages/transformers/utils/generic.py`

**问题**: 新版 transformers 移除了 `check_model_inputs` 函数，但 SpecForge 仍然引用它。

**修改**: 在文件末尾添加缺失的函数

```python
def check_model_inputs(fn):
    """
    Decorator that checks model inputs. This was removed in newer transformers versions
    but is still referenced by some code. This is a minimal implementation for compatibility.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper
```

---

## 2. sglang/srt/configs/qwen3_5.py

**文件路径**: `/home/pairshoe/anaconda3/envs/jimoke/lib/python3.11/site-packages/sglang/srt/configs/qwen3_5.py`

**问题**: Qwen3.5-4B 配置中的 `vision_config` 包含 `deepstack_visual_indexes` 参数，但 `Qwen3_5VisionConfig` 类没有显式定义该参数。

**修改**: 为 `Qwen3_5VisionConfig` 添加完整的 `__init__` 方法

```python
class Qwen3_5VisionConfig(Qwen3VLVisionConfig):
    model_type = "qwen3_5"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=24,
        hidden_size=1024,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=4096,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=2560,
        num_position_embeddings=2304,
        deepstack_visual_indexes=None,  # 关键修改：接受该参数
        initializer_range=0.02,
        **kwargs,
    ):
        # Qwen3.5-4B uses empty array for deepstack_visual_indexes
        if deepstack_visual_indexes is None:
            deepstack_visual_indexes = []
        super().__init__(
            depth=depth,
            hidden_size=hidden_size,
            hidden_act=hidden_act,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            in_channels=in_channels,
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            temporal_patch_size=temporal_patch_size,
            out_hidden_size=out_hidden_size,
            num_position_embeddings=num_position_embeddings,
            deepstack_visual_indexes=deepstack_visual_indexes,
            initializer_range=initializer_range,
            **kwargs,
        )
```

---

## 3. specforge/modeling/target/eagle3_target_model.py

**文件路径**: `/home/pairshoe/ljl/train_eagle3/SpecForge/specforge/modeling/target/eagle3_target_model.py`

**问题**: sglang 加载的 Qwen3.5 模型实例没有 `set_eagle3_layers_to_capture` 方法，需要动态补丁。

**修改 1**: 在文件开头导入补丁函数

```python
# Import Qwen3.5 EAGLE3 patch
try:
    from specforge.modeling.qwen3_5_eagle_patch import patch_qwen3_5_instance
except ImportError:
    patch_qwen3_5_instance = None
```

**修改 2**: 修改 `SGLangEagle3TargetModel.set_aux_hidden_states_layers` 方法

```python
def set_aux_hidden_states_layers(
    self, aux_hidden_states_layers: Optional[List[int]] = None
) -> None:
    # Apply Qwen3.5 EAGLE3 patch if the model doesn't have the method yet
    model = self.model_runner.model
    if not hasattr(model, "set_eagle3_layers_to_capture"):
        if patch_qwen3_5_instance is not None:
            patch_qwen3_5_instance(model)
        else:
            raise AttributeError(
                f"Model {type(model).__name__} does not have set_eagle3_layers_to_capture method. "
                "Please ensure the EAGLE3 patch is properly applied."
            )
    self.model_runner.model.set_eagle3_layers_to_capture(aux_hidden_states_layers)
```

---

## 4. specforge/modeling/draft/llama3_eagle.py

**文件路径**: `/home/pairshoe/ljl/train_eagle3/SpecForge/specforge/modeling/draft/llama3_eagle.py`

**问题**: Qwen3.5-4B 的 RoPE scaling 类型是 "default"，但代码只支持 "linear", "dynamic", "yarn", "longrope", "llama3", "mrope"。

**修改**: 在 `_init_rope` 方法中添加 "default" 类型的处理

```python
# Handle "default" as no scaling - use standard rotary embedding
if scaling_type is None or scaling_type == "default":
    self.rotary_emb = LlamaRotaryEmbedding(
        self.head_dim,
        max_position_embeddings=self.max_position_embeddings,
        base=getattr(self.config, "rope_theta", 10000),
    )
elif scaling_type == "linear":
    # ... 原有代码
```

---

## 新增文件

### 1. specforge/modeling/qwen3_5_eagle_patch.py

**文件路径**: `/home/pairshoe/ljl/train_eagle3/SpecForge/specforge/modeling/qwen3_5_eagle_patch.py`

**作用**: 为 Qwen3.5 模型添加 EAGLE3 训练所需的补丁，包括：
- `patch_qwen3_5_for_eagle3()`: 类级别补丁
- `patch_qwen3_5_instance()`: 实例级别补丁
- `_qwen3_5_forward_with_eagle3()`: 修改后的 forward 方法，支持捕获 auxiliary hidden states

---

## 训练脚本参数说明

### Qwen3.5-4B 特定参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `--embedding-key` | `model.language_model.embed_tokens.weight` | Qwen3.5 使用嵌套的 embedding 键 |
| `--lm-head-key` | `model.language_model.embed_tokens.weight` | Qwen3.5 使用 tied embeddings，lm_head 共享 embedding 权重 |
| `--chat-template` | `qwen` | 使用 Qwen 的对话模板 |
| `--attention-backend` | `sdpa` | 使用 sdpa 而非 flex_attention（后者有内存问题） |

### 推荐训练参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--batch-size` | 1 | 显存限制，建议使用 1 |
| `--max-length` | 1024 | 更长的序列可能导致 OOM |
| `--learning-rate` | 5e-5 | 标准学习率 |
| `--num-epochs` | 1-3 | 根据数据量调整 |
| `--attention-backend` | sdpa | flex_attention 在某些 GPU 上有 Triton 内存问题 |

---

## 已知问题与解决方案

### 1. flex_attention Triton 内存问题

**错误**: `OutOfMemoryError: out of resource: triton_tem_fused_0 Required: 151552 Hardware limit:101376`

**解决方案**: 使用 `--attention-backend sdpa` 替代默认的 flex_attention

### 2. lm_head.weight 不存在

**错误**: `KeyError: 'lm_head.weight'`

**原因**: Qwen3.5-4B 使用 tied embeddings (`tie_word_embeddings=True`)

**解决方案**: 设置 `--lm-head-key "model.language_model.embed_tokens.weight"`

### 3. 词表映射问题

**说明**: Qwen3.5-4B 词表大小为 248320，draft model 使用压缩词表 32000

**解决方案**: SpecForge 会自动生成词表映射文件，保存位置在 `cache/vocab_mapping/`

---

## 目录结构

```
SpecForge/
├── configs/
│   └── qwen3.5-4b-eagle3.json          # Draft model 配置
├── scripts/
│   ├── prepare_hidden_states.py        # 生成 hidden states
│   └── train_eagle3.py                 # 训练脚本
├── specforge/
│   ├── modeling/
│   │   ├── qwen3_5_eagle_patch.py     # Qwen3.5 EAGLE3 补丁 [新增]
│   │   ├── draft/
│   │   │   └── llama3_eagle.py        # [修改: 添加 default RoPE 支持]
│   │   └── target/
│   │       └── eagle3_target_model.py # [修改: 自动补丁实例]
│   └── ...
├── cache/
│   ├── dataset/                        # 训练数据
│   ├── hidden_states/                  # 生成的 hidden states
│   ├── processed_dataset/              # 缓存的预处理数据
│   └── vocab_mapping/                  # 词表映射缓存
├── outputs/                            # 训练输出
├── train_qwen35_4b_full.sh            # 完整训练脚本 [新增]
└── train_qwen35_4b_multi_gpu.sh       # 多卡训练脚本 [新增]
```

---

## 快速开始

1. **生成 hidden states** (只需一次):
```bash
bash train_qwen35_4b_full.sh
# 脚本会自动完成两步训练
```

2. **使用已生成的 hidden states 直接训练**:
```bash
CUDA_VISIBLE_DEVICES=6 torchrun \
    --standalone \
    --nproc_per_node 1 \
    scripts/train_eagle3.py \
    --target-model-path /path/to/Qwen3.5-4B \
    --draft-model-config configs/qwen3.5-4b-eagle3.json \
    --train-data-path cache/dataset/ultrachat_train_fixed.jsonl \
    --train-hidden-states-path cache/hidden_states/qwen35_4b_ultrachat \
    --output-dir outputs/qwen3.5-4b-eagle3 \
    --embedding-key "model.language_model.embed_tokens.weight" \
    --lm-head-key "model.language_model.embed_tokens.weight" \
    --attention-backend sdpa \
    --batch-size 1 \
    --max-length 1024 \
    --num-epochs 1 \
    --learning-rate 5e-5
```

---

## 注意事项

1. **显存需求**: Qwen3.5-4B + sglang 后端 + EAGLE3 训练约需 16-20GB 显存
2. **数据缓存**: 第一次运行会自动缓存预处理数据，后续运行会直接使用缓存
3. **词表映射**: 自动从数据集中统计词频并生成映射，支持 draft_vocab_size=32000
4. **multi-gpu**: 如需多卡训练，设置 `NUM_GPUS` > 1，并相应调整 batch size

---

## 多卡训练配置

SpecForge 支持两种并行策略：

### 1. 数据并行 (Data Parallel, DP)

所有 GPU 处理不同的数据，模型完整复制到每块 GPU。

**配置**:
```bash
NUM_GPUS=2        # 总 GPU 数量
TP_SIZE=1         # Tensor Parallelism = 1（关闭）
GPU_IDS="0,1"
```

**特点**:
- 训练更快（通信开销小）
- 需要更多显存（每块 GPU 存完整模型）
- 推荐用于 1-4 卡场景

**有效 Batch Size** = `NUM_GPUS × BATCH_SIZE`

### 2. Tensor 并行 (Tensor Parallel, TP)

模型被分片到多块 GPU，每块 GPU 只存模型的一部分。

**配置**:
```bash
NUM_GPUS=2        # 总 GPU 数量
TP_SIZE=2         # Tensor Parallelism = NUM_GPUS
GPU_IDS="0,1"
```

**特点**:
- 显存占用更少（模型分片）
- 通信开销较大
- 推荐用于大模型或显存不足场景

**有效 Batch Size** = `BATCH_SIZE`（与单卡相同）

### 3. 混合并行 (TP + DP)

结合 TP 和 DP，适合 4+ 卡场景。

**配置**:
```bash
NUM_GPUS=4        # 总 GPU 数量
TP_SIZE=2         # 每 2 卡一组进行 TP
GPU_IDS="0,1,2,3"
```

**结果**:
- DP Size = 4 / 2 = 2（2 组数据并行）
- TP Size = 2（每组内 2 卡 tensor 并行）
- 有效 Batch Size = `2 × BATCH_SIZE`

### 参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--nproc_per_node` | torchrun 启动的进程数 = `NUM_GPUS` | 等于 GPU 数 |
| `--tp-size` | Tensor Parallel 大小 | 1 (DP 模式) |
| `--batch-size` | 单卡 batch size | 1-2 |

### 使用示例

**单卡训练**:
```bash
bash train_qwen35_4b_full.sh
```

**双卡 DP 训练**（推荐）:
```bash
# 修改 train_qwen35_4b_multi_gpu.sh:
NUM_GPUS=2
TP_SIZE=1
GPU_IDS="0,1"

bash train_qwen35_4b_multi_gpu.sh
```

**四卡混合训练**:
```bash
# 修改 train_qwen35_4b_multi_gpu.sh:
NUM_GPUS=4
TP_SIZE=2
GPU_IDS="0,1,2,3"

bash train_qwen35_4b_multi_gpu.sh
```

### 显存参考

| 配置 | 每卡显存占用 | 总显存 |
|------|-------------|--------|
| 1卡 DP (TP=1) | ~18GB | 18GB |
| 2卡 DP (TP=1) | ~18GB | 36GB |
| 2卡 TP (TP=2) | ~10GB | 20GB |
| 4卡混合 (TP=2) | ~10GB | 40GB |
