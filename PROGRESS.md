# Qwen3.5 EAGLE3 训练进度记录

## 日期：2026-03-23

## 目标
为 Qwen3.5-4B 模型实现 EAGLE3 speculative decoding 训练支持

## 主要修改

### 1. Monkey Patch 文件
**文件**: `specforge/modeling/qwen3_5_eagle_patch.py`

#### 主要功能
- 为 Qwen3.5 模型添加 EAGLE3 训练所需的 `aux_hidden_states` 捕获功能
- 同时修补 `Qwen3_5ForCausalLM` 和 `Qwen3_5ForConditionalGeneration` (VLM wrapper)

#### 修改内容

**新增函数**:
1. `_qwen3_5_forward_with_eagle3()` - 修补后的语言模型 forward 方法
   - 在指定层捕获 hidden states
   - 返回 `(hidden_states, aux_hidden_states)` 元组

2. `_patched_general_mm_embed_routine()` - 修补后的 VLM embedding 函数
   - 处理语言模型返回的 tuple
   - 将 aux_hidden_states 存储在语言模型中

3. `_patched_vlm_forward()` - 修补后的 VLM wrapper forward 方法
   - 从语言模型获取 aux_hidden_states
   - 将其传递给 logits processor

**补丁类方法**:
- `set_eagle3_layers_to_capture()` - 设置需要捕获 hidden states 的层
  - 支持 VLM wrapper 和直接语言模型
  - 自动处理 `+1` 的 sglang 层索引偏移

### 2. 训练脚本
**文件**: `run_qwen35_4b_6gpu.sh`

#### 参数调整
- `--sglang-mem-fraction-static`: 0.5 → 0.42 (内存平衡)
- `--max-length`: 2048 → 1024 (减少内存使用)
- 新增环境变量: `TORCH_COMPILE_DISABLE=1` (禁用 torch compile)

## 解决的问题

### 问题 1: AttributeError: 'Qwen3_5ForConditionalGeneration' object has no attribute 'set_eagle3_layers_to_capture'
**原因**: VLM wrapper 类没有 EAGLE3 方法
**解决**: 通过 monkey patch 添加方法

### 问题 2: AttributeError: 'Qwen3_5ForConditionalGeneration' object has no attribute 'embed_tokens'
**原因**: forward 补丁应用到 VLM wrapper，但 `embed_tokens` 在内部语言模型
**解决**: 分别修补语言模型和 VLM wrapper 的 forward 方法

### 问题 3: AssertionError: hidden_states.size(-1) != hidden_size * 3
**原因**: aux_hidden_states 没有被正确捕获和传递
**解决**: 实现完整的 hidden states 捕获和传递链路

### 问题 4: 导入错误
**原因**: `get_pp_group` 导入路径不正确
**解决**: 修正为 `from sglang.srt.distributed.parallel_state import get_pp_group`

## 当前状态

### 已完成
- ✅ Monkey patch 实现完成
- ✅ `aux_hidden_states` 成功捕获 (3 层，每层 shape: (1951, 2560))
- ✅ 补丁在所有 GPU 进程中正确应用
- ✅ 训练脚本成功启动

### 待解决
- ⚠️ 内存平衡问题：
  - `mem-fraction-static` < 0.4: 模型初始化失败
  - `mem-fraction-static` >= 0.5: Triton 编译器内存不足
  - 需要找到合适的平衡点或使用其他内存优化策略

## 下一步计划

1. 解决内存平衡问题：
   - 尝试使用梯度累积来减少 batch size
   - 探索使用离线训练模式
   - 考虑使用更小的模型或减少层数

2. 验证训练稳定性：
   - 确认 aux_hidden_states 正确传递到 draft model
   - 验证梯度计算正确
   - 测试模型保存和加载

## 文件清单

### 新增文件
- `specforge/modeling/qwen3_5_eagle_patch.py` - Monkey patch 实现

### 修改文件
- `scripts/train_eagle3.py` - 添加 patch 导入
- `run_qwen35_4b_6gpu.sh` - 参数调整

### 配置文件
- `configs/qwen3.5-4b-eagle3.json` - Draft model 配置
- `cache/dataset/ultrachat_train_fixed.jsonl` - 训练数据 (207,865 样本)
