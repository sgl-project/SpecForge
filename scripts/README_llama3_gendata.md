# Llama3 数据生成脚本说明

## 问题描述

在运行 `llama3_gendata_patch.py` 脚本时，可能会遇到以下错误：

```
OverflowError: There was an overflow with type <class 'list'>. Try to reduce writer_batch_size to have batches smaller than 2GB.
(offset overflow while concatenating arrays, consider casting input from `list<item: list<item: float>>` to `list<item: large_list<item: float>>` first.)
```

## 问题原因

这个错误是由于数据量过大导致的 PyArrow 溢出问题：

1. **隐藏状态数据量大**：每条记录包含多个层的隐藏状态，数据量很大
2. **批次大小过大**：默认的 `group_size=400` 导致单个批次数据超过2GB
3. **PyArrow限制**：PyArrow在处理大型列表时遇到了溢出问题

## 解决方案

### 1. 减小批次大小

使用 `--group_size` 参数控制每个chunk的记录数量：

```bash
# 使用较小的批次大小（推荐）
python scripts/llama3_gendata_patch.py --group_size 20 --start 0 --end 100

# 如果仍然出错，可以进一步减小
python scripts/llama3_gendata_patch.py --group_size 10 --start 0 --end 100
```

### 2. 内存管理优化

脚本已经添加了以下优化：

- **GPU内存清理**：每次处理完一条记录后清理GPU内存
- **错误处理**：添加try-catch块处理异常
- **动态批次调整**：如果保存失败，自动尝试更小的writer_batch_size

### 3. 硬件建议

- **GPU内存**：建议至少16GB GPU内存
- **系统内存**：建议至少32GB系统内存
- **存储空间**：确保有足够的磁盘空间存储生成的chunk文件

## 使用示例

```bash
# 基本使用
python scripts/llama3_gendata_patch.py \
    --start 0 \
    --end 100 \
    --group_size 20 \
    --outdir /path/to/output

# 使用多个GPU
python scripts/llama3_gendata_patch.py \
    --start 0 \
    --end 100 \
    --group_size 20 \
    --gpu_index 0 1 \
    --outdir /path/to/output

# 处理不同数据集
python scripts/llama3_gendata_patch.py \
    --dataset ultrachat \
    --start 0 \
    --end 100 \
    --group_size 20
```

## 输出文件

脚本会在指定的输出目录中生成多个chunk文件：
```
outdir/
├── chunk_0/
├── chunk_1/
├── chunk_2/
└── ...
```

每个chunk包含指定数量的记录，包含以下字段：
- `input_ids`: 输入token序列
- `loss_mask`: 损失掩码
- `hidden_state`: 三个层的隐藏状态（低、中、高层）
- `target_hidden_states`: 最后一层的隐藏状态

## 故障排除

如果仍然遇到问题：

1. **进一步减小group_size**：尝试 `--group_size 5` 或更小
2. **检查GPU内存**：使用 `nvidia-smi` 监控GPU内存使用
3. **检查磁盘空间**：确保有足够的存储空间
4. **分批处理**：将大的数据集分成多个小批次处理 