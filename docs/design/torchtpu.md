# SpecForge on TorchTPU

gpt-oss-120B DSpark（[configs/gpt-oss-120B-dspark.json](../../configs/gpt-oss-120B-dspark.json)）在 TPU 上 online 训练：
target 抓取在 sglang-torchtpu，draft 训练在 torch_tpu。所有 TPU 代码在 `specforge_tpu/`，核心代码不出现 tpu 分支。

## 1. 结构

```
 server pod (v7x-8)                                   trainer pod (v7x-8)
 ┌──────────────────────────────────────┐            ┌──────────────────────────────────────┐
 │ mooncake_master                      │            │ specforge train (supervisor)         │
 │ sglang-torchtpu  TP=8                │  /generate │  ├─ producer  (CPU, 不碰芯片)          │
 │  └─ plugin specforge_tpu.sglang_capture◄──────────┤  │    refs.jsonl                      │
 │      aux[1,9,17,25,33] + post-norm   │            │  └─ consumer torchrun ×8, tpu_dist   │
 │      last hidden ──► Mooncake (tcp) ─┼───────────►│       DDP + sdpa, S 按 bucket pad    │
 └──────────────────────────────────────┘            └──────────────────────────────────────┘
```

插件关系（has-a，全部是运行时 patch）：

```
specforge.cli ── load_plugins() ──► entry point "specforge.plugins"  ──► specforge_tpu.register
sglang        ── load_plugins() ──► entry point "sglang.srt.plugins" ──► specforge_tpu.sglang_capture.register
```

## 2. Trainer patch（specforge_tpu/patches.py, shapes.py）

| 目标 | 原因 |
|---|---|
| `_DISTRIBUTED_BACKENDS["tpu"]`、`_bind_local_device`、`get_local_device` | 每个 rank 只看到一个 core（tpu:0） |
| plugin 加载时设 `TORCH_TPU_TOPOLOGY` | `set_seed` 在 `init_distributed` 之前就会初始化 PjRt |
| `_trainer_command` | torchrun 各 rank 共享父进程选的 slicebuilder 端口 |
| `build_launch_plan` | producer 子进程设 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`：`torch.manual_seed` 会独占全部芯片 |
| `_padded_collator`、`_cpu_max_valid_anchors` | S pad 到 `SPECFORGE_TPU_SEQ_BUCKETS`，anchor 宽度只依赖 bucket；超出最大 bucket 直接报错 |
| `SGLangServerCaptureAdapter.__init__` | wire codec：spec 以 JSON 放进 `sampling_params.custom_params` |

## 3. Capture 插件（specforge_tpu/sglang_capture）

对应 GPU 的 `patches/sglang/v0.5.18/spec-capture.patch`，由 `SPECFORGE_SPEC_CAPTURE_LAYER_IDS` 开启：

| Hook | 作用 |
|---|---|
| `configure_aux_hidden_state_capture` (AFTER) | 无 draft worker 时开启 DFlash 层抓取 |
| `LogitsProcessor._get_hidden_states_to_store` (AROUND) | 把 post-norm last hidden 拼到 packed aux 后（最后一层的边界抓取是 norm 之前的值） |
| `_append_prefill_hidden_states` (AROUND) | TP rank 0 按请求切片写 Mooncake；累积所有 chunk，支持 chunked prefill |
| `_GenerationStreamAccumulator.accept` (AFTER) | 结果放进 `meta_info["spec_capture"]`（`customized_info` 按输出 token 切片，max_new_tokens=0 时会被丢掉） |

为什么不加 `GenerateReqInput.spec_capture` 字段：dataclass 字段无法在运行时安全扩展，所以复用已有的 `custom_params` 透传。

## 4. 运行

- 镜像：`docker/tpu.Dockerfile`（sglang-torchtpu + `mooncake-transfer-engine-non-cuda` + SpecForge）。
- 数据：gpt-oss 模板要求 harmony 角色，先用 TPU server（`--reasoning-parser gpt-oss`）跑 `scripts/regenerate_train_data.py --is-gpt-oss --reasoning save`。
- 配方：`examples/configs/online/disaggregated/external/gpt-oss-120b-dspark-disaggregated-tpu.yaml`。

验证：`lm_head(last_hidden[-1])` 的 argmax 等于 server greedy token；offline/online 8 rank 训练稳定约 0.6 s/step。

## 5. 未做

多 host trainer、FSDP2、torch.compile 整图、capture sink 异步写（目前在 scheduler 线程同步 `batch_put_from`）。
