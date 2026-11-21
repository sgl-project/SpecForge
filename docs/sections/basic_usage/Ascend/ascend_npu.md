# Ascend NPU Tutorial

This is an end-to-end tutorial for running SpecForge on Ascend NPU hosts. It
walks through **installation → data preparation → online disaggregated training
with external services → the managed-local full stack → split multi-node
roles**, using Qwen3.5-4B DFlash as the running example. Validated on a 16-card
A3 (64GB) host.

---

## 1. Installation

You need an Ascend host with the driver, CANN, and a `torch_npu`-enabled
PyTorch already installed, plus an NPU-compatible SGLang `main` build with
the capture extension APIs described below. Then install
SpecForge without touching that stack:

```bash
git clone https://github.com/sgl-project/SpecForge.git
cd SpecForge
python -m pip install -e . --no-deps
```

`--no-deps` keeps pip from pulling CUDA wheels over the working NPU
torch/sglang. If a later step reports a missing lightweight dependency, install
just that package, also with `--no-deps`.

`liger-kernel` is not needed: on NPU, DFlash drafts keep the native
RMSNorm/SwiGLU (`model.use_liger_kernel` resolves to off). Liger's Ascend
backend is not validated in SpecForge; an explicit `model.use_liger_kernel:
true` tries it and logs a warning.

### Install the SGLang capture plugin (online and offline preparation)

Use an Ascend-compatible SGLang build with the generic extension APIs required
by [the plugin guide](https://github.com/sgl-project/SpecForge/blob/main/plugins/sglang-spec-capture/README.md):

```bash
python -m pip install --no-deps plugins/sglang-spec-capture
export SPECFORGE_SPEC_CAPTURE=1
```

Release 0.5.23 lacks the required capture APIs. The plugin handles Ascend's CPU
segment mount directly. Offline training from
existing feature files needs no capture service.

### Device visibility on Ascend

Ascend selects devices through `ASCEND_RT_VISIBLE_DEVICES`, and the driver
rejects an *empty* value — hiding devices from a process means **unsetting**
the variable, not setting it to `""`. SpecForge handles this internally: device
ordinals from the config are injected through `CUDA_VISIBLE_DEVICES` on CUDA
hosts and `ASCEND_RT_VISIBLE_DEVICES` on Ascend hosts, and a "hide all devices"
role unsets the variable. You only need to export the visible set once for the
supervisor:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

Two more environment settings are recommended for long online runs:

```bash
export HCCL_CONNECT_TIMEOUT=7200 HCCL_EXEC_TIMEOUT=7200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

### Attention backends on Ascend

Use `sdpa` for the **trainer**. Sequence parallelism (`sp_ulysses_size` /
`sp_ring_size` > 1, the `usp` backend) requires `yunchang`, whose import probes
the CUDA device and crashes NPU-only torch builds, so USP is CUDA-only for now.
SpecForge imports `yunchang` lazily only when SP sizes exceed 1, so the default
SP=1 path never touches it.

For the **capture server**, the recipes set `model.sglang_attention_backend:
ascend`. If a config leaves it at the flashinfer default, the launcher falls
back to `ascend` automatically on Ascend hosts (flashinfer does not exist
there).

---

## 2. Data preparation

Data preparation is platform independent. From the repository root:

```bash
python scripts/prepare_data.py --dataset sharegpt
```

This writes `./cache/dataset/sharegpt_train.jsonl`. For custom datasets and
target-model regeneration, see the [Data Preparation](../data_preparation.md)
guide.

---

## 3. Online training (external capture server)

Online training is always **disaggregated**: a producer drives prompts through
a SGLang capture server, features stream through Mooncake, and a
consumer trains the draft. The checked-in
[`qwen3.5-4b-dflash-npu.yaml`](https://github.com/sgl-project/SpecForge/blob/main/examples/configs/online/disaggregated/external/qwen3.5-4b-dflash-npu.yaml)
recipe targets an externally started capture server.

### Step 1: Start Mooncake and the capture server

```bash
mooncake_master --enable_http_metadata_server=true \
  --rpc_port=35551 --http_metadata_server_port=35880 \
  --metrics_port=35903 --enable_metric_reporting=false &
```

Start the plugin-enabled capture server on NPU 0. The `--return-aux-hidden-state-layer-ids`
must match the draft's `target_layer_ids` — for DFlash these come from
`specforge/config/draft/presets/qwen3.5-4b-dflash.yaml`: `1 8 15 22 29`. A mismatch produces zero
features with no error:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
MOONCAKE_LOCAL_HOSTNAME=127.0.0.1 \
MOONCAKE_METADATA_SERVER=http://127.0.0.1:35880/metadata \
MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:35551 \
MOONCAKE_PROTOCOL=tcp \
MOONCAKE_GLOBAL_SEGMENT_SIZE=$((32<<30)) \
SPECFORGE_SPEC_CAPTURE=1 \
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-4B \
  --trust-remote-code \
  --skip-tokenizer-init \
  --tp-size 1 \
  --mem-fraction-static 0.8 \
  --attention-backend ascend \
  --enable-return-aux-hidden-states --return-hidden-states-mode full --speculative-algorithm DFLASH \
  --return-aux-hidden-state-layer-ids 1 8 15 22 29 \
  --host 127.0.0.1 --port 30000 &
```

Wait for `curl --fail http://127.0.0.1:30000/health` to return 200.

### Step 2: Launch training

On the remaining NPUs:

```bash
ASCEND_RT_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 \
specforge train -c examples/configs/online/disaggregated/external/qwen3.5-4b-dflash-npu.yaml
```

Before rerunning, clear stale control state:
`rm -rf outputs/qwen3.5-4b-dflash-npu-online`.

### Success criteria

- Producer log ends with `prompts_failed=0`.
- Consumer prints `step N: {...loss..., acc...}` lines and no
  `could not drain` error at teardown.
- If the capture producer dies with `ACL_ERROR_RT_CONTEXT_NULL` (107002), the
  installed SpecForge predates the NPU transport bind — upgrade past #722.

### Qwen3-30B-A3B-Instruct-2507 with EAGLE3.1 on 910B3

The [Qwen EAGLE3.1 recipe](https://github.com/sgl-project/SpecForge/blob/5d4c6a1483afec88b19270d369d16b99492abfdf/examples/configs/online/disaggregated/external/qwen3-30b-a3b-eagle3.1.yaml)
ran online on three 64 GiB Ascend 910B3 devices. Two devices served the BF16
target with SGLang TP2 and the third trained the draft with SDPA. The target
was `Qwen/Qwen3-30B-A3B-Instruct-2507` at
`0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`.

The run used driver 26.0.rc1, CANN/HCCL 9.1.0, Python 3.12.13,
PyTorch 2.10.0+cpu with torch_npu 2.10.0.post6, Transformers 5.19.0,
SGLang NPU kernels 2026.9.0, and Mooncake 0.3.14 over TCP.
The capture server and inference benchmark loaded SGLang source at
`c757c4bb7c55d1eb69fdb2bdc6e6fb8507acb7a0`. The
[reproduction bundle](https://github.com/LUOseu/SpecForge/tree/ascend-910b3-eagle3-smoke-20261121/validation/ascend-eagle31-910b3)
includes the Ascend image digest, installation commands, pinned ShareGPT
preparation, vocabulary map generation, and the three-way evaluation command.

Start the external Mooncake master as in step 1. In a separate shell, start
the target with the auxiliary layer IDs from the EAGLE3.1 draft preset.
Set `TARGET_MODEL` to the local model snapshot. Device lists for capture and
training use the chip logic IDs reported by `npu-smi info -m`.

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 SPECFORGE_SPEC_CAPTURE=1 \
MOONCAKE_LOCAL_HOSTNAME=127.0.0.1 MOONCAKE_PROTOCOL=tcp \
MOONCAKE_METADATA_SERVER=http://127.0.0.1:35880/metadata \
MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:35551 \
MOONCAKE_GLOBAL_SEGMENT_SIZE=1073741824 SGLANG_SPEC_CAPTURE_GPU_PUT=0 \
python -m sglang.launch_server --model-path "$TARGET_MODEL" \
  --device npu --dtype bfloat16 --tp-size 2 --attention-backend ascend \
  --skip-tokenizer-init --context-length 1024 --max-running-requests 1 \
  --mem-fraction-static 0.8 --chunked-prefill-size -1 --disable-radix-cache \
  --disable-cuda-graph --enable-return-aux-hidden-states \
  --return-hidden-states-mode full --speculative-algorithm EAGLE3 \
  --return-aux-hidden-state-layer-ids 1 23 44 --host 127.0.0.1 --port 30000
```

From the repository root, set `DATA` to the prepared directory containing
`train.jsonl` and `vocab.pt`, and `RUN_BASE` to a fresh output directory.
These eight-update runs use the same pretokenized prompts, vocabulary map,
seed 0, batch size 1, learning rate 0.0001, and rollout depth 7.

```bash
export ASCEND_RT_VISIBLE_DEVICES=2 MOONCAKE_LOCAL_HOSTNAME=127.0.0.1
curl --fail http://127.0.0.1:30000/health
for method in eagle3.1 eagle3; do
  run="qwen30b-910b3-$method"
  out="$RUN_BASE/$run"
  specforge train \
    -c "examples/configs/online/disaggregated/external/qwen3-30b-a3b-$method.yaml" \
    "model.target_model_path=$TARGET_MODEL" "model.vocab_mapping_path=$DATA/vocab.pt" \
    'data.train_data_path=' "data.prompts_path=$DATA/train.jsonl" data.max_length=512 \
    data.build_dataset_num_proc=1 data.dataloader_num_workers=0 \
    training.attention_backend=sdpa deployment.trainer.nproc_per_node=1 \
    training.num_epochs=1 training.warmup_ratio=0 training.max_steps=8 \
    training.save_interval=8 training.log_interval=1 tracking.report_to=none \
    "run_id=$run" "output_dir=$out" "deployment.disaggregated.control_dir=$out/control" \
    "deployment.disaggregated.consumer_state_dir=$out/consumer-state" || exit 1
  specforge export --converter sglang --target-model "$TARGET_MODEL" \
    --checkpoint "$out/$run-step8" --output-dir "$out/sglang-export" || exit 1
done
```

Each run captured all eight ShareGPT examples. Each token yielded 6,144
auxiliary features and a 2,048-wide final hidden state. EAGLE3.1 applies a
separate RMSNorm to each layer's features before projection. The final
normalized draft state feeds the language-model head and the next draft
step. The EAGLE3.1 export preserved all 18 tensors and sets `fc_norm` and
`norm_output` to true. Use `--speculative-algorithm EAGLE3` for both
serving exports. A separate eight-token SGLang request verified that both TP
workers loaded all four norm weights, applied the three feature norms, and
returned the final normalized tensor for speculative feedback.

The serving comparison used the same two 910B3 devices, BF16 target, eight
held-out ShareGPT prompts and concurrency 1. Each fresh server handled a
warmup followed by eight greedy requests of 128 output tokens each. The
drafts used three speculative steps, top-k 1 and four draft tokens. Mean
acceptance length counts completion tokens per verification round, including
the bonus token. Throughput counts completion tokens per timed wall second.

| Serving mode | Output tokens | Verification rounds | Mean acceptance length | Output tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Target only | 1,024 | — | — | 46.19 |
| EAGLE3, step 8 | 1,024 | 971 | 1.0546 | 30.35 |
| EAGLE3.1, step 8 | 1,024 | 964 | 1.0622 | 39.39 |

The two draft runs produced identical token sequences on all eight prompts.
Each matched the target-only output on four of eight prompts. Both drafts
had lower throughput than the target-only run. This single timing pass and
eight-update training budget measure the integration smoke test.
[Stepwise training losses](https://github.com/LUOseu/SpecForge/blob/ascend-910b3-eagle3-smoke-20261121/validation/ascend-eagle31-910b3/results/train-loss.png)
and the [downloadable serving checkpoints](https://github.com/LUOseu/SpecForge/releases/tag/ascend-910b3-eagle3-smoke-20261121)
refer to these same two runs.

### Training DFlash 2 drafts

DFlash 2 reuses the DFlash strategy and capture path end to end, so the
external-server setup above carries over unchanged — including the
`--return-aux-hidden-state-layer-ids 1 8 15 22 29` flags, which match
`specforge/config/draft/presets/qwen3.5-4b-dflash2.yaml`. Use the checked-in
[`qwen3.5-4b-dflash2-npu.yaml`](https://github.com/sgl-project/SpecForge/blob/main/examples/configs/online/disaggregated/external/qwen3.5-4b-dflash2-npu.yaml)
recipe:

```bash
ASCEND_RT_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 \
specforge train -c examples/configs/online/disaggregated/external/qwen3.5-4b-dflash2-npu.yaml
```

Notes specific to DFlash 2 on Ascend:

- **`attention_backend: sdpa` is mandatory** (already set in the recipe). The
  schema default `flex_attention` is rejected at the first training forward
  with `flex_attention is not available on this device; use sdpa/eager`.
- **`sdpa` and `eager` share the explicit boolean-mask path**; both implement
  the same block-local draft visibility, and the grouped convolutions operate
  inside each proposal block independently of the attention mask. The checked-in
  4B draft uses `full_attention` layers only, so no sliding-window mask is
  involved.
- **Selector knobs keep their defaults** (`loss.terms.selector.coefficient: 1.0`,
  zero warmup/ramp). The selector and convolutions are stock torch ops and need
  no NPU-specific settings.
- **Serving the export** still requires an SGLang build with DFlash 2 support
  (SGLang PR #35371); training-side capture is unchanged.

---

## 4. Managed-local full stack (one command)

Instead of starting Mooncake and capture servers by hand, the
[`qwen3.5-4b-dflash-npu.yaml`](https://github.com/sgl-project/SpecForge/blob/main/examples/configs/online/disaggregated/managed-local/qwen3.5-4b-dflash-npu.yaml)
recipe lets a single `specforge train` command own the whole single-node
stack — Mooncake, capture server(s), and the trainer — and derives their
endpoints and device assignments:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HCCL_CONNECT_TIMEOUT=7200 HCCL_EXEC_TIMEOUT=7200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

specforge train -c examples/configs/online/disaggregated/managed-local/qwen3.5-4b-dflash-npu.yaml
```

The checked-in layout parks the capture server on device 0 and runs a 14-rank
trainer on devices 2-15 (8-card hosts: trainer on devices 1-7 with
`deployment.trainer.nproc_per_node=7`).

### Scaling capture throughput

One capture server is enough for correctness but bounds the pipeline: the
trainer's data wait then dominates step time. To give capture more cards,
override the layout inline — e.g. a 6:10 split with six TP=1 capture servers:

```bash
specforge train -c examples/configs/online/disaggregated/managed-local/qwen3.5-4b-dflash-npu.yaml \
  'deployment.trainer.nproc_per_node=10' \
  'deployment.disaggregated.managed_local.trainer_cuda_visible_devices=["6","7","8","9","10","11","12","13","14","15"]' \
  'deployment.disaggregated.managed_local.capture_servers=[{port: 40000, cuda_visible_devices: ["0"], tp_size: 1}, {port: 40001, cuda_visible_devices: ["1"], tp_size: 1}, {port: 40002, cuda_visible_devices: ["2"], tp_size: 1}, {port: 40003, cuda_visible_devices: ["3"], tp_size: 1}, {port: 40004, cuda_visible_devices: ["4"], tp_size: 1}, {port: 40005, cuda_visible_devices: ["5"], tp_size: 1}]'
```

In the validated 16-card run this cut step time from ~20s to ~4.5s.

### Cleanup between runs

Managed children exit with the supervisor, but after an interrupted run check
for strays before relaunching:

```bash
pkill -9 -f specforge; pkill -9 -f mooncake_master; pkill -9 -f torch.distributed.run
rm -rf outputs/qwen3.5-4b-dflash-npu-managed
```

---

## 5. Split producer and consumer roles across nodes

The same configs split across nodes with an explicit `--role`:

```bash
specforge train -c run.yaml --role producer   # inference / capture pool
specforge train -c run.yaml --role consumer --node-rank 0   # trainer-0
specforge train -c run.yaml --role consumer --node-rank 1   # trainer-1
```

Every capture server must use the same target model, capture method, and aux
layer ids. For external-service prerequisites, freshness rules, and resume, see
the [Disaggregated training](../disaggregated_training.md) guide.

---

## 6. NPU notes and troubleshooting

- **Empty `ASCEND_RT_VISIBLE_DEVICES` is invalid** — the Ascend driver rejects
  it. SpecForge unsets the variable instead of emptying it; do not export
  `ASCEND_RT_VISIBLE_DEVICES=` by hand.
- **USP is CUDA-only for now** — `yunchang` probes CUDA at import. Keep
  `sp_ulysses_size` / `sp_ring_size` at 1 and use the `sdpa` trainer backend.
- **Ascend Mooncake rejects wildcard buffer registration** — the trainer side
  forces `local_buffer_size=0` automatically (SpecForge roles are pure
  zero-copy clients), and the capture plugin mounts its feature segment
  with `location="cpu"`.
- **`ACL_ERROR_RT_CONTEXT_NULL` in the capture producer** — the Mooncake
  transfer engine needs a bound device context; SpecForge binds the local NPU
  before `setup()`. Seeing this error means the installation predates #722.
- **Teardown drain** — the lifecycle drain window (~20s) covers Mooncake's
  read-lease TTL, and managed-local masters start with
  `default_kv_lease_ttl_ms=500`. A `could not drain pending removals` error
  after an otherwise successful run means one of these is missing — upgrade
  SpecForge.

---

## Reference results on 16x A3 (64GB)

End-to-end online run on the managed-local stack, training a **Qwen3.5-4B
DFlash** draft (`specforge/config/draft/presets/qwen3.5-4b-dflash.yaml`) with 6 TP=1 capture servers
(devices 0-5) and 10 trainer ranks (devices 6-15): ~9.5k prompts x 10 epochs,
global batch 80/step, ~1.18k optimizer steps, ~1.5h total.

- Loss falls from ~6.0 to ~2.0 over the run with cosine LR (6e-4 -> 0); no
  NaN, no stall.
- ~4.5s/optimizer step with six capture servers (~20s/step with one).
- Memory: ~29GB per trainer rank, ~31-33GB per capture server; no OOM with
  `num_anchors: 512` (lower it on smaller cards).
- Clean terminal drain on shutdown.

![Qwen3.5-4B DFlash online training on 16x A3](https://github.com/user-attachments/assets/1b94b455-cb61-4374-a548-9a219a37ed64)

These numbers are a functional reference for the NPU managed-local stack, not
a tuned performance benchmark.
