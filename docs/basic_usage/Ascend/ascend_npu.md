# Ascend NPU Tutorial

This is an end-to-end tutorial for running SpecForge on Ascend NPU hosts. It
walks through **installation → data preparation → online disaggregated training
with external services → the managed-local full stack → split multi-node
roles**, using Qwen3.5-4B DFlash as the running example. Validated on a 16-card
A3 (64GB) host.

---

## 1. Installation

You need an Ascend host with the driver, CANN, and a `torch_npu`-enabled
PyTorch already installed, plus SGLang `0.5.14` with NPU support. Then install
SpecForge without touching that stack:

```bash
git clone https://github.com/sgl-project/SpecForge.git
cd SpecForge
python -m pip install -e . --no-deps
```

`--no-deps` keeps pip from pulling CUDA wheels over the working NPU
torch/sglang. If a later step reports a missing lightweight dependency, install
just that package, also with `--no-deps`.

### Apply the SGLang capture patches (online runs only)

Online capture needs two patches on top of the installed SGLang, applied **in
this order**:

```bash
# Base capture patch: --enable-spec-capture server flags + the capture sink
bash scripts/apply_sglang_spec_capture_patch.sh

# Ascend companion patch: skip the wildcard segment mount that Ascend
# Mooncake rejects, and mount the feature segment with location="cpu"
SGLANG_DIR=$(python -c "import sglang, os; print(os.path.dirname(os.path.dirname(sglang.__file__)))")
cd "$SGLANG_DIR" && git apply /path/to/SpecForge/patches/sglang/v0.5.14/spec-capture-ascend-mount.patch
```

Skip both for offline training, which reads features from disk. The companion
patch is a no-op on non-Ascend hosts (it keys off `ASCEND_RT_VISIBLE_DEVICES`),
so a shared installation stays CUDA-safe.

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
a patched SGLang capture server, features stream through Mooncake, and a
consumer trains the draft. The checked-in
[`qwen3.5-4b-dflash-online-npu.yaml`](../../../examples/configs/online/disaggregated/external/qwen3.5-4b-dflash-online-npu.yaml)
recipe targets an externally started capture server.

### Step 1: Start Mooncake and the capture server

```bash
mooncake_master --enable_http_metadata_server=true \
  --rpc_port=35551 --http_metadata_server_port=35880 \
  --metrics_port=35903 --enable_metric_reporting=false &
```

Start the patched capture server on NPU 0. The `--spec-capture-aux-layer-ids`
must match the draft's `target_layer_ids` — for DFlash these come from
`configs/qwen3.5-4b-dflash.json`: `1 8 15 22 29`. A mismatch produces zero
features with no error:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
MOONCAKE_LOCAL_HOSTNAME=127.0.0.1 \
MOONCAKE_METADATA_SERVER=http://127.0.0.1:35880/metadata \
MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:35551 \
MOONCAKE_PROTOCOL=tcp \
MOONCAKE_GLOBAL_SEGMENT_SIZE=$((32<<30)) \
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-4B \
  --trust-remote-code \
  --skip-tokenizer-init \
  --tp-size 1 \
  --mem-fraction-static 0.8 \
  --attention-backend ascend \
  --enable-spec-capture --spec-capture-method dflash \
  --spec-capture-aux-layer-ids 1 8 15 22 29 \
  --host 127.0.0.1 --port 30000 &
```

Wait for `curl --fail http://127.0.0.1:30000/health` to return 200.

### Step 2: Launch training

On the remaining NPUs:

```bash
ASCEND_RT_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 \
specforge train -c examples/configs/online/disaggregated/external/qwen3.5-4b-dflash-online-npu.yaml
```

Before rerunning, clear stale control state:
`rm -rf outputs/qwen3.5-4b-dflash-npu-online`.

### Success criteria

- Producer log ends with `prompts_failed=0`.
- Consumer prints `step N: {...loss..., acc...}` lines and no
  `could not drain` error at teardown.
- If the capture producer dies with `ACL_ERROR_RT_CONTEXT_NULL` (107002), the
  installed SpecForge predates the NPU transport bind — upgrade past #722.

---

## 4. Managed-local full stack (one command)

Instead of starting Mooncake and capture servers by hand, the
[`qwen3.5-4b-dflash-disaggregated-npu.yaml`](../../../examples/configs/online/disaggregated/managed-local/qwen3.5-4b-dflash-disaggregated-npu.yaml)
recipe lets a single `specforge train` command own the whole single-node
stack — Mooncake, capture server(s), and the trainer — and derives their
endpoints and device assignments:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HCCL_CONNECT_TIMEOUT=7200 HCCL_EXEC_TIMEOUT=7200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

specforge train -c examples/configs/online/disaggregated/managed-local/qwen3.5-4b-dflash-disaggregated-npu.yaml
```

The checked-in layout parks the capture server on device 0 and runs a 14-rank
trainer on devices 2-15 (8-card hosts: trainer on devices 1-7 with
`deployment.trainer.nproc_per_node=7`).

### Scaling capture throughput

One capture server is enough for correctness but bounds the pipeline: the
trainer's data wait then dominates step time. To give capture more cards,
override the layout inline — e.g. a 6:10 split with six TP=1 capture servers:

```bash
specforge train -c examples/configs/online/disaggregated/managed-local/qwen3.5-4b-dflash-disaggregated-npu.yaml \
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
  zero-copy clients), and the capture side needs the companion patch from
  Section 1.
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
DFlash** draft (`configs/qwen3.5-4b-dflash.json`) with 6 TP=1 capture servers
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
