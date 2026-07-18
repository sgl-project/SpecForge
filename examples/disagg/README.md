# Disaggregated examples

Two flavors live here:

- **Online DFlash via server-capture** (`run_disagg_dflash.py` +
  `run_qwen3.6_27b_dflash_disagg.sh`) — the real zero-copy split: a live patched
  SGLang server captures features and writes them straight into Mooncake; the
  trainer consumes them by key. See [Online DFlash](#online-dflash-via-server-capture-real-zero-copy).
- **Offline EAGLE3** (`run_disagg_eagle3.py`) — precomputed features ingested
  into a shared store, documented below.

# Disaggregated offline EAGLE3 example

Runs the offline EAGLE3 training of `scripts/train_eagle3_dataflow.py`, but splits
it across **two pools that share only a filesystem mount** — the M6 disaggregation
seam (`SharedDirFeatureStore`). It is the runnable proof that *disaggregation
changes where features live, not their values*: the training curve matches the
colocated offline run.

## How it works

```
 producer pool (node 0)                shared mount                 training pool (node 1)
 ─────────────────────                ──────────────              ──────────────────────
 ingest_offline_features()  ──put()──▶ SharedDirFeatureStore ──get()──▶ FeatureDataLoader
 write_ref_manifest()       ──json──▶  refs.json (no tensors) ──read──▶ build_disagg_eagle3_runtime
                                                                         TrainerController.fit()
```

The control plane carries only tensor-free `SampleRef` metadata (the manifest);
feature tensors travel through the shared store. `build_disagg_eagle3_runtime`
reuses the exact offline trainer assembly, so results align by construction.

## Backends

The feature transport is selected by `DISAGG_BACKEND` (default `shared_dir`):

| backend | store | shared *data* mount? |
|---|---|---|
| `shared_dir` (default) | `SharedDirFeatureStore` (`torch.save` on a POSIX mount) | required |
| `mooncake` | `MooncakeFeatureStore` (RDMA/TCP network object store) | not needed |

`mooncake` is the M6 **fast path**: producer `put()`s and consumer `get()`s by key
across nodes peer-to-peer, so feature tensors need no shared *data* mount (only the
small ref manifest still uses `DISAGG_MANIFEST`). Each object is hard-pinned so
Mooncake's cache LRU never drops a committed feature. Because a Mooncake object
lives in the **producer's** memory segment, the producer must stay alive until the
consumer finishes — the example holds it open until the consumer writes
`<manifest>.consumed` (or `DISAGG_PRODUCER_HOLD_S` elapses). Enable with:

```bash
export DISAGG_BACKEND=mooncake
export MOONCAKE_LOCAL_HOSTNAME=<this-node-ip>
export MOONCAKE_METADATA_SERVER=<metadata url>
export MOONCAKE_MASTER_SERVER_ADDR=<master host:port>
export MOONCAKE_PROTOCOL=tcp   # or rdma
```

Requires the `mooncake` package and a running Mooncake master/metadata service
(verify on a Mooncake-enabled GPU host). The contract itself is unit-tested
backend-agnostically in `tests/test_runtime/test_mooncake_store.py`.

## Run it (rcli, 2 nodes)

1. Generate offline features on node 0 (any EAGLE3 feature generator), e.g. into
   `/root/disagg/features` as `*.ckpt` with keys
   `input_ids,loss_mask,hidden_state,aux_hidden_state`.
2. Drive both pools at once — node 0 ingests, node 1 trains:

   ```bash
   rcli exec --per-node <job> 'bash examples/disagg/run_qwen2.5_7b_eagle3_disagg.sh'
   ```

The wrapper branches on `RCLI_NODE_RANK`. Override paths/steps via env
(`DISAGG_STORE_ROOT`, `FEATURES_DIR`, `MAX_STEPS`, `NPROC`, …). Both pools must
share `DISAGG_STORE_ROOT`/`DISAGG_STORE_ID` and (if set) `DISAGG_AUTH_TOKEN`
(B9 auth).

## Single-host smoke

`DISAGG_ROLE` overrides the rank-derived role, so you can run both halves on one
host sharing a local dir — run the producer once, then the consumer:

```bash
DISAGG_ROLE=producer  python examples/disagg/run_disagg_eagle3.py <args>
DISAGG_ROLE=consumer  torchrun --standalone --nproc_per_node 1 \
    examples/disagg/run_disagg_eagle3.py <args>
```

The bit-exact equivalence to the colocated path is covered by
`tests/test_runtime/test_disagg_launch.py`.

## Head-to-head vs colocated (Qwen2.5-7B, 2-node H200)

`DISAGG_ROLE=colocated` runs the same model build + assembly through
`build_offline_eagle3_runtime` (`LocalFeatureStore`). On identical features/seed,
the disaggregated consumer and the colocated baseline produce the same training
metrics to ~5 significant figures (residual ~1e-6–1e-8 is GPU run-to-run
floating-point noise, not the transport — feature tensors are byte-identical):

| step | metric | disagg | colocated |
|---|---|---|---|
| 20 | acceptance_rate | 0.0013300 | 0.0013300 |
| 20 | ploss | 5.386736 | 5.386740 |
| 20 | acc | 0.0272590 | 0.0272590 |
| 120 | acceptance_rate | 0.0223610 | 0.0223505 |
| 180 | acceptance_rate | 0.0337013 | 0.0336982 |

acc / acceptance_rate climb over training in both (baseline direction). Per-step
values are noisy at `batch_size=1` over 64 diverse samples. Note this is the
training-time acceptance proxy; the serving accept-length (τ via spec-decoding) is
a separate eval gate.

# Online DFlash via server-capture (real zero-copy)

The engine-side transport: a live SGLang server does the capture and writes
straight into Mooncake, so inference and training are fully separate processes
that share only the object store — no in-process target, no shared *data* mount.

```
 inference pool (GPU)                 mooncake                training pool (GPU)
 ────────────────────                ─────────               ───────────────────
 patched SGLang server  ──put()────▶  object    ──get()────▶ FeatureDataLoader
 (--enable-spec-capture)   RDMA/TCP    store       zero-copy  -> OnlineDFlashModel
        ▲                                                     TrainerController.fit()
        │ /generate + spec_capture   (keys only)
 SGLangServerCaptureAdapter ───────▶ StreamingRefChannel ───▶ (consumer reads refs)
```

The server is stock `sglang==0.5.14` patched with
`patches/sglang/v0.5.14/spec-capture.patch` — apply it to the installed package
with `scripts/apply_sglang_spec_capture_patch.sh`. Only tiny `SampleRef`s cross
the ref channel; the DFlash context hidden states go server → Mooncake →
trainer with no re-copy. Strategy-agnostic: the same server serves eagle3 or
dflash requests, named per request by the client
(`specforge/inference/adapters/server_capture.py`).

## Run it (single 8-GPU node)

```bash
export WANDB_API_KEY=<key>            # or add --report-to none in the wrapper
bash examples/disagg/run_qwen3.6_27b_dflash_disagg.sh
```

The wrapper starts a `mooncake_master`, the patched server (GPU 0), a thin CPU
producer, and the trainer (GPU 1). `--spec-capture-aux-layer-ids` must match
`dflash_config.target_layer_ids` in `configs/qwen3.6-27b-dflash.json` (the
producer reads the same list for contract verification). Mooncake connection is
the standard `MOONCAKE_*` env (see the wrapper for defaults).

The transport is gate-validated end-to-end (patched server → Mooncake → one real
train step, with aux parity vs an HF reference) in
`tests/test_runtime/test_server_capture_gate.py`
(`SPECFORGE_RUN_SERVER_CAPTURE_TESTS=1`, GPU); the contract/ref/adapter logic is
covered on CPU in `tests/test_runtime/test_server_capture.py`.

## Run it (two physical nodes with RCLI)

`run_qwen3_8b_dflash_disagg_2node.sh` separates the online Qwen3-8B workload by
role: node rank 0 runs Mooncake, one TP=1 capture server, and the CPU producer;
node rank 1 runs the DP=4 consumer/trainer. The two nodes exchange feature
tensors through Mooncake. A shared filesystem is still required for the small
ref channel, logs, and lifecycle markers under `DISAGG_RUN_ROOT`.

Create a unique run ID and start one detached session per rank with identical
environment settings (replace the job, repository, and dataset paths):

```bash
JOB=your-two-node-job
RUN_ID=qwen3-8b-dflash-$(date +%Y%m%d-%H%M%S)
REMOTE_ROOT=/workspace/SpecForge
DATA_PATH=/workspace/data/perfectblend_train.jsonl
REMOTE_CMD="cd $REMOTE_ROOT && DISAGG_STORE_ID=$RUN_ID TRAIN_DP=4 REPORT_TO=none TRAIN_DATA_PATH=$DATA_PATH bash examples/disagg/run_qwen3_8b_dflash_disagg_2node.sh"

rcli exec -d --node-rank 0 --name dflash-inference "$JOB" "$REMOTE_CMD"
rcli exec -d --node-rank 1 --name dflash-training "$JOB" "$REMOTE_CMD"
```

The job scheduler must place the two pods on different physical hosts;
`--num-nodes 2` alone does not guarantee this, so verify physical placement
before launching. Both pods must share the repository/output path and have
direct network reachability for the Mooncake and SGLang ports. The launcher
consumes Kubernetes GPU UUIDs from `NVIDIA_VISIBLE_DEVICES`; override
`SERVER_GPUS`, `CONSUMER_GPUS`, or `TRAIN_DP` for other allocations. A
`CONSUMER_GPUS` override must list exactly `TRAIN_DP` devices. The tested
topology allocates four GPUs per pod, uses one GPU on rank 0, and trains DP=4 on
rank 1. Set `REPORT_TO=wandb` and the usual W&B variables to enable remote
reporting. `DISAGG_IDLE_TIMEOUT` defaults to 600 seconds so a lost inference
pod cannot leave the consumer blocked forever.

## Multi-server (scale the inference pool)

```bash
bash examples/disagg/run_qwen3.6_27b_dflash_disagg_multiserver.sh   # 2x TP=2 + DP=2
```

`DISAGG_SERVER_URLS` (comma-separated) fans the producer out: one
`SGLangServerCaptureAdapter` + one `RolloutWorker` *per server*, each on its own
thread, leasing **disjoint** prompts from the one controller — N servers prefill
concurrently into the same Mooncake namespace (every server registers a segment
with the one master; the trainer fetches by key, oblivious to which server
captured). All servers must run the same model + capture flags.

Failure semantics (`specforge/launch.py:build_disagg_online_producer`): a dead
server's worker fails its leases retryable (survivors re-lease them) and is
dropped after `max_worker_failures` consecutive errors; all workers dead with
prompts remaining raises instead of truncating; a prompt the pool rejects every
time goes terminal after `max_prompt_attempts`. CPU coverage:
`tests/test_runtime/test_disagg_multiserver.py`.
