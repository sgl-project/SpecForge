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
