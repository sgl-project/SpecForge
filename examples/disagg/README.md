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
