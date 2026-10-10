# specforge-sglang-capture

An SGLang plugin that writes a target model's prefill hidden states straight
into Mooncake for SpecForge's disaggregated training. It replaces the
versioned source patch under `patches/sglang/` on SGLang builds that ship the
generic extension points it uses:

- `--aux-hidden-state-capture` / `--aux-hidden-state-layer-ids`: aux
  hidden-state capture on a target without a draft. The post-norm last hidden
  states are kept next to the aux concatenation.
- `ModelRunner.forward_observer`: reads each prefill's hidden states and hands
  them to the auxiliary-output path.
- `DeferredOutputSource`: holds a finished request's response until its
  objects are written.

## Install

Install into the SGLang server's environment. It declares no dependencies, so
the server's pins stay intact:

```bash
pip install --no-deps plugins/sglang-spec-capture
```

SGLang discovers the plugin through the `sglang.srt.plugins` entry point. If
`SGLANG_PLUGINS` is set, it must include `specforge_spec_capture`.

## Launch a capture server

```bash
SPECFORGE_SPEC_CAPTURE=1 \
MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:50051 \
MOONCAKE_METADATA_SERVER=http://127.0.0.1:8080/metadata \
python -m sglang.launch_server --model-path <target> --skip-tokenizer-init \
  --chunked-prefill-size -1 \
  --aux-hidden-state-capture dflash --aux-hidden-state-layer-ids 1 16 31 46 61 \
  --return-hidden-states-mode full
```

The scheduler refuses to start if capture is enabled without these flags.
SpecForge's managed launcher renders them when
`deployment.disaggregated.server_capture: plugin` is set. For an external
server, set that config field or `DISAGG_SERVER_CAPTURE=plugin` so the
producer sends specs the plugin's way.

## Requests

The producer sends `max_new_tokens=0` requests. Each one carries its spec as a
JSON string in `sampling_params.custom_params["spec_capture"]`:
`{store_id, sample_id, gen, replace, features, passthrough}`.

The response's `meta_info["spec_capture"]` holds `[result]`, where `result` is
`{sample_id, store_id, gen, aux_layer_ids, features: {name: {shape, dtype}}}`
or `{sample_id, error}`. Objects are stored at
`{store_id}/{sample_id}/g{gen}/{name}`.

## Environment

| Variable | Meaning |
|---|---|
| `SPECFORGE_SPEC_CAPTURE=1` | Enable the plugin (otherwise it does nothing). |
| `SGLANG_SPEC_CAPTURE_GPU_PUT` | `1`: publish device tensors (needs RDMA). `0`: publish from pinned host memory. Unset: device on RDMA+CUDA, host otherwise. |
| `SGLANG_SPEC_CAPTURE_MAX_PENDING_BATCHES` | Scheduler batches the writer may hold (default 2). |
| `SGLANG_SPEC_CAPTURE_TIMING=1` | Log per-batch publish timings. |
| `MOONCAKE_*` | Mooncake client settings. |
