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

## Multi-node NVLink

On GB200/GB300 NVL72 systems, `MOONCAKE_PROTOCOL=nvlink` moves captures from
GPU to GPU without the Mooncake store. Mooncake's NVLink transport only exports
fabric memory, and the store cannot place objects there. So the sink copies
captures into one arena that the TransferEngine allocates on the writer GPU,
on the first capture. Its size comes from the required
`SGLANG_SPEC_CAPTURE_NVLINK_ARENA_BYTES`, which the scheduler checks at
startup. Reserve that much HBM below `--mem-fraction-static`.

Each result also carries `nvlink: {session, control}`, and every feature gains
an `address`. Objects keep the store keys and stay until a client frees them:
the client writes a JSON list of keys, one line per list, to the `control`
TCP endpoint. Frees are one-way. The writer thread reads them before it
allocates, so no extra thread competes with the scheduler for the GIL. If a
batch does not fit, the writer waits up to 30 s for frees, then fails the
batch.

NVLink always publishes device tensors, so `SGLANG_SPEC_CAPTURE_GPU_PUT=0` is
rejected. You need a Mooncake build with `USE_MNNVL` (the aarch64 CUDA
wheels), an IMEX channel in the container, and `MC_FORCE_MNNVL=1` on hosts
with RDMA NICs. Without it, Mooncake installs its RDMA transport instead.

## Environment

| Variable | Meaning |
|---|---|
| `SPECFORGE_SPEC_CAPTURE=1` | Enable the plugin (otherwise it does nothing). |
| `SGLANG_SPEC_CAPTURE_GPU_PUT` | `1`: publish device tensors (needs RDMA). `0`: publish from pinned host memory. Unset: device on RDMA+CUDA or NVLink, host otherwise. |
| `SGLANG_SPEC_CAPTURE_MAX_PENDING_BATCHES` | Scheduler batches the writer may hold (default 2). |
| `SGLANG_SPEC_CAPTURE_TIMING=1` | Log per-batch publish timings. |
| `SGLANG_SPEC_CAPTURE_NVLINK_ARENA_BYTES` | Arena size in bytes. Required with `MOONCAKE_PROTOCOL=nvlink`. |
| `SGLANG_SPEC_CAPTURE_CONTROL_PORT` | Port of the NVLink free endpoint (default: ephemeral). |
| `MOONCAKE_*` | Mooncake client settings. |
