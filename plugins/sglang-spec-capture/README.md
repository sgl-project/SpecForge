# specforge-sglang-capture

An SGLang plugin that writes a target model's prefill hidden states straight
into Mooncake for SpecForge's disaggregated training. It replaces the
versioned source patch under `patches/sglang/` on SGLang builds that ship the
generic extension points it uses:

- `--enable-return-aux-hidden-states` / `--return-aux-hidden-state-layer-ids`:
  aux hidden-state capture on a target without a draft. `--speculative-algorithm`
  only names the capture convention (EAGLE3, DFLASH or DSPARK); SGLang resets
  it, so no draft runs. The post-norm last hidden states are kept next to the
  aux concatenation.
- `ModelRunner.forward_observer`: reads each prefill's hidden states and hands
  them to the auxiliary-output path.
- `Scheduler.defer_response(req, completion)`: SGLang holds a finished
  request until its objects are written, then reads the completion on the
  scheduler thread before returning capture metadata. Cancelling one request
  does not cancel a shared write needed by other requests.

## Install

This version requires SGLang's [aux-capture changes in PR #43449](https://github.com/sgl-project/sglang/pull/43449)
and the completion-based `defer_response` revision of
[PR #43450](https://github.com/sgl-project/sglang/pull/43450), or a branch containing
both. These extension points are not available in stock SGLang 0.5.18.

Use separate environments for the SGLang server and SpecForge training.
SpecForge currently pins SGLang 0.5.18 and different Torch/Transformers versions;
installing the main SpecForge package into the server environment would replace
the server's dependencies. The server only needs this plugin, its SGLang build,
and Mooncake; the plugin does not import the main SpecForge package.

Install into the SGLang server's environment. The plugin declares no
dependencies, so the server's pins stay intact:

```bash
pip install --no-deps plugins/sglang-spec-capture
```

Install a Mooncake wheel compatible with the server's CUDA version separately.
For a live store test, it must provide both `mooncake.store` and the
`mooncake_master` executable (for example, `mooncake-transfer-engine-cuda13`
on CUDA 13).

Source installation requires the Rust toolchain pinned by SGLang, a C/C++
compiler, and CUDA development tools. An existing compatible SGLang server
environment can also be used.

For example, from the SpecForge checkout on a CUDA 13 host, with an SGLang
checkout containing both changes:

```bash
python3.12 -m venv .venv-sglang .venv-training
.venv-sglang/bin/python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
.venv-sglang/bin/python -m pip install -e /path/to/sglang/python mooncake-transfer-engine-cuda13
.venv-sglang/bin/python -m pip install --no-deps plugins/sglang-spec-capture
.venv-training/bin/python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
.venv-training/bin/python -m pip install -e . mooncake-transfer-engine-cuda13
.venv-sglang/bin/python -m pip check
.venv-training/bin/python -m pip check
```

SGLang discovers the plugin through the `sglang.srt.plugins` entry point. If
`SGLANG_PLUGINS` is set, it must include `specforge_spec_capture`.

## Launch a capture server

```bash
SPECFORGE_SPEC_CAPTURE=1 \
MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:50051 \
MOONCAKE_METADATA_SERVER=http://127.0.0.1:8080/metadata \
.venv-sglang/bin/python -m sglang.launch_server --model-path <target> --skip-tokenizer-init \
  --chunked-prefill-size -1 \
  --speculative-algorithm DFLASH --enable-return-aux-hidden-states \
  --return-aux-hidden-state-layer-ids 1 16 31 46 61 \
  --return-hidden-states-mode full
```

The scheduler refuses to start if capture is enabled without these flags.

SpecForge selects the plugin with `deployment.disaggregated.server_capture`.
The default, `auto`, finds it without configuration:

- Managed-local servers: while planning, SpecForge probes each capture
  server's Python. It selects the plugin when `specforge_sglang_capture`,
  `sglang.srt.model_executor.forward_observer` and
  `sglang.srt.managers.deferred_output` (with `ResponseCompletion`, the
  `defer_response` completion protocol) import. Otherwise it selects the
  patch if the patched SGLang imports. It then renders the flags above and
  `SPECFORGE_SPEC_CAPTURE=1`. `specforge train --plan` shows the result.
- External servers: before the first request, the producer reads each
  server's `/server_info`. The enabled plugin adds `spec_capture_backend:
  plugin` to every scheduler's `internal_states`.

`server_capture: plugin`, or `DISAGG_SERVER_CAPTURE=plugin` when the config
leaves it at `auto`, skips detection.

For managed servers in a separate environment, set the server's Python
executable explicitly while launching SpecForge from its training environment:

```yaml
deployment:
  disaggregated:
    managed_local:
      capture_servers:
        - port: 32000
          cuda_visible_devices: ["0"]
          tp_size: 1
          python_executable: /path/to/sglang-venv/bin/python
```

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
`SGLANG_SPEC_CAPTURE_ARENA_BYTES`, which the scheduler checks at
startup. Reserve that much HBM below `--mem-fraction-static`.

Before submitting capture, the client discovers `spec_capture_control` in
`/server_info`'s `internal_states`. The plugin's writer scheduler exposes a TCP
listener without allocating the GPU arena; the first capture reuses that
listener. This lets clients clean up objects even if the first response is lost.

Each result also carries `arena: {session, control}`, and every feature gains
an `address`. Objects keep the store keys and stay until a client frees them:
the client writes a JSON list of keys, one line per list, to the `control`
TCP endpoint. Frees are one-way. The writer thread reads them before it
allocates, so no extra thread competes with the scheduler for the GIL. If a
batch does not fit, all its objects share one 30 s allocation deadline. On
timeout, the writer finishes queued copies and releases objects placed by that
batch before reporting failure.

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
| `SGLANG_SPEC_CAPTURE_ARENA_BYTES` | Arena size in bytes. Required with `MOONCAKE_PROTOCOL=nvlink`. |
| `SGLANG_SPEC_CAPTURE_CONTROL_PORT` | Port of the NVLink free endpoint (default: ephemeral). |
| `MOONCAKE_*` | Mooncake client settings. |
