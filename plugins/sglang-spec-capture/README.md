# specforge-sglang-capture

An SGLang plugin that writes a target model's prefill hidden states straight
into Mooncake for SpecForge's disaggregated training and offline preparation. It requires an SGLang
build with these generic extension points:

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
  does not cancel a shared write needed by other requests. It queues reclamation
  of that request's objects on the writer after publication, so a client cleanup
  that ran before the objects existed cannot leave them orphaned. Cleanup RPC
  failures are retried and logged if they persist.

## Install

This version requires SGLang's [aux-capture changes in PR #43449](https://github.com/sgl-project/sglang/pull/43449)
and the completion-based `defer_response` revision of
[PR #43450](https://github.com/sgl-project/sglang/pull/43450), or a branch containing
both. Install SGLang manually at commit
`c19e9c7d19a873498c4fee1d789596c1a2b6e6f0` on the `release/v1.0.0` branch; the released
0.5.23 wheel still lacks the forward observer, completion and auxiliary-capture
APIs required here.

The aux-capture API returns `last_hidden_states` after the final norm and
before LM-head-only scaling, including Inkling's muP division. The plugin
preserves those values; SpecForge folds `logits_mup_width_multiplier` into
its frozen target head. Older builds that return scaled states are unsupported;
upgrade the capture environment to a build with the required API.

The supported capture setup uses this SGLang revision with
Torch 2.14.1 and Transformers 5.19.0. SpecForge does not install SGLang as a
package dependency. They can run in the same environment;
the server remains a separate process. Separate environments are also supported
through `capture_servers[].python_executable`. The plugin itself does not import
the main SpecForge package. CI uses a separate `sf-capture` environment for
the live server. Both CI environments install the official revision specified
by `SGLANG_COMMIT` in `.github/workflows/test.yaml`. The separate TP contract
step uses the companion patches after the stock checks finish.

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

For example, from the SpecForge checkout on a CUDA 13 host with the Rust,
C/C++ and CUDA build tools required by the pinned SGLang revision:

```bash
uv venv .venv -p 3.12
source .venv/bin/activate
uv pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install wheel setuptools setuptools-rust setuptools-scm
uv pip install --no-build-isolation --prerelease=allow -e .
uv pip install --no-build-isolation --prerelease=allow \
  "sglang @ git+https://github.com/sgl-project/sglang.git@c19e9c7d19a873498c4fee1d789596c1a2b6e6f0#subdirectory=python"
uv pip install --no-deps plugins/sglang-spec-capture
uv pip install mooncake-transfer-engine-cuda13
```

The SGLang requirement uses the exact commit above; no cache refresh is forced. Rebuild
optional compiled training kernels against the installed Torch when upgrading.

SGLang discovers the plugin through the `sglang.srt.plugins` entry point. If
`SGLANG_PLUGINS` is set, it must include `specforge_spec_capture`.

## Launch a capture server

```bash
SPECFORGE_SPEC_CAPTURE=1 \
MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:50051 \
MOONCAKE_METADATA_SERVER=http://127.0.0.1:8080/metadata \
python -m sglang.launch_server --model-path <target> --skip-tokenizer-init \
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
  `defer_response` completion protocol) import. Otherwise startup fails with
  installation instructions. It renders the flags above and
  `SPECFORGE_SPEC_CAPTURE=1`. `specforge train --plan` shows the result.
- External servers: before the first request, the producer reads each
  server's `/server_info`. The enabled plugin adds `spec_capture_backend:
  plugin` to every scheduler's `internal_states`, together with
  `spec_capture_method` and `spec_capture_aux_layer_ids`. Offline preparation
  validates these fields before submitting prompts; reinstall this plugin on
  existing servers when updating the preparation client.

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

## Opt-in HS capture protocol v1

Ordinary capture uses the official SGLang pin in [Install](#install). A Teacher
with `SPECFORGE_HS_CAPTURE=1` also requires the [six SGLang companion
patches](patches/sglang/c757c4bb/), including when `SPECFORGE_HS_CACHE=0`.
The series adds scheduler-owned prefix admission, exact GDN checkpoints,
cancellation-safe control replies, and rollback and shutdown handling.
At startup the plugin checks `managers.prefix_admission.CAPTURE_ADMISSION_PROTOCOL == 1`.

Use a clean, separate SGLang checkout at the exact base commit below. From
the root of this SpecForge checkout, with the Teacher environment activated
and the source-build prerequisites from [Install](#install) in place, apply
the series in order and install the patched source.

```bash
(
    set -eu
    plugin_dir="$(pwd -P)/plugins/sglang-spec-capture"
    cd /absolute/path/to/sglang
    test "$(git rev-parse HEAD)" = "c757c4bb7c55d1eb69fdb2bdc6e6fb8507acb7a0"
    test -z "$(git status --porcelain)"
    git am "$plugin_dir"/patches/sglang/c757c4bb/000[1-6]-*.patch
    test "$(git rev-parse 'HEAD^{tree}')" = "1e2bd88315469e2d3f522b171613c4d9032491a5"
    uv pip install --no-build-isolation --prerelease=allow -e ./python
    uv pip install --no-deps -e "$plugin_dir"
)
```

The expected source tree is from the validated SGLang commit
`d3dd55e5c8ba260939da7740b8038d483adab412`. Replaying the series with `git am`
can produce different commit IDs with the same source tree.

Enable `SPECFORGE_HS_CAPTURE=1` to negotiate server-issued attempt reservations
at `POST /spec_capture/control`. Set `SPECFORGE_HS_CACHE=1` to reuse hidden-state
prefixes. With only capture enabled, the plugin uses the same complete-output,
byte-budget and immutable-attempt protocol with a zero prefix.

| Variable | Default |
|---|---:|
| `SPECFORGE_HS_CHUNK_TOKENS` | 256 |
| `SPECFORGE_HS_CACHE_BYTES` | 1073741824 |
| `SPECFORGE_HS_PINNED_BYTES` | 268435456 |
| `SPECFORGE_HS_OUTPUT_BYTES` | 536870912 |
| `SPECFORGE_HS_DEVICE_BYTES` | 268435456 |
| `SPECFORGE_HS_MAX_WORKERS` | 4 |
| `SPECFORGE_HS_ATTEMPT_CAPACITY` | 4096 |

All limits must be positive. Pinned staging uses one retained arena, conservatively
rounded down to a power of two, with aligned independent reservation slots.
Cache adoption charges complete underlying storage, including unmatchable tails.
Output, pinned and device limits are separate. Hard oversize fails explicitly;
temporary admission pressure permits only the native transaction's single
fully-rolled-back reclaim/retry. Atomic registry-capacity rejection returns
`error_code=capture_capacity_declined, reservation_granted=false` before granting
any attempt in that begin operation.

HS capture supports complete causal text, nonchunked prefill, PP1/DP1, no
offload/session/LoRA/weight updates, and host publication with Mooncake hard pin.
Set `SGLANG_SPEC_CAPTURE_GPU_PUT=0` when using RDMA host staging. Reuse additionally
requires deterministic batch-invariant inference, enabled radix, and an exact
selected recurrent checkpoint backend. Unsupported modes fail startup rather
than changing capture semantics. Capability metadata advertises these limits.

The content namespace binds model/config/load epoch, ordered raw auxiliary
layers, post-norm last hidden state, dtype/layout, causal position/attention,
TP/KV/checkpoint/backend and deterministic semantics. Tokens form complete parent
hash chains; sample/attempt IDs and loss masks do not enter content keys.
Passthrough data remains part of the registered request digest.

For negotiated HS capture the producer processes one lease per endpoint at a
time. A lease may contain multiple prompts up to the endpoint's advertised
atomic registration limit. Separate endpoints make progress independently;
`runtime.producer_concurrency` still controls ordinary capture.

Each external attempt has immutable keys
`store/sample/g1/attempt/server_epoch/reservation_token/feature`. The native
observer retains independently owned suffix snapshots; its auxiliary output
carries controller jobs and performs no second D2H. Assembly reconstructs every
row `[0,N)` from the leased prefix and the current `[L,N)` suffix. Scheduler safe
points batch ready samples into the native sink, then resolve native deferred
responses. A PUBLISHED certificate is issued only after exact key/schema/digest
validation and source-reader drain. Sink completion permits cache adoption; it
does not retire storage retained by cache leases. Unknown DMA/native transfer
drain quarantines sources, registrations and charges until an external drain
proof exists. Timeouts do not authorize reuse.

Install the plugin in the interpreter running the tests, including for
ordinary test discovery against the official SGLang revision.

`tests.test_runtime.test_hs_runtime` covers CPU cache ownership and assembly;
`tests.test_runtime.test_hs_plugin` covers native-output wiring with a CPU buffer
store; `test_hs_controller_parallel_cuda` requires actual CUDA/pinned copies.
CPU fixtures do not certify numerical model, real Mooncake, scheduler TP or
training equivalence.

Internal TP control/admission collectives use named local phases: owner/poll,
writer payload construction, received-state application, target lookup and
permit validation all vote local errors before entering a later collective.
Ranks agree request membership and control branches before dispatch. Startup
epoch/model/schema/core failures use the same protocol. `poll` and `lifecycle`
themselves remain local because native completion paths may call them only on
the writer. Begin operations preflight a separate registry on every rank;
only unanimous capacity refusal of uncommitted trials can emit the typed
`reservation_granted=false` proof. A rank disagreement or partial commit fails
closed without that proof.

Run the CPU TP contract from the SpecForge checkout using the environment
with the patched SGLang and plugin installed as above.

```bash
python -m unittest tests.test_runtime.hs_tp_protocol -v
```

The contract imports the installed SGLang `PrefixAdmissionCoordinator` and
drives its phase protocol and the plugin runtime from two scheduler threads.
A blocking rendezvous checks collective order under injected faults. A missing
companion module or incompatible protocol fails the contract. Distributed TP,
CUDA and transport failure validation remain separate.
