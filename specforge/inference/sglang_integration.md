# SGLang capture integration

## Online: capture plugin

Online capture uses [specforge-sglang-capture](../../plugins/sglang-spec-capture/README.md).
Install it in an SGLang environment with native auxiliary hidden-state capture,
`ModelRunner.forward_observer` and completion-based `Scheduler.defer_response`.
Use SGLang commit `c757c4bb7c55d1eb69fdb2bdc6e6fb8507acb7a0`; released 0.5.23 does not expose those APIs. SpecForge does not ship source
patches or fall back to patched servers.

The launcher accepts `server_capture: auto` or `plugin`. Managed-local `auto`
checks each capture interpreter; external `auto` checks the enabled plugin's
`spec_capture_backend` marker in `/server_info`. Missing support fails before
capture. The server needs `SPECFORGE_SPEC_CAPTURE=1` and:

```bash
--speculative-algorithm <EAGLE3|DFLASH|DSPARK> \
--enable-return-aux-hidden-states \
--return-aux-hidden-state-layer-ids <layer ids> \
--return-hidden-states-mode full --chunked-prefill-size -1
```

Requests carry a JSON spec in `sampling_params.custom_params["spec_capture"]`.
The plugin publishes auxiliary and unscaled post-norm final hidden states to
Mooncake, then returns metadata in `meta_info["spec_capture"]`. Algorithm
providers map these generic artifacts to their training feature names.
The client boundary is [server_capture.py](adapters/server_capture.py).

Every attempt uses a fresh `extra_key` so radix-cache hits cannot omit tokens.
Publication runs on a background thread, bounded by
`SGLANG_SPEC_CAPTURE_MAX_PENDING_BATCHES`. `SGLANG_SPEC_CAPTURE_GPU_PUT`
controls host/device publication over TCP/RDMA. With `MOONCAKE_PROTOCOL=nvlink`,
the plugin uses a fabric-memory arena sized by `SGLANG_SPEC_CAPTURE_ARENA_BYTES`;
clients discover the cleanup endpoint before the first write. See the plugin
guide for transport setup and lifecycle details.

## Offline: plugin capture to files

[offline_capture](../offline_capture) connects `scripts/prepare_hidden_states.py`
to plugin servers through `--server-urls`. The CPU client checks the target,
capture method and layer order, reads Mooncake features, and writes the existing
algorithm-specific `.ckpt` or `.ckpt.gz` files before releasing temporary objects.
There is no local ModelRunner or SGLang patch in the preparation client.
See [offline preparation](../../examples/offline/README.md) for server setup,
worker sharding and resume behavior.

Training from already prepared feature files does not require a capture server.
