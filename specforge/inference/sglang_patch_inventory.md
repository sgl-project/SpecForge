# SGLang patch inventory and supported version

SpecForge pins `sglang==0.5.14` by default. The online patch is also kept
compatible with SGLang's public `inkling-support` layout, and a separately
versioned patch supports the Kimi K3 SGLang fork at the current validated
`kimi-k3` branch tip `9acd9cb` (and its original `f8493a4` integration point).
There are two deliberately separate SGLang integration surfaces.

## Online: external spec-capture server

Online training uses one of these source-specific patches:

| Target | Patch | Capture methods |
|---|---|---|
| SGLang v0.5.14 / `inkling-support` | [`patches/sglang/v0.5.14/spec-capture.patch`](../../patches/sglang/v0.5.14/spec-capture.patch) | EAGLE3, DFlash, DSpark (DSpark reuses the DFlash aux wiring; stock models expose only `set_dflash_layers_to_capture`) |
| Kimi K3 SGLang `9acd9cb` (`f8493a4` compatible) | [`patches/sglang/kimi-k3-f8493a4/spec-capture.patch`](../../patches/sglang/kimi-k3-f8493a4/spec-capture.patch) | EAGLE3, DFlash, DSpark |

The patch adds `--enable-spec-capture` and a server-side sink that:

1. captures requested auxiliary and final hidden states during prefill;
2. writes tensors directly into Mooncake using
   `MooncakeFeatureStore`'s key layout; and
3. returns only key, shape, and dtype metadata in
   `meta_info["spec_capture"]`.

The client boundary is
[`adapters/server_capture.py`](adapters/server_capture.py). Algorithm-owned
providers map generic server artifacts (`aux`, `last_hidden`, passthrough
inputs) to training feature names. No trainer or producer process imports
SGLang model-runner internals or loads a target model.

The same patch is dry-run validated against the v0.5.14 tag and SGLang #31847
commit `b7252cc`. Capture requests carry a unique `extra_key`, so every
training sample executes a full prefill even when radix cache support is
present. Managed-local launch preserves the historical disabled-cache default;
hybrid targets that require the unified radix tree set
`model.sglang_disable_radix_cache: false`.

For targets that declare `logits_mup_width_multiplier`, the SGLang model passes
an LM-head-scaled hidden state into the logits processor. The capture patch
restores the pre-head-scale post-norm representation because SpecForge folds
the same multiplier into the frozen target head used during training.

Apply the default patch with `scripts/apply_sglang_spec_capture_patch.sh`, or
the K3 patch with
`scripts/apply_sglang_spec_capture_patch.sh --target kimi-k3-9acd9cb`.
The K3 patch routes `--spec-capture-method dspark` to the model's dedicated
`set_dspark_layers_to_capture` hook. It also keeps 64K capture correct by using
64-bit Triton pointer arithmetic, scale-stable residual scoring, and a generic
Marlin reduction fallback when the token dimension exceeds CUDA grid.y's
65,535 limit. The server-capture unit and GPU gates must pass before updating
either supported source revision.

## Online-live: capture from production serving traffic

[`patches/sglang/online-live/spec-capture-live.patch`](../../patches/sglang/online-live/spec-capture-live.patch)
is a separate layer on top of the v0.5.14 base patch (apply with
`scripts/apply_sglang_spec_capture_patch.sh --live`, or let
`scripts/online_live/launch_capture_server.py` apply it and launch the server
in one command). It captures ordinary
user requests — no client `spec_capture` field — and pushes the resulting
records to a SpecForge producer, closing the serve→train loop:

- Two new flags: `--spec-capture-intake-url http://<producer>:<port>` and
  `--spec-capture-sample-rate` (deterministic per request id, so all TP ranks
  make the same capture decision).
- At startup the sink GETs `/v1/spec-capture/config` from the intake
  ([`runtime/data_plane/intake_server.py`](../runtime/data_plane/intake_server.py)):
  store id, feature names, passthrough synthesis rules (`input_ids` from the
  request tokens, `loss_mask`/`attention_mask` all-ones), and the token cap.
- Capture covers prefill **and** generated tokens: one row per fed token, so a
  finished request yields `prompt + output[:-1]` rows aligned with the
  `input_ids` passthrough.
- Tensors still go straight into Mooncake via the base sink; the tensor-free
  record then POSTs to `/v1/spec-capture/records` from a bounded background
  writer queue (never the scheduler loop). Any failure — queue full, Mooncake
  down or pool full, intake down, 429 shed, 422 reject — drops just that
  capture, removes any keys already written, and never touches the user's
  response. Flow control is therefore shed-based; the Mooncake pool size is
  the final buffer bound and trainer acks free it.

v1 limitations: `--chunked-prefill-size -1` is still required; the live
capture server cannot itself serve with speculative decoding; a retracted
request drops its capture; a sink crash between the Mooncake write and the
key removal leaks hard-pinned orphans until the store is restarted.

## Offline: dedicated local capture

[`../offline_capture`](../offline_capture) is used exclusively by
`scripts/prepare_hidden_states.py`. Its `sglang_backend` owns the local,
version-pinned APIs required for offline EAGLE3 preprocessing:

| Dependency | Upgrade risk |
|---|---|
| `CaptureHiddenMode.FULL` and logits-processor replacement | hidden-state output fields or pruning behavior may change |
| `set_eagle3_layers_to_capture` / `set_dflash_layers_to_capture` / `set_dspark_layers_to_capture` | strategy-specific layer-selection APIs may move |
| `ScheduleBatch`, `ForwardBatch`, and `ModelRunner` construction | constructor and memory-pool setup may change |
| splitting captured states by request input length | token packing conventions may change |
| DP-attention/model-parallel initialization patches | distributed group signatures may change |

This package computes no logits and supports text EAGLE3, DFlash, Domino, and
K3 DSpark state capture needed by the preprocessing script. It does not provide
HF/custom backends, VLM capture, online rollout, or a general target-engine
factory.

`tests/test_runtime/test_sglang_0514_compat.py` guards the patched 0.5.14 API
seams, and
`tests/test_offline_capture/test_sglang_backend.py`
provides the GPU smoke coverage for dense and MoE offline capture.
