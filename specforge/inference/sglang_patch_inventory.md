# SGLang patch inventory and supported version

SpecForge pins `sglang==0.5.14`. There are two deliberately separate SGLang
integration surfaces.

## Online: external spec-capture server

Online training uses
[`patches/sglang/v0.5.14/spec-capture.patch`](../../patches/sglang/v0.5.14/spec-capture.patch).
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

Apply the patch with `scripts/apply_sglang_spec_capture_patch.sh`. The
server-capture unit and GPU gates must pass before updating the SGLang pin.

## Offline: dedicated local capture

[`../offline_capture`](../offline_capture) is used exclusively by
`scripts/prepare_hidden_states.py`. Its `sglang_backend` owns the local,
version-pinned APIs required for offline EAGLE3 preprocessing:

| Dependency | Upgrade risk |
|---|---|
| `CaptureHiddenMode.FULL` and logits-processor replacement | hidden-state output fields or pruning behavior may change |
| `set_eagle3_layers_to_capture` / `set_dflash_layers_to_capture` | strategy-specific layer-selection APIs may move |
| `ScheduleBatch`, `ForwardBatch`, and `ModelRunner` construction | constructor and memory-pool setup may change |
| splitting captured states by request input length | token packing conventions may change |
| DP-attention/model-parallel initialization patches | distributed group signatures may change |

This package computes no logits and supports text EAGLE3 and DFlash-family
state capture needed by the preprocessing script. It does not provide
HF/custom backends, VLM capture, online rollout, or a general target-engine
factory.

`tests/test_runtime/test_sglang_0514_compat.py` guards the patched 0.5.14 API
seams, and
`tests/test_offline_capture/test_sglang_backend.py`
provides the GPU smoke coverage for dense and MoE offline capture.
