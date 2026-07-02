# O1.3 Capture Spike — findings & transport decision (July 2026)

Run on: `lmsysorg/sglang` dev image (sglang `0.0.0.dev1+g70df09b83`, torch 2.11,
H200), stock server, tiny + real-model probes over HTTP.

## What the stock SGLang server exposes today

| Surface | Verdict |
|---|---|
| `GenerateReqInput.return_hidden_states` (+ `--enable-return-hidden-states`) | **Yes** — per request, `meta_info["hidden_states"]`: one entry per forward pass (the prefill entry carries every prompt position), **LAST layer only** |
| Raw-token serving (`--skip-tokenizer-init`, `input_ids` in `/generate`) | **Yes** — the capture path needs no tokenizer round-trip |
| EAGLE3 aux-layer capture per request (3-layer concat) | **No** — `set_eagle3_layers_to_capture` exists on models but is only driven by the *speculative-serving* path (target+draft); nothing routes it into `/generate` responses or a store |
| Engine-side write-to-FeatureStore transport | **No** |

Calibration: TorchSpec closed exactly this gap with a **17-file / ~530-line
patch to sglang** (custom `/generate_for_spec_training` endpoint,
`spec_training_data_id` threading, logits-processor aux capture, scheduler-side
Mooncake writes). That is the cost of engine-side capture without an upstream
API.

## Decision — reforward transport now, upstream API next

`SGLangServerEagle3TargetEngine` (landed with this spike) is **patch-free on a
stock server**:

1. the live server does the decoding (`/generate`, raw input_ids, greedy by
   default — the frozen-target stream is reproducible);
2. an in-process capture engine over the same weights (hf/sglang/custom — the
   extraction-gate-validated backends) runs ONE extend over
   `[prompt + completion]` for aux+target features.

Same `Eagle3TargetOutput` contract as every backend, so
RolloutWorker → FeatureStore → SampleRef and the trainer are untouched; the
"Done when" of O1.3 (live server feeds training with zero precomputed
features) holds — the capture *compute* runs producer-side until the engine
can emit aux states itself. Gate: `tests/test_runtime/test_o13_server_capture.py`
(opt-in, `SPECFORGE_RUN_SERVER_TESTS=1`; launches a real server).

Cost note: the reforward pays one extra prefill over the generated sequence on
the producer pool. Decode dominates end-to-end time for realistic
completion:prompt ratios; the extra prefill is batched and runs on the
producer GPUs, not the server. Measure per model before scale-out (O2).

## The upstream proposal (what sglang should grow, so both the reforward and
TorchSpec's patch evaporate)

SpecForge is sgl-project's own framework — this is ours to land:

1. per-request `return_aux_hidden_states: list[int]` (layer ids) on
   `/generate`, reusing the existing `set_eagle3_layers_to_capture` model hook
   + `CaptureHiddenMode.FULL` plumbing that speculative serving already has;
2. (transport, later) an optional engine-side sink so captured tensors can
   land in a store instead of the HTTP response (the W3 Mooncake path — the
   response then carries keys, TorchSpec-style, but behind a stable API).

With (1) alone, `SGLangServerEagle3TargetEngine` drops its inner engine and the
W3′ inline-HTTP transport becomes real; with (2), W3 capture-into-store needs
no producer-side GPUs at all.
