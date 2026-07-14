# SGLang Patch Inventory & Supported-Version Matrix (M4 / B7-B8)

This is the published inventory the M4 exit gate requires: every place SpecForge
depends on SGLang internals to extract EAGLE3/DFlash features, and the SGLang
versions that are validated against the extraction-correctness gate
(`test_extraction_vs_hf_reference`).

Per ADR (`open_questions.md` #1), the SGLang patch surface is treated as a
**version-pinned fork**, surfaced through one boundary — `SGLangAdapter` in
[`sglang_adapter.py`](sglang_adapter.py). The decision is **versioned `.patch`
files + a declared version matrix + a CI extraction-drift gate**, not runtime
monkey-patching, so a patch is auditable and an SGLang upgrade can never silently
change what is extracted.

## Extraction surface (what we depend on)

| # | Dependency | Where | Why it can break on upgrade |
|---|---|---|---|
| 1 | `CaptureHiddenMode.FULL` + `LogitsProcessorForEAGLE3` returning `aux_hidden_states` / `last_hidden_states` | `specforge/modeling/target/eagle3_target_model.py` `_extend`, `sglang_backend/` | SGLang may rename the capture mode, change `logits_output` fields, or change aux-state layout |
| 2 | `model.set_eagle3_layers_to_capture(layer_ids)` | `set_aux_hidden_states_layers` | Aux-layer registration API may move/rename |
| 3 | `ScheduleBatch` / `ForwardBatch` / `ModelRunner` construction signature | `_extend` | New required ctor args on minor releases (e.g. `is_draft_worker`, `moe_dp_rank`) |
| 4 | Per-request split of `aux_hidden_states` by `input_lens` | `_extend`, `_get_sharded_return` | Token packing / padding convention changes |
| 5 | `wrap_eagle3_logits_processors_in_module(..., return_full_logits=False)` | `from_pretrained` | Logits-processor wrapping hook may change |

The `SGLangAdapter` is the *only* code that touches these; everything downstream
sees the typed `Eagle3TargetOutput` / per-sample feature dicts.

## Supported version matrix

The extraction-correctness gate must be green on every pinned version. `status`:
**validated** = `test_extraction_vs_hf_reference` green in CI on this version;
**target** = intended next pin.

| SGLang version | torch / transformers | Backend(s) | Aux capture | Status |
|---|---|---|---|---|
| `0.5.5` (`lmsysorg/sglang:v0.5.5`) | 2.x / 4.x | sglang, hf, custom | `set_eagle3_layers_to_capture` | validated (repo CI image) |
| `dev` (`lmsysorg/sglang:dev`) | 2.11 / 5.8 | sglang, hf, custom | `set_eagle3_layers_to_capture` | validated (H200 box) |
| `0.5.9` (pyproject pin) | 2.9.1 / 4.57.1 | sglang, hf, custom | `set_eagle3_layers_to_capture` | superseded by 0.5.14 |
| `0.5.14` (pyproject pin) | 2.9.1 / 4.57.1 | sglang, hf, custom, **server (spec-capture patch)** | `set_eagle3_layers_to_capture` / `--enable-spec-capture` | pin (PR #652); server transport gated by `test_server_capture_gate` |

`0.5.9` is the dependency pin, but it is not an M4 sign-off until the
extraction-correctness gate is green on that exact image/version.

The **HF backend** path (`HFEagle3TargetModel`, forward hooks on the aux layers)
is version-robust and is what `test_extraction_vs_hf_reference` exercises in CI
without a GPU SGLang server; the **sglang backend** is validated on the GPU box.

## CI extraction-drift gate

`test_extraction_vs_hf_reference` asserts the adapter's extracted aux hidden
states equal an independent HF `output_hidden_states=True` forward at the
recorded layer IDs (and target logits match a direct `lm_head`) within a
documented bf16 tolerance (`rtol=atol=2e-2`). Run it against each pinned image;
a failure on a new image blocks the version bump.

## Patch files

When an SGLang upgrade requires source changes to the extraction surface, add a
versioned patch under `patches/sglang/<version>/` and record it in the matrix
above. The in-process backends need none (extraction goes through the public
capture API).

### `patches/sglang/v0.5.14/spec-capture.patch` — server-side capture sink

The **server transport**: 8 small hooks + one self-contained module
(`sglang/srt/spec_capture_sink.py`) that let a live SGLang server, launched
with `--enable-spec-capture [--spec-capture-aux-layer-ids ...]
--chunked-prefill-size -1 --disable-radix-cache`, capture per-request aux/last
hidden states during
prefill and write them straight into a Mooncake store in
`MooncakeFeatureStore`'s zero-copy key layout
(`{store_id}/{sample_id}/g{gen}/{name}`, raw bytes, hard-pinned). The response
`meta_info["spec_capture"]` carries only keys/shapes/dtypes; the client side is
`specforge/inference/adapters/server_capture.py`. Apply to an installed sglang
with `scripts/apply_sglang_spec_capture_patch.sh`. Strategy naming is decided
per request by the client (eagle3 / dflash / domino registered;
`register_server_capture_schema` for new ones). It also answers the stock
`model_runner.py` TODO ("expose this to server args?") — the aux-capture
upstream proposal from the O1.3 spike (PR #641) is exactly this surface.
