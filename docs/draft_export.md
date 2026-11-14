# Draft export

Canonical training checkpoints include `config.json` beside training state.
Its `specforge_artifact` field records format metadata outside the draft structure.
Export reads that configuration; no training YAML is needed.

```bash
specforge export --checkpoint CHECKPOINT_PATH --output-dir MODEL
```

`--export-mode standalone` is the default for every architecture, including MTP,
DFlash and DSpark. It publishes a canonical draft directory. Native merge-back is
an explicit public export mode and does not require a serving engine:

```bash
specforge export --export-mode native --checkpoint CHECKPOINT_PATH \
  --target-model TARGET --output-dir MERGED_TARGET
```

Native export selects an adapter by target model type and draft architecture.
A missing adapter is an error; it never falls back to standalone. Registering an
adapter does not change standalone exports. Dense Qwen3.5 MTP is the current
built-in adapter; other combinations can register their own implementation.
`--copy-target` is native-only. Native artifacts retain the target's configuration
format, with `specforge_artifact: {format: native, version: 1}`; reload the canonical
draft checkpoint to continue draft training.

Native export requires explicit target computation fields and checks them against
the draft without changing the target configuration. Missing or conflicting fields
are rejected; target RoPE parameters never inherit draft defaults. It rewrites shards containing old `mtp.*`
tensors and links unaffected shards. `--copy-target` copies unaffected shards for
a self-contained directory. Embedding/head copies are written under native names;
The target is not modified.

`--converter sglang` converts a standalone draft into a serving format. Native
export already writes the target format, so explicit native mode takes no
`--converter`. Serving compatibility is separate from these publication modes.

HF export writes canonical configuration, exact owned tensors, and model code
for `AutoModel.from_pretrained(MODEL, trust_remote_code=True)`. SpecForge need
not be installed. Use `specforge.loading.load_draft(MODEL, target=TARGET)` inside
SpecForge. Target-bound token tables are omitted; callers must bind real target
weights before running the HF model. They are never randomly initialized.
SGLang cannot serve this directory; it needs a separate serving export.
Model class names such as `DFlashDraftModel` may be shared by both implementations;
their configuration formats and weight layouts still differ. The canonical format
is identified by `model_type: specforge_draft` and its schema, not by the class
name alone. `auto_map` selects the bundled implementation for Transformers.
HF code bundling currently supports in-tree SpecForge architectures only; external
architectures and ambiguous flattened module names are rejected before publication.
Bundled code may import only torch, transformers and the standard library, so loading
by Hub repo id needs nothing else; optional packages such as flash-attn are imported
inside `try`, only when their backend is selected.

A historical checkpoint needs `--target-model TARGET`, and also
`--draft-config OLD.json` when it has no `config.json`. The target configuration
resolves legacy target-derived semantics, and any copy of a target table that the
checkpoint saved must equal the target's. Legacy training remains available during
the migration; its old checkpoints must use this import path. New HF exports always
use canonical names and store each tensor in its own dtype; `config.json` declares
a dtype only when all floating-point tensors share one.

The retired `--to`, `--embedding-source` and `--embedding-key` options are
no longer accepted. Use `--converter sglang` for serving conversion and
`--target-model` for target metadata and bound-weight validation.

Historical EAGLE3 checkpoints without recorded vocabulary buffers can supply
an explicit mapping at the checkpoint import boundary:

```bash
specforge export --converter sglang --checkpoint CHECKPOINT_PATH --draft-config OLD.json \
  --target-model TARGET --vocab-mapping MAPPING --output-dir SERVING
```

The imported mapping becomes part of the validated canonical state. Export preserves
recorded `bos_token_id`, `eos_token_id`, and `pad_token_id` metadata.

Export does not replace vocabulary mappings or accept arbitrary weight renames.
Restore those explicitly before exporting. Missing, unexpected or misshaped
weights fail before publication, and so does serving state that was never
initialized (a reduced vocabulary without its mapping, an uninitialized DSpine
transfer space). A nonempty output directory is rejected.

Both `save_draft(model, ...)` and canonical HF export use `write_artifact` for
sharding, metadata and atomic directory publication. Model saving performs its
wrapper/materialization checks first; export validates its checkpoint weight plan
first. Training optimizer/RNG state remains owned by the training checkpoint manager.

## SGLang

```bash
specforge export --converter sglang --checkpoint CHECKPOINT_PATH --output-dir SERVING
# Explicit native export is independent of SGLang.
specforge export --export-mode native --checkpoint CHECKPOINT_PATH --target-model TARGET \
  --output-dir MERGED_TARGET
```

For compatibility, an old `--converter sglang` command with no export mode still
performs native merge-back for Qwen3.5 MTP, with a migration warning. Replace
`--converter sglang` with `--export-mode native`. This exception is fixed to the
historical target/architecture pair; future registrations do not extend it.
Legacy `--to` exporters remain unchanged and do not accept `--export-mode`.

The converter targets SGLang commit
`c757c4bb7c55d1eb69fdb2bdc6e6fb8507acb7a0`, also pinned in CI. Each artifact records
that revision and the converter version in `config.json`'s `specforge_artifact`
field. Conversion needs neither SGLang nor a
GPU. Load/re-export/continued training use the canonical checkpoint; public draft
loading rejects a serving artifact.

| Architecture | Serving format / restriction |
|---|---|
| DFlash | `DFlashDraftModel`; both token tables bound to the target |
| DFlash2 | `DFlash2DraftModel`, including convolution and selector |
| Domino | `DFlashDraftModel` with explicit Domino projector settings |
| DSpark | `DSparkDraftModel`; Markov head required; confidence head optional in static mode |
| EAGLE3 | `LlamaForCausalLMEagle3`; embedding bound, draft head owned |
| MTP | No standalone SGLang converter; use explicit native export for dense Qwen3.5 merge-back |
| DSpine, DMTP, P-EAGLE | Rejected: this pinned SGLang has no matching serving implementation |

Supported DFlash-family layers use uniform dense SiLU MLPs, standard residuals,
GQA/MHA or MLA, and explicit supported RoPE settings. Layer-specific values are
accepted only where SGLang can express them. Mixed causality is supported only
for causal sliding layers with bidirectional full layers; `is_causal: null` selects
that SGLang rule. Different head geometry across layers and window=1 are rejected.
Noncausal sliding windows must cover at least one model block.
The pinned converter rejects muP targets for DFlash2/Domino and offset RMSNorm
for DSpark because their serving paths do not implement those computations. Unsupported component kinds and binding
sets also fail. The pinned DSpark implementation chooses confidence-head use
through its ragged-verification runtime mode; `cap-accept` exercises that head.
Drafts without a confidence head require `SGLANG_RAGGED_VERIFY_MODE=static`.
EAGLE3 supports multi-axis RoPE for VL targets; native MTP uses equal text
positions on all axes.

Compared with historical exports, configuration fields such as causality, heads,
rotary settings and block quantities are now written explicitly. Canonical module
prefixes are removed for serving, vocabulary readiness is validated rather than
serialized, and Domino uses SGLang's registered DFlash architecture name. These
are format changes; tensors are not cast or transformed.

The pinned compatibility gate constructs real SGLang models and loads converted
weights, records top-level and nested configuration reads for non-native models, and compares
FP32 reference outputs. It checks that exported weights reach serving loaders,
uses production recipe fields as an independent baseline, and covers MLA context
projection, scaled/multi-axis RoPE and the DSpark stacked context-KV writer. It uses
a dense attention oracle in place of paged KV-cache attention, so it is not an
end-to-end serving/performance test. Separate tests cover native physical shard
replacement and exact weight recovery, unsupported structures, and rejection of
canonical HF configurations by SGLang's native constructor and generic
Transformers compatibility gate. Run the gate with:

```bash
SPECFORGE_SGLANG_EXPORT_TESTS=1 TORCHDYNAMO_DISABLE=1 \
  python -m unittest tests.test_checkpoint.test_sglang_export \
  tests.test_checkpoint.test_sglang_compatibility -v
```

## Adding family adapters

Keep family-specific code in `specforge/checkpoint/formats/native/models/` or
`specforge/checkpoint/formats/sglang/models/`. Add an explicit import in that
package's `__init__.py` so registration runs when the package loads.

Native adapters call `register_native_target(model_type, architecture, ...)` and
provide validation and export callbacks. They can reuse `native/writer.py` for
weight checks and publication without changing the public exporter.
Standalone serving converters use `register_converter(architecture)`; shared
algorithm conversions remain in `sglang/converters.py`.

The SGLang Qwen3.5 module is a compatibility helper for the historical native
command and parity tests, not a standalone MTP converter. Do not register it
under `MTPDraftModel`: that would incorrectly select Qwen3.5 for every target.
Native registration never selects an export mode or adds standalone engine support.
