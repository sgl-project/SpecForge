# Draft export

Canonical training checkpoints include `config.json` beside training state.
Its `specforge_artifact` field records format metadata outside the draft structure.
Export reads that configuration; no training YAML is needed.

```bash
specforge export --checkpoint CHECKPOINT_PATH --output-dir MODEL
```

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

During migration, `--to hf` and `--to sglang` retain the original exporters and
require `--draft-config`. In particular, legacy DFlash serving scripts can continue
using `--to hf`; this is distinct from the default canonical export. Only legacy
`--to hf` accepts `--embedding-source` and copies that embedding. Canonical export
rejects it and uses `--target-model` for import validation and configuration.
`--format` is not needed. Use `--converter sglang` for canonical serving export.
Legacy `--to` paths remain available during migration.

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
# Native dense Qwen3.5 MTP: target checkpoint with its MTP tensors replaced.
specforge export --converter sglang --checkpoint CHECKPOINT_PATH --target-model TARGET \
  --output-dir SERVING
```

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
| DSpark | `DSparkDraftModel`; Markov and confidence heads required by this converter |
| EAGLE3 | `LlamaForCausalLMEagle3`; embedding bound, draft head owned |
| MTP | Dense Qwen3.5 native merge-back; one layer, offset RMSNorm, default RoPE |
| DSpine, DMTP, P-EAGLE | Rejected: this pinned SGLang has no matching serving implementation |

Supported DFlash-family layers use uniform dense SiLU MLPs, standard residuals,
GQA/MHA or MLA, and explicit supported RoPE settings. Layer-specific values are
accepted only where SGLang can express them. In particular, mixed causality,
different head geometry across layers and window=1 fail instead of using SGLang
defaults. Noncausal sliding windows must cover at least one model block.
The pinned converter rejects muP targets for DFlash2/Domino and offset RMSNorm
for DSpark because their serving paths do not implement those computations. Unsupported component kinds and binding
sets also fail. The pinned DSpark implementation chooses confidence-head use
through its ragged-verification runtime mode; `cap-accept` exercises that head.

Compared with historical exports, configuration fields such as causality, heads,
rotary settings and block quantities are now written explicitly. Canonical module
prefixes are removed for serving, vocabulary readiness is validated rather than
serialized, and Domino uses SGLang's registered DFlash architecture name. These
are format changes; tensors are not cast or transformed.

Native export checks target computation fields against the draft, adds missing
fields, and rejects conflicting values. It rewrites shards containing old `mtp.*`
tensors and links unaffected shards. `--copy-target` copies unaffected shards for
a self-contained directory. Embedding/head copies are written under native names;
SGLang's worker also attaches the target token modules. The target is not modified.

The pinned compatibility gate constructs real SGLang models and loads converted
weights, records top-level configuration reads for non-native models, and compares
FP32 reference outputs. It uses
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
