# Draft export

Canonical training checkpoints include `config.json` beside training state.
Its `specforge_artifact` field records format metadata outside the draft structure.
Export reads that configuration; no training YAML is needed.

```bash
specforge export --format hf --checkpoint RUN/STEP --output-dir MODEL
```

HF export writes canonical configuration, exact owned tensors, and model code
for `AutoModel.from_pretrained(MODEL, trust_remote_code=True)`. SpecForge need
not be installed. Use `specforge.loading.load_draft(MODEL, target=TARGET)` inside
SpecForge. Target-bound token tables are omitted; callers must bind real target
weights before running the HF model. They are never randomly initialized.
SGLang cannot serve this directory; it needs a separate serving export.
Model class names such as `DFlashDraftModel` may be shared by both implementations;
their configuration formats and weight layouts still differ.

A historical checkpoint without `config.json` additionally needs
`--draft-config OLD.json --target-model TARGET`. The target configuration resolves
legacy target-derived semantics. Legacy training remains available during the
migration; its old checkpoints must use this import path. New HF exports always
use canonical names and preserve each tensor's dtype. `--to` remains an alias for
`--format`; `--embedding-source` is a deprecated alias for `--target-model` for HF
export, and no longer embeds target tables in the output.

Export does not replace vocabulary mappings or accept arbitrary weight renames.
Restore those explicitly before exporting. Missing, unexpected or misshaped
weights fail before publication. A nonempty output directory is rejected.

## SGLang

```bash
specforge export --format sglang --checkpoint RUN/STEP --output-dir SERVING
# Native dense Qwen3.5 MTP: target checkpoint with its MTP tensors replaced.
specforge export --format sglang --checkpoint RUN/STEP --target-model TARGET \
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
different head geometry across layers, noncausal sliding windows and window=1
fail instead of using SGLang defaults. Unsupported component kinds and binding
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
