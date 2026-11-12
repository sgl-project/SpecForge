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

A historical checkpoint without `config.json` additionally needs
`--draft-config OLD.json --target-model TARGET`. The target configuration resolves
legacy target-derived semantics. Legacy training remains available during the
migration; its old checkpoints must use this import path. New HF exports always
use canonical names and preserve each tensor's dtype.

During migration, `--to hf` and `--to sglang` retain the original exporters and
require `--draft-config`. In particular, legacy DFlash serving scripts can continue
using `--to hf`; this is distinct from the default canonical export. Only legacy
`--to hf` accepts `--embedding-source` and copies that embedding. Canonical export
rejects it and uses `--target-model` for import validation and configuration.
`--format` is not needed. The SGLang converter and retirement of `--to` belong to
the next export step (#1162); no converter flag is advertised before it exists.

Export does not replace vocabulary mappings or accept arbitrary weight renames.
Restore those explicitly before exporting. Missing, unexpected or misshaped
weights fail before publication. A nonempty output directory is rejected.

Both `save_draft(model, ...)` and canonical HF export use `write_artifact` for
sharding, metadata and atomic directory publication. Model saving performs its
wrapper/materialization checks first; export validates its checkpoint weight plan
first. Training optimizer/RNG state remains owned by the training checkpoint manager.
