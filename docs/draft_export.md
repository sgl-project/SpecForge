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
