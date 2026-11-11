# Canonical draft checkpoints and loading

`specforge.checkpoint` owns model artifacts and weight plans. `specforge.loading`
provides file-based model construction. Neither restores optimizer, scheduler,
step or RNG state; those remain the training layer's responsibility.

```python
from specforge.checkpoint import save_draft
from specforge.loading import initialize_draft, load_draft, read_draft

# config is the resolved, immutable DraftConfig produced by the codec.
model = initialize_draft(config, target="/checkpoints/target")
save_draft(model, "/checkpoints/draft", max_shard_size="5GB")
restored = load_draft("/checkpoints/draft", target="/checkpoints/target")
checkpoint = read_draft("/checkpoints/draft")  # Config and lazy tensor references.
```

The artifact contains canonical `config.json`, safetensors weights and
format metadata in `config.json`'s `specforge_artifact` field. Bound embeddings/heads are omitted; their target source is
required on model load. Tied parameters are stored once. Saving requires a new
or empty directory and materialized, gathered tensors. Normal HF canonical
`save_pretrained` directories are also readable. Source paths may be local or
Hugging Face repositories; custom Python code is never imported from them.

Canonical model classes use names such as `DFlashDraftModel`, `DSpineDraftModel`
and `Eagle3DraftModel`. Names may also occur in legacy or serving implementations;
`model_type: specforge_draft` identifies the canonical configuration. The legacy
`AutoDraftConfig`/`AutoDraftModel` APIs reject canonical inputs; use the entry
points above.

`load_draft` infers a single floating dtype from the stored weights; pass `dtype`
explicitly for mixed-dtype checkpoints or an intentional conversion. It returns
an evaluation-mode model. An optional `draft=config` checks exact structural
identity. It never overrides checkpoint structure.

A warm start uses the new structure and an explicit assignment of its weights:

```python
model = initialize_draft(
    new_config,
    target="/checkpoints/target",
    weights_from="/checkpoints/old-draft",
    mapping={"backbone": "backbone", "fusion": "fusion"},
    allow_new=["markov_head", "confidence_head"],
)
```

Omitting `mapping` selects matching canonical names, excluding `allow_new`.
Every owned tensor must be assigned or explicitly new; unused source tensors
are reported. A warm-start source supplies weights only and is not checked
against the new target configuration; a legacy source's stored copies of
target-bound tables count as source tensors. Mapping keys are draft
prefixes and values are source prefixes. A `*` matches one path segment.

Target copies are declared separately:

```python
model = initialize_draft(
    config,
    target="/checkpoints/target",
    from_target={"backbone.layers.0": {
        "from": "model.layers.35",
        "rename": {"mlp.old_name": "mlp.new_name"},
    }},
)
```

Target copies require safetensors, complete prefix coverage, matching shapes
and bit-preserving dtypes. Explicit renames change names only; no tensor values
are transformed. Copies cannot overlap warm-start assignments or `allow_new`.
Current components use separate checkpoint tensors. A future component with
fused parameters must supply its layout rather than relying on family guesses.

Legacy draft imports use the existing configuration codec and format-owned
weight renames. They require the target configuration. A `training_state.pt`
without `config.json` additionally needs `legacy_config` (a dictionary or JSON
file); tensors come only from `draft_state_dict`, via PyTorch's restricted
unpickler, and recorded settings are checked as described below.
Continuing a reduced-vocabulary EAGLE checkpoint requires its saved `t2d` and
`d2t` mapping. A weights-only warm start can explicitly allow a new `vocabulary`;
the caller must install its mapping before using logits.
Native drafts inside target checkpoints use an explicit preset and
`from_target`, rather than an import adapter. SGLang artifacts are rejected.

For distributed construction, use the same `make_initialization_plan` or
`make_weight_plan` on each rank, then `apply_weight_plan` after materialization;
a scratch or warm-start model then needs `finalize_initialization` as in
`initialize_draft`.
Plans validate global metadata; Shard/Replicate DTensor destinations read only
their local slices. Partial placements are rejected. CLI/training orchestration
and export conversion will adopt these APIs in subsequent integration steps.

Legacy imports check recorded capture, mask, residual and binding settings,
including the DFlash-family `*_target_layer_ids`, `*_mask_token_id` and
`*_block_size` records.
A saved copy of a target-bound table must match the supplied target; importing
must not silently replace trained weights. Hub imports also fetch
`training_state.pt` for these structural records, without restoring training state.

After copying and binding weights, `initialize_draft` calls the model's
`finalize_initialization(loaded_keys=...)`. Models own derived-state initialization
and consistency checks; the loader supplies only the loaded draft tensor names.
DSpine initializes a fresh transfer space here, preserves complete loaded state,
and rejects partial transfer-state loads. `load_draft` does not call this hook.
