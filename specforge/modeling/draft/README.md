# Draft models and configurations

`DraftConfigClass` (also named `AutoDraftConfig`) is the shared entry point for
reading a setting, validating its architecture, and serializing it for reuse.
It does not allocate model parameters. `AutoDraftModel` constructs the registered
architecture and loads weights separately.

## Layout

```text
draft/
  config/
    auto.py                    loading, finalization, public serialization
    base.py                    immutable options and common validation
    layout.py                  public JSON / legacy runtime translation
    hf.py                      HF config serialization and direct-call adaptation
    components.py              component choices and supported combinations
    eagle3.py, peagle.py        EAGLE-family options
    dflash.py                  DFlash base options
    dspark.py, dflash2.py,
    domino.py, dspine.py        DFlash subclasses and their specific semantics
    mtp.py                     existing Qwen3.5 native MTP options
  registry.py                  explicit architecture-to-model registration
  models/
    eagle3.py, peagle.py        generic EAGLE-family model assembly
    dflash.py, dflash2.py,
    dspark.py, domino.py,
    dspine.py                  generic block-draft model assembly and heads
    base.py, mtp.py             existing model-family contracts
    layers/
      attention/
        dflash.py              context/query execution shared by dense projections
        gqa.py, mla.py          implemented DFlash projection structures
        eagle.py               EAGLE feature/cache attention contract
        masks.py               block visibility, causal and sliding-window masks
        backends/              eager, SDPA, Flash and Flex execution
        parallel/              Ring communication and attention execution
      ffn/                     FFN factory and dense FFN
      convolution/
        dflash2.py             grouped convolution, eager path and backend selection
        dflash2_triton.py      optional Triton forward/backward kernels
      residual/                residual contract and ordinary additive residual
      normalization.py         shared RMSNorm
      kernels.py               optional optimized kernel factories
    specialized/
      base.py                  native checkpoint adapter protocol
      qwen3_5_mtp/
        model.py               existing hand-written Qwen3.5 MTP architecture
        adapter.py             native weight layout and shared-weight contract
```

Settings remain in the repository's root `configs/` directory and use
`${model}-${method}[-native].json`. The `native` suffix describes structure,
not an instruction to load weights. Only `architectures` selects a model.
`llama3_eagle.py` is now `models/eagle3.py`; the registered architecture name
`LlamaForCausalLMEagle3` is retained so existing checkpoints keep loading.

Old Python import paths have been removed. Import model classes from
`models/`, typed options from `config/`, and reusable components from
`models/layers/`. Update external imports to these canonical paths.
The general-purpose embedding, linear, and LM-head modules in
`specforge/layers/` retain their existing location. Ring's implementation lives
at `models/layers/attention/parallel/ring.py`; its old imports have been removed.

## Public JSON

Keep HF dimensions, vocabulary, activation, RoPE and window/layer fields at the
root. Four parallel objects describe the extensions:

```json
{
  "architectures": ["DSparkDraftModel"],
  "model_type": "qwen3",
  "hidden_size": 4096,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "attention_config": {"type": "gqa", "is_causal": false},
  "residual_config": {"type": "standard"},
  "ffn_config": {"type": "dense"},
  "spec_config": {
    "schema_version": 1,
    "algorithm": "dspark",
    "block_size": 7,
    "markov_rank": 256
  }
}
```

This excerpt illustrates placement; use a complete setting from `configs/`.
Algorithm field definitions, defaults and validation live in the corresponding
`config/*.py` class. DSpark, DFlash2, Domino and DSpine inherit DFlash options. A subclass
explicitly documents any differing meaning: DFlash's `block_size` includes the
anchor, while DSpark's counts proposals. Inheritance does not change that
established training/serving convention.

MLA's `kv_lora_rank`, `q_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`,
`v_head_dim` and `rope_interleave` belong in `attention_config`. GQA dimensions
continue to use HF root fields. SWA uses root `layer_types` and `sliding_window`.
For DFlash-family models, `is_causal: null` preserves the existing per-layout
causal defaults; a boolean explicitly overrides direction. EAGLE and the current
native MTP require causal attention.

Unknown extension fields, unsupported combinations, and conflicting public and
legacy aliases fail validation. Unrecognized legacy metadata is retained for
checkpoint compatibility. Validation runs before allocating the model's weights.

## Loading, saving and export

```python
from specforge.modeling.auto import AutoDraftModel
from specforge.modeling.draft.config import DraftConfigClass

config = DraftConfigClass.from_file("configs/qwen3-8b-dspark.json")
config = DraftConfigClass.finalize(config)
model = AutoDraftModel.from_config(config)
model.save_pretrained("draft-checkpoint")
restored = AutoDraftModel.from_pretrained("draft-checkpoint")
```

Parsing and finalization are separate because training providers apply run
settings between them. A parsed HF config retains legacy Python attributes
(`dflash_config`, `eagle_config`, etc.) used by existing training code.
`config.to_dict()` is that runtime representation; use
`DraftConfigClass.to_dict(config)` for the validated public setting.
`save_pretrained` writes the public format, including when a model was directly
constructed with a normal HF config. Legacy JSON checkpoints remain readable.
The serialization policy is derived from each algorithm's typed options, rather
than maintaining another per-algorithm list inside the loader.

The HF/SGLang exporters currently render legacy fields in `config.json` because
existing serving implementations consume those fields. They also write the
canonical setting to `specforge_config.json`. This translation belongs to the
export boundary and does not change model parameters. Both exporters share the
same conversion. Native MTP merge export retains the target model's config and
native tensor layout; the concrete adapter declares the draft's weight contract.

## Extending components

- **Attention:** add a full module and an explicit construction branch in
  `layers/attention/`. GQA and MLA currently share the dense context/query
  scaffold. DSA/QSA selection and cache logic must have their own module, not be
  squeezed into `_compute_qkv`. KDA may require a distinct state contract. The
  algorithm's query/context layout must be supported explicitly before enabling
  any new structure. EAGLE's different cache contract remains separate.
- **Execution:** Flex is a backend; Ring is a parallel execution strategy.
  Neither selects the projection structure. Check backend, mask, window and
  parallel compatibility rather than silently falling back to another structure.
- **Residual:** implement `ResidualStream.branch_input` and `combine`, then add
  construction and validation. GR/mHC also require explicit stream initialization,
  output collapse, dimensions and branch context where their structure needs it.
  The current additive path preserves existing normalization and weight names.
- **FFN:** `build_ffn` is the assembly boundary. Dense kernels are injectable.
  Future MoE variants must own routing/shared-expert semantics and explicitly
  integrate auxiliary outputs with training. Add a concrete `variant` and typed
  parameters when an actual implementation exists.
- **Native models:** register the hand-written model in `specialized/`, declare
  its config/options class, and put native checkpoint mapping in its adapter.
  Shared components must not import concrete generic or native model classes.

DSA, QSA, KDA, GR, mHC and MoE names are reserved and rejected as unimplemented;
there are no placeholder models that silently run dense attention or FFNs.
DSpine uses the same loading and serialization contract while retaining its own
injection/refinement settings. Qwen3.8 MTP and DeepSeek-V4.1 native draft
architectures are not implemented or registered. Existing Qwen3.5 MTP is migrated
as-is. Native checkpoint adapter extraction is covered by a separate PR.
