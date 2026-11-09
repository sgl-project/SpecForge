# Draft Model and Component Design

This document covers model base classes, configuration inheritance, component interfaces, and computational constraints.

The following describes the target design, not support for every combination in the current implementation. Specialized model names are illustrative; their actual structures must follow the corresponding checkpoints and native implementations.

## 1. Model Base Classes and Composition Boundaries

Models such as `DFlashDraftModel` inherit from `PreTrainedModel`, directly or through a project-wide base class built on it. They do not depend on model-family bases such as `Qwen3PreTrainedModel` or `LlamaPreTrainedModel`.

Configuration classes derive from `DraftModelConfig(PretrainedConfig)`. Each model's `config_class` references its corresponding `xxxDraftModelConfig`. Pass this configuration object to the HF model constructor; a plain immutable dataclass cannot replace it.

| Layer | Responsibility |
|---|---|
| HF base classes and project-wide base classes | Loading, saving, initialization lifecycle, and common framework capabilities |
| Algorithm model skeletons | Feature fusion, input/output interfaces, prediction layouts, specialized heads, and layer assembly |
| Attention, FFN, and residual components | Component computation and requirements for dimensions, state, and tensor layouts |

General-purpose architectures expose supported component combinations. Specialized architectures implement specific connections, fusion, and sharing relationships while reusing common components. Do not introduce a specialized class when a general-purpose implementation already represents the native structure accurately.

Each architecture declares which components are replaceable. Matching dimensions alone do not make components interchangeable: attention visibility, KDA state interfaces, and mHC residual-stream layouts must satisfy the model interface. Specialized architectures must also use the components declared by their configuration rather than silently substituting fixed implementations.

When replacing a model base class, explicitly take ownership of initialization, RoPE, defaults, gradient checkpointing, and model metadata. Common components must not accept model-family-specific configuration objects. Algorithm implementations own training objectives and serving-time verification scheduling; these responsibilities do not belong in the model base class.

## 2. Configuration Inheritance and Parameter Semantics

Configuration classes live in `specforge/config/draft/` and reuse fields only when their semantics match:

```text
PretrainedConfig
└── DraftModelConfig
    ├── Eagle3DraftModelConfig
    ├── MTPDraftModelConfig
    ├── PEagleDraftModelConfig
    └── DFlashDraftModelConfig
        ├── DFlash2DraftModelConfig
        ├── DominoDraftModelConfig
        ├── DSparkDraftModelConfig
        └── DSpineDraftModelConfig
```

- DFlash subclasses own their additional parameters. When a field retains its name but changes meaning, override its description, default, validation, and derived calculations.
- Do not assume native MTP uses the same input/output or supervision protocol as EAGLE3. Reuse algorithms only when their protocols are compatible.
- Configuration inheritance does not require the same model inheritance hierarchy or introduce nested parent configuration sections.
- Express attention, FFN, and residual choices through composition rather than creating a configuration subclass for every combination.
- Parameter documentation can be generated from Python configuration definitions, including subclass overrides of defaults and semantics.

Algorithm parameters are stored directly in `spec_config`. Fields are valid only for their corresponding algorithms; do not pass arbitrary values through an untyped `params` dictionary.

| Algorithm | Examples of specific parameters |
|---|---|
| EAGLE3 | Feature selection, draft vocabulary, etc. |
| DFlash | `block_size`, `mask_token_id`, `target_layer_ids` |
| DFlash2 | DFlash parameters, `selector_rank`, `selector_top_k`, convolution parameters |
| DSpark | Block and feature parameters, `markov_rank`, `markov_head_type`, confidence-head settings |
| Domino | Block and feature parameters, fusion and projection parameters |

### Block Parameters

Preserve each algorithm's conventions rather than forcing identically named parameters to have the same meaning:

| Algorithm example | Configuration value | New draft tokens | Verification window |
|---|---:|---:|---:|
| DFlash2 | `block_size = 16` | 15 | 16 |
| DSpark | `block_size = 7` | 7 | 8 |

Configuration parsing derives prediction length, query layout, and verification window according to the algorithm, avoiding redundant storage of derivable length fields. EAGLE inference expansion steps belong to serving settings, not model structure.

## 3. Example Structural Fields

The following DSpark JSON illustrates field organization, assuming that the source configuration or target has `model_type: qwen3`. It does not imply that DSpark inherits from Qwen3, nor is it a complete configuration for a real checkpoint. See the [usage guide](config_usage_graph-eng.md) for field sources and HF export rules for `auto_map`.

```json
{
  "architectures": [
    "DSparkDraftModel"
  ],
  "model_type": "qwen3",
  "hidden_size": 4096,
  "intermediate_size": 12288,
  "num_hidden_layers": 5,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "vocab_size": 129280,
  "hidden_act": "silu",
  "attention_bias": false,
  "attention_dropout": 0.0,
  "rms_norm_eps": 1e-06,
  "tie_word_embeddings": false,
  "max_position_embeddings": 1048576,
  "rope_parameters": {
    "rope_type": "yarn",
    "rope_theta": 10000,
    "factor": 16,
    "original_max_position_embeddings": 65536
  },
  "attention_config": {
    "type": "gqa",
    "is_causal": false
  },
  "residual_config": {
    "type": "standard"
  },
  "ffn_config": {
    "type": "dense"
  },
  "spec_config": {
    "schema_version": 1,
    "block_size": 7,
    "mask_token_id": 128799,
    "target_layer_ids": [
      1,
      11,
      21,
      31,
      41
    ],
    "markov_rank": 256,
    "markov_head_type": "vanilla",
    "enable_confidence_head": true,
    "confidence_head_with_markov": true
  }
}
```

Keep common head parameters, `intermediate_size`, normalization, and RoPE at the top level rather than duplicating them in component extensions. `schema_version` identifies the extension configuration version; it does not replace HF metadata.

## 4. Attention

Use the top-level `attention_config`. Its `type` selects a configuration branch containing only parameters specific to that structure.

| Type | Configuration principles |
|---|---|
| GQA | Use HF top-level fields for head counts and head dimensions; declare the type and required attention semantics in the extension |
| MLA | Place specific parameters such as low-rank dimensions and QK/V component dimensions in the attention branch |
| DSA | Preserve projection structure, indexer, and selection-policy settings; MLA projection configuration can be reused |
| QSA | Preserve the projection, selection, quantization, or compression components required by the specific variant |
| KDA | Use its own state, gating, and projection fields; do not require parameters that apply only to GQA/MLA |

For example, DSA can combine an MLA projection with an indexer. The following illustrates organization only; actual use requires all mandatory fields for each branch:

```json
{
  "attention_config": {
    "type": "dsa",
    "projection": {
      "type": "mla"
    },
    "indexer": {
      "type": "learned_topk"
    },
    "is_causal": true
  }
}
```

### Causality and SWA

Keep `is_causal` as an explicit attention parameter. For block drafts, define its scope: bidirectional attention within a block must not bypass context-isolation constraints during training.

For SWA, prefer existing HF top-level window and layer-type fields such as `sliding_window` and `layer_types`; do not duplicate equivalent settings in the attention extension. If a structure requires additional local-window parameters, its type must explicitly define their semantics.

Also prefer existing fields for mixed layer layouts. Add per-layer configuration only when the existing format cannot represent the structure accurately; do not introduce a generic layer-override language in advance.

### Compute Backends and Distributed Execution

Organize structure, compute backends, and distributed execution separately:

| Dimension | Code location, relative to `models/layers/attention/` | Configuration location |
|---|---|---|
| Structures such as GQA, MLA, DSA, QSA, and KDA | `gqa.py`, `mla.py`, etc. | Model `attention_config` |
| Eager, SDPA, FlashAttention, and FlexAttention backends | `backends/` | Runtime configuration |
| Ring Attention distributed execution | `parallel/ring.py` | Runtime configuration |

`backends/flex.py` translates attention inputs and semantics into FlexAttention calls, including masks and score modifications. It does not own learnable projections or selection algorithms.

`parallel/ring.py` adapts Ring execution, including communication, blockwise computation, and result merging. The project's distributed infrastructure creates and manages process groups and passes them into this module; individual attention layers do not create process groups.

`masks.py` contains reusable visibility descriptions and validation. Algorithm models provide algorithm-specific query/context layouts, attention components specify causal, window, or sparse-selection requirements, and backends convert these into their accepted representations. Selecting Flex, Flash, or Ring must not implicitly change attention semantics.

Validate compatibility among structure, backend, parallel execution, masks, and windows at runtime. Separate directories do not imply that every combination is supported: Ring's local compute interface may require additional intermediate results, so arbitrary backends cannot be assumed to work with it. DSA/QSA selection outputs require backend support; KDA need not use the same interface as softmax attention.

Backend and parallel settings may be saved with training runtime configuration for reproducibility. They do not determine the draft's structural identity or replace its exported structural description.

## 5. Residual Streams

Use the top-level `residual_config`.

Standard residual streams:

```json
{
  "residual_config": {
    "type": "standard"
  }
}
```

GR organization:

```json
{
  "residual_config": {
    "type": "gr"
  }
}
```

mHC organization:

```json
{
  "residual_config": {
    "type": "mhc",
    "num_streams": 4
  }
}
```

Place GR variants and gating parameters, as well as mHC mixer and constraint parameters, in their respective branches. These examples do not enumerate every type-specific field.

Common normalization parameters continue to use HF fields such as `rms_norm_eps`. Only normalization parameters unique to residual-stream modules and with distinct semantics belong in the residual extension.

Residual-stream configuration must participate in cross-module dimension and tensor-layout validation. If different layers are later allowed to switch residual-stream types, define explicit conversion rules; changing a type name alone is insufficient.

## 6. FFN / MoE

Use the top-level `ffn_config`.

Dense FFNs read common fields such as `intermediate_size` and `hidden_act` directly from the HF top level:

```json
{
  "ffn_config": {
    "type": "dense"
  }
}
```

For MoE, `type` identifies the FFN category and `variant` identifies the specific computational structure. The following illustrates field organization, not a complete configuration for a real model:

```json
{
  "ffn_config": {
    "type": "moe",
    "variant": "deepseek_v3",
    "num_experts": 64,
    "num_experts_per_token": 4,
    "expert_intermediate_size": 2048
  }
}
```

`variant` should identify a concrete structure, such as a particular version of routing and shared-expert rules, rather than a broad name like `qwen` or `deepseek`. Components from different model families can reuse a variant when their computational semantics match. Differences in weight names alone belong in adapters and do not warrant a new variant.

Add explicit fields or subconfigurations when routers, shared experts, or expert types need further distinction. Expert counts, top-k, and dimensions remain explicit parameters; `variant` must not implicitly supply an entire parameter set. Avoid expressing distinct structures through many interacting Boolean switches.

Specialized architectures select variants matching their native structures. General-purpose architectures can select those variants independently, without adopting attention or residual streams from the same source model.

`expert_intermediate_size` describes each expert's intermediate dimension and is distinct from a standard Dense FFN's `intermediate_size`. Handle synonymous MoE fields from existing configurations through compatibility parsing or export adapters; do not maintain multiple independent parameter sources.

Structural routing parameters belong to model configuration. Training-objective parameters such as auxiliary-loss weights remain in training configuration.

## 7. Code and Validation Constraints

See the [usage guide](config_usage_graph-eng.md) for the full directory tree. Model code follows these conventions:

- `specforge/modeling/draft/models/`: general-purpose algorithm model skeletons.
- `models/layers/`: common computation components, including attention `backends/` and `parallel/`.
- `models/specialized/<model>/model.py`: specialized connections and forward computation.
- `models/specialized/<model>/adapter.py`: specialized configuration and weight adaptation.

Common components and backends must not depend on algorithm models or specialized implementations. Configuration parsing additionally follows these constraints:

- Reject unknown or inapplicable extension fields; preserve valid HF top-level metadata.
- Normalize legacy aliases at the entry point rather than keeping two independently mutable sources; reject conflicting old and new values.
- `variant` identifies computational semantics and must not implicitly replace all numerical hyperparameters.
- Validate fixed native-structure constraints before construction; metadata added by HF must not change the structure.
- Reserve `DraftModelConfig` for the HF configuration base class; do not reuse the name for read-only options.

## 8. Implementation and Validation

Proceed incrementally: configuration classes → model base-class migration → common components → specialized architectures. Keep existing entry points until migration is complete. Map existing `dflash_config` and algorithm-specific top-level fields into the new structure at the compatibility entry point.

In addition to the loading and export validation described in the usage guide, focus model-refactor checks on:

1. Matching outputs before and after base-class replacement with identical weights and inputs, while preserving initialization and gradient checkpointing.
2. Consistent subclass defaults, validation, and derived calculations, especially differing `block_size` semantics.
3. Forward, backward, and parameter-update checks for every declared supported component combination.
4. Preserved mask and numerical semantics for supported attention-backend and Ring combinations; unsupported combinations fail before execution.
5. Specialized architectures using their declared components and rejecting configurations that violate fixed structural constraints before construction.
