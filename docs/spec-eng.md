# Draft Spec Design

Specs describe structure, computational semantics, and capability constraints. Parsing produces a single immutable `DraftConfig`; builders construct components from configuration nodes, and architecture models own assembly and execution order.

This page is a condensed reference for the configuration nodes. The [design document](design-eng.md) is normative; where the two differ, the design document applies and this page is corrected.

## 1. Configuration Nodes

```text
DraftConfig
├── attention_configs: name → AttentionSpec
├── mlp_configs:       name → FFNSpec
├── conv_configs:      name → ConvSpec
├── layer_configs:     name → LayerSpec
├── layer_types:       tuple[layer definition name, ...]
├── residual_config:   ResidualSpec
├── norm_config:       NormSpec
├── spec_config:       Architecture-specific protocol, fusion, and auxiliary module configuration
└── Model-level dimensions, RoPE, target_bindings, target_logit_transform, etc.
```

All nodes are frozen dataclasses, with immutable nested collections so that the complete configuration is hashable. Nodes own their fields, local validation, and pure derivations; they hold no weights, activations, or caches. Different structures have their own types and required fields. Unknown fields are rejected.

### AttentionSpec

`AttentionSpec` is a family of attention configuration types, distinguished by `type`. Different types need not share the same head or cache fields.

| Type | Main parameters |
|---|---|
| GQA / MHA | `num_attention_heads`, `num_key_value_heads`, `head_dim`, `qk_norm`, `attention_bias`, `output_gate`, `is_causal`, `sliding_window` (`null` for none); MHA requires equal query and KV head counts |
| MLA | `num_attention_heads`, `q_lora_rank` (`null` for none), `kv_lora_rank`, projection dimensions, rotation convention, and `is_causal` |
| DSA / QSA | Projection sub-nodes and the corresponding indexing, selection, quantization, or compression parameters |
| KDA | Its own head, state, gating, short-convolution, and projection parameters |

Every field that changes a node's parameters or computation is required, so strict copies and the SGLang converter can rely on the node alone. Model-level `rope_parameters` supplies the default RoPE, including partial rotation; an attention node may explicitly override or disable it. Causality and window size are specified separately. Bidirectional attention within a block must still preserve isolation between the context and different draft blocks. Capability contracts declare the implementation's input, output, cache, and state semantics.

### FFNSpec, ResidualSpec, NormSpec, and ConvSpec

| Spec | Configuration location | Responsibility |
|---|---|---|
| `FFNSpec` | `mlp_configs.<name>` | Dense: `intermediate_size` and `hidden_act`; MoE: expert count, top-k, per-expert width, shared experts, and routing parameters |
| `ResidualSpec` | Model-level `residual_config` | Residual-stream structure, gating, and mixing parameters for Standard, GR, and mHC |
| `NormSpec` | Model-level `norm_config` | Normalization used by decoder layers, the final norm, and component-internal norms: `type`, `eps`, and `weight_offset` (1 for `1 + w`) |
| `ConvSpec` | `conv_configs.<name>` | Sublayer convolution structure, such as `kernel_size` and `group_size` for `grouped_causal` |

`FFNSpec` is serialized through `mlp_configs` and the layer reference field `mlp`; there is no separate `ffn_configs`. Component parameters live directly in their nodes. Builders receive model-level parameters such as `hidden_size` and `norm_config` explicitly.

There is one `ResidualSpec` at model level. The residual-stream layout is consistent across layers, with expansion and reduction at model boundaries. Layers cannot independently switch residual types without explicit conversion rules.

DFlash2 references convolution through a layer node's `sublayer_conv`, wrapping both the attention and MLP sublayers. Convolution belongs inside the decoder layer; cross-node validation checks the relationship between the kernel and the block.

### LayerSpec

`LayerSpec` is a fixed component-composition schema containing name references only. It has no layer-level `type` field and requires no layer-type registration. The codec handles parsing, `DraftConfig` manages named definitions and their per-layer order, and validation checks references and compatibility.

- It contains only references to `attention`, `mlp`, and optional `sublayer_conv` definitions. It copies no component parameters and contains no independent residual configuration.
- Native drafts use the same components. Model origin is not a layer type, and the target model's whole-layer implementation is not called directly.
- `layer_types` selects a named `LayerSpec` for each layer. Its entries are layer-definition names, not implementation types or registry keys. `num_hidden_layers` is derived from its length.

Referencing the same layer definition means identical structure, not shared weights. Each instantiation creates an independent module. Draft modules never share parameters with each other, except the embedding and LM head when `tie_word_embeddings` is true; the only modules taken from the target are those listed in `target_bindings`.

`LayerSpec` describes neither training loops nor arbitrary execution graphs. Architecture models still organize DMTP's input fusion and DSpine's inter-layer injection; these do not belong in ordinary decoder-layer configuration.

## 2. Parsing and Ownership

The parsing flow is: **select the structural source → decode and expand the input → construct immutable nodes → validate across nodes → build the model**.

- When creating a new structure, training YAML supplies configuration through `draft`, either inline or as a configuration-file reference. A preset is an ordinary YAML file maintained by the project. It uses the same file input and requires neither a dedicated field nor name registration.
- The project maintains the bundled preset YAML files; users may modify or add configuration files. File names and directories do not participate in model-construction dispatch.
- Inline YAML retains homogeneous-layer shorthand: a single `attention`, `mlp`, and `num_hidden_layers`. The codec expands these into named components, layer definitions, and `layer_types`. Shorthand cannot be mixed with named definitions. Configuration files may use the canonical format or a legacy format supported by the legacy codec. Referencing a file loads structure only, not weights.
- Continued training and resumption read structure from the checkpoint or import adapter, usually without requiring the structure to be supplied again. An old checkpoint without configuration needs a supplementary legacy configuration; insufficient information is an error, not a reason to guess. If `draft` is additionally supplied for validation, its parsed structure must match the loaded configuration.
- `initialization.checkpoint` loads structure and weights and creates new training state. `resume` also restores optimizer and other training state. `initialization.weights_from` supplies only mapped weights for a new structure, which is still specified through YAML.
- Structural fields have no implicit defaults. Values depending on the target model or tokenizer are derived at parse time only through explicit rules, with the required sources supplied explicitly. Results are saved as concrete values. Import codecs convert source fields and semantics.
- Only the legacy codec handles aliases and historical default semantics in old formats. For example, it resolves the omitted causality setting of legacy full/sliding attention into explicit values during conversion. Algorithms validate constraints; they do not fill in or change component settings.
- Nodes perform local validation. A shared validation stage checks name references, unused definitions, dimensions, layer counts, residual streams, and algorithm capability requirements. Incompatible combinations fail before model parameters are constructed.
- `model.config` is the HF shell `SpecForgeDraftConfig`; `model.config.draft` is the authoritative configuration. Components receive only their own nodes and explicit dimensions, not a `LayerConfig` property view that can access the entire model configuration.

Purely derived quantities are computed through methods. Values such as `num_hidden_layers` that are written to JSON for compatibility are recomputed and checked on read. Configuration is immutable after construction; structural changes require a new configuration and a new model.

## 3. Registration and Capability Contracts

Architectures and extensible components register their configuration and computational implementations separately. `LayerSpec` uses a fixed schema and does not participate in type dispatch:

```text
config.draft:  (component kind, type) → component schema and capability declarations
modeling:      (component kind, type) → component builder

Parse configuration → validate nodes and cross-node constraints → look up component builders → assemble layers from LayerSpec
```

`config.draft` imports neither torch, transformers, nor model implementations. It holds no builders responsible for loading computational code. Architecture specs, model classes, training objectives, and format converters are registered by their respective packages.

Contracts describe inputs, outputs, tensor layouts, state, and supported computational capabilities. Algorithm names do not select behavior inside components.

| Caller requirement | Validation target |
|---|---|
| DFlash family requires K/V from injected target context | Whether attention declares and implements this capability |
| EAGLE3's first layer receives fused input with twice the hidden width | Layer input width |
| DSpine uses causal in-block attention and injection | Attention causality and injection's residual-stream layout requirements |
| DFlash2 sublayer convolution | Convolution parameters, layer composition, and block protocol |
| mHC multiple residual streams | Layout compatibility of all layers and inter-layer modules |

Different computational structures may use different interfaces. Stateful attention is not forced to emulate an ordinary KV cache. Backends, masks, caches, and parallel implementations must jointly satisfy the contract. Switching backends preserves semantics, and unsupported combinations are explicitly rejected.

Native presets use the same specs and builders. Missing computational capabilities require explicit component implementations and types, without falling back to target-family builders. No code derives nodes from a target's configuration: a native draft's structure is its preset, and its weights are copied by an ordinary `from_target` mapping.

External extensions register through the `specforge.extensions` entry point. The configuration's `extensions` declares loading prerequisites, contains no Python import paths, and is not part of `DraftConfig`.

## 4. Model Execution Boundaries

| Layer | Responsibility |
|---|---|
| Specs and codec | Structural semantics, validation, derivation, serialization, and migration |
| Builders and computational components | Construct modules from nodes; manage parameters and computational state |
| Backends and parallel implementations | Adapt attention computation, perform communication, and merge results |
| `DraftBackbone` | Shared layers, RoPE, residual-stream boundaries, and final normalization; sequential and per-layer execution |
| Architecture models | Input fusion, execution order, auxiliary modules, and explicitly typed outputs |
| Training and inference workflows | Feature provision, sample preparation, supervision, and scheduling; candidate sampling, verification, and cache updates |

DFlash, DFlash2, and DMTP use the backbone's sequential forward. DMTP's trainable MTP-feature fusion belongs to the model forward. Frozen MTP features are supplied externally, and training and inference follow the same token-alignment rules.

DSpine owns its “decoder layer → injection” loop through the backbone's `begin`, `run_next_layer`, and `finalize`, the same calls that make up the sequential forward. One injection module manages per-layer readout/write parameters and parameters shared across layers. Outputs preserve the final layer's pre-injection state and each layer's post-injection features with their gradients. Refinement reuses the final injection and final normalization.

The backbone receives no algorithm-specific hooks or training schedules. Training generates `replace_mask` and passes it to the model. The protocol spec derives the number of model slots, draft prediction positions, candidates supplied by other predictors, and the verification window.

## 5. Configuration and Serialization

Reference a bundled preset YAML as follows; the path is illustrative:

```yaml
draft: specforge/config/draft/presets/native/qwen3_5_mtp.yaml
```

User YAML or JSON configurations use the same file-reference mechanism, or can be supplied inline under `draft`.

With named layer definitions, `num_hidden_layers` is derived from `layer_types`. Users need not provide format metadata such as `schema_version` or `auto_map`.

The canonical format uses `model_type: specforge_draft`, a single architecture in `architectures`, and `schema_version`. Component nodes are saved in named maps; ordinary decoder layers only reference those definitions. The following canonical configuration fragment omits model-level dimensions, `spec_config`, and other fields. It is not a complete training YAML:

```yaml
schema_version: 1
attention_configs:
  local:
    type: gqa
    num_attention_heads: 32
    num_key_value_heads: 8
    head_dim: 128
    qk_norm: true
    attention_bias: false
    output_gate: false
    is_causal: true
    sliding_window: 2048
  full:
    type: gqa
    num_attention_heads: 32
    num_key_value_heads: 8
    head_dim: 128
    qk_norm: true
    attention_bias: false
    output_gate: false
    is_causal: false
    sliding_window: null
mlp_configs:
  narrow: {type: dense, intermediate_size: 11008, hidden_act: silu}
  wide: {type: dense, intermediate_size: 14336, hidden_act: silu}
layer_configs:
  local_dense: {attention: local, mlp: narrow}
  full_dense: {attention: full, mlp: wide}
layer_types: [local_dense, full_dense]
residual_config: {type: standard}
```

Saving writes the complete effective configuration, including the derived `num_hidden_layers: 2`. Loading restores the same `DraftConfig`. A checkpoint does not depend on the original configuration file's path or subsequent changes.

The HF shell stores HF metadata; the checkpoint stores format and weight-layout metadata. Neither belongs to `DraftConfig`. Training configuration and loading plans own copies from target weights (`initialization.from_target`): copies are bit-preserving, using only explicit renames and the layout operations that components with fused parameters declare, every element of every mapped draft parameter and every target tensor must be used, and quantized tensors and narrowing casts are rejected. Copies are draft weights and are saved with the draft; decoder layers are never bound. `target_bindings` lists which of `embed_tokens` and `lm_head` come from the target; they are frozen and not saved with the draft. A bound `lm_head` reads `target_logit_transform`, a target-derived value computed at parse time, so model code never reads the target configuration. The draft is still built from its declared components.

HF export uses the canonical configuration. SGLang never reads canonical configurations: the SGLang converter writes SGLang's own draft format, with every field that affects computation written explicitly. SGLang directories are one-way serving artifacts and are never imported. Format representability and engine execution support are validated separately. Unsupported structures and binding sets fail export. Export does not modify the model configuration.

## 6. Code Organization and Validation

| Location | Contents |
|---|---|
| `config/draft/attention.py` | `AttentionSpec` type family |
| `config/draft/mlp.py` | `FFNSpec` type family |
| `config/draft/residual.py`, `norm.py`, `conv.py`, `layer.py` | `ResidualSpec`, `NormSpec`, `ConvSpec`, `LayerSpec` |
| `config/draft/spec/` | Architecture protocols and auxiliary-module configuration; algorithms reuse these through composition |
| `config/draft/codec/`, `validate.py`, `registry.py` | Encoding and decoding, cross-node validation, and configuration-type registration |
| `config/draft/presets/`, `presets/native/` | Project-maintained configuration YAML files; the latter describes native draft structures |
| `modeling/draft/layers/`, `heads/`, `backbone.py`, `models/` | Components, auxiliary modules, shared backbone, and architecture models |
| `checkpoint/formats/` | External draft adapters and the SGLang converter |

Adding a component requires a node definition, a computational implementation, separate schema and builder registration, a checkpoint layout only if it fuses parameters, and capability-combination validation. A new parameter combination using existing types needs only a YAML file, without registering a configuration name or a new model.

Validation covers immutability and configuration round-trips, missing or conflicting fields, invalid references and capabilities, forward/backward and state behavior, agreement between sequential and per-layer execution, gradients under checkpointing, weight sharing, and export rejection. Refactoring existing components preserves parameter naming, initialization, and numerical behavior.
