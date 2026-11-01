# Draft models

Training settings live under `model.draft_model_config` in a recipe. Configuration
code is in `specforge/config/draft/`; this package contains model implementations.

```text
config/draft/
  __init__.py              public config API
  auto.py                  embedded YAML parsing and validation
  model.py                 immutable DraftModelConfig and reference-only layer view
  base.py                  shared config base and scalar/dimension validation
  registry.py              architecture-to-config registration
  algorithms/              generic EAGLE/DFlash-family parameters and constraints
  native/
    qwen3_5_mtp.py          native Qwen3.5 MTP settings and alignment checks
    rope.py                native RoPE compatibility checks
  layers/
    specs.py               AttentionSpec / FFNSpec / ResidualSpec / ConvSpec / LayerSpec
    decoder.py             per-layer resolution and legacy structural conversion
    registry.py            component specs and builders keyed by execution contract
  serialization/
    format.py              versioned public YAML/JSON format
    hf.py                  temporary HF save adapter
    legacy.py              legacy attention-field compatibility
    contract.py            structural compatibility comparison
    checkpoint.py          saved metadata used only for compatibility checks

modeling/draft/
  registry.py              model architecture registry
  serialization.py         save/load without mutating runtime settings
  models/
    eagle3.py, peagle.py    EAGLE assemblies and cross-layer wiring
    dflash.py, dflash2.py,
    domino.py, dspark.py,
    dspine.py               DFlash-family assemblies and prediction heads
    mtp.py                  native weight-loading helpers
    specialized/
      qwen3_5_mtp/model.py  handwritten native Qwen MTP architecture
    layers/
      attention/           GQA/MLA structure and context/cache contracts
        backends/          eager, SDPA, Flash and Flex execution
        parallel/ring.py   Ring execution strategy
      ffn/                 token-wise feed-forward components
      residual/            branch-input and stream-combination components
      convolution/         DFlash2 grouped convolution and Triton kernels
      parallel/            tensor-parallel embedding, linear and LM head primitives
      normalization.py     shared normalization layers
      kernels.py           kernel selection/injection
```

A component registers its frozen spec class and builders for the contracts it
implements. Algorithms declare the contract and structural constraints they
need, rather than allowlisting attention names. Unknown components are rejected.
Sparse attention, stateful attention, MoE and multi-stream residuals need real
implementations before they can be registered; this layout does not provide
placeholder implementations.

Every model uses an immutable per-layer tuple. Shared-layer syntax is only an
input shorthand. Layer adapters reference the model config and their layer
without storing a second set of settings. Create a new model to change structure.

Weight loading requires `config=AutoDraftConfig.from_yaml("run.yaml")`.
Neither HF repository IDs nor checkpoint directories can supply initialization
settings. Both automatic and concrete model loaders validate saved metadata
against that config before loading weights. Exporters take `--config run.yaml`.

Checkpoint saves use versioned canonical config. Serving export creates legacy
HF/SGLang fields only when that format can represent the complete layout.
Mixed full/SWA is supported with a shared window; incompatible heterogeneous
structures fail before writing weights. The canonical config is retained as
`specforge_config.json` in a serving export.

Training recipes in `examples/configs/` own their draft settings under
`model.draft_model_config`; there is no separate JSON preset catalog.
The architecture selects a generic or handwritten native model independently
of random initialization, checkpoint loading, or native weight loading. All
weight loading validates against the embedded settings.
