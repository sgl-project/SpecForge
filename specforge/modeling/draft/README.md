# Draft models

Training settings live under `model.draft_model_config` in a recipe. Configuration
code is in `specforge/config/draft/`; this package contains model implementations.

```text
config/draft/
  model.py                 immutable DraftModelConfig and reference-only layer view
  layers.py                AttentionSpec / FFNSpec / ResidualSpec / ConvSpec / LayerSpec
  component_registry.py    spec types and builders keyed by execution contract
  layer_decoder.py         per-layer resolution and legacy structural conversion
  format.py                versioned public YAML/JSON format
  hf.py                    temporary HF save adapter
  eagle3.py, peagle.py      EAGLE-family algorithm parameters and constraints
  dflash.py, dflash2.py     DFlash-family algorithm parameters and constraints
  domino.py, dspark.py,
  dspine.py, mtp.py         remaining algorithm/native settings

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

Checkpoint saves use versioned canonical config. Serving export creates legacy
HF/SGLang fields only when that format can represent the complete layout.
Mixed full/SWA is supported with a shared window; incompatible heterogeneous
structures fail before writing weights. The canonical config is retained as
`specforge_config.json` in a serving export.

Files in `configs/` are reusable settings, named `model-method[-native].json`.
The `-native` suffix identifies a handwritten native architecture, not a weight
initialization policy. Training embeds the setting in YAML and separately chooses
random initialization, checkpoint loading, or native weight loading. All weight
loading validates against the embedded settings.
