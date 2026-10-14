# DSpark-KV with convolution and DPaCE

Use `examples/configs/online/disaggregated/external/qwen3-8b-dspark-kv-conv-dpace-online.yaml` with the Qwen3-8B KV convolution draft config. The grouped attention/MLP convolutions start as identity mappings. Models without convolution settings keep their existing layers.

Set `loss_type: dpace` to weight CE and L1 with detached hard-label continuation credit. The independent `is_causal` flag selects draft direction. Resume contracts include direction, convolution and DPaCE settings.

Online training requires SGLang 0.5.18 with the base capture patch followed by `scripts/apply_sglang_target_kv_patch.sh`, plus configured external capture/Mooncake endpoints. Liger is needed when `use_liger_kernel: true`. Update model/data paths before use. Omit `max_checkpoints` for unlimited retention. Training capture does not provide speculative serving support.

Run `python -m unittest tests.test_modeling.test_dspark_conv_dpace -v` for identity initialization, export/reload, independent DPaCE loss/gradient references and chunk/accumulation equivalence.

## D-PARD alternative

Use `examples/configs/online/disaggregated/external/qwen3-8b-dspark-kv-conv-dpard-online.yaml` for detached overlap-credit-weighted Renyi-half divergence instead of CE/L1. D-PARD and DPaCE are mutually exclusive. See [D-PARD](dspark-dpard.md) for the formula, valid-token normalization, resume contract and validation.
