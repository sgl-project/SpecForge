# Draft components

Algorithm models own input preparation, token visibility, target-feature fusion
and prediction heads. This package owns reusable layer computations.

- `attention/base.py`: `AttentionContext`, `AttentionOutput[State]` and the
  `Attention[State]` contract. `DenseAttention` is a separate scaffold for
  explicit Q/K/V and Hugging Face caches.
- `attention/gqa.py`, `attention/mla.py`: projections and dense attention
  implementations. Projection mixins contain no backend dispatch.
- `attention/backends/`: eager, SDPA, FlashAttention and FlexAttention execution;
  the existing FlexAttention compilation/backend adapters live here.
- `attention/rotary.py`: full and partial rotary position encoding.
- `residual/base.py`: explicit residual stream state, separate from attention cache.
- `residual/standard.py`: pre-norm read and additive write. Norms remain
  registered on their decoder so existing checkpoint keys stay unchanged.
- `residual/gr.py`: per-branch Gemma RMSNorm, low-rank gated read and gated write.
- `residual/mhc.py`: single-pass mHC, FP32 coefficient computation and Sinkhorn.
- `ffn/dense.py`: gated FFN, including the original EAGLE tensor-parallel path.
- `config.py`, `factory.py`: typed construction options and explicit factories.
  `LayerKernels` injects the existing native or Liger norm/FFN implementations.
- `decoder.py`: ordinary attention/FFN composition. Its result retains attention
  state and accepts separate attention/FFN residual modules; it does not require
  attention state to be a KV cache.
- `linear.py`, `embedding.py`, `lm_head.py`, `ring/`: relocated parallel layers.

DFlash, Domino and DSpark use the common decoder. DFlash2 retains its convolution
placement, and EAGLE3 retains its two-input attention layer; both reuse the
applicable components. Algorithm entry points still return tensors as before.

## Attention inputs

```python
attention = build_attention(config, layer_idx, kernels, AttentionConfig(kind="mla"))
result = attention(
    hidden_states,
    AttentionContext(
        position_embeddings=(cos, sin),
        attention_mask=mask,
        key_value_prefix=prefix,
    ),
    state=cache,
)
```

For the current GQA/MLA implementations, new keys are ordered as prefix followed
by `hidden_states`. RoPE describes these new keys; queries take the trailing
positions. Omit the prefix for ordinary self attention. Cache state contains
already rotated keys.

Masks and causality reach the backends exactly as before this package existed:
an explicit mask from the algorithm is used as supplied, and
`AttentionConfig.is_causal` / `sliding_window` are exposed as
`module.is_causal` / `module.sliding_window` for mask-free backends
(FlashAttention, and SDPA without a mask). Eager converts boolean allow-masks to
additive masks and zeroes queries with no visible keys, including after the
output-projection bias. `output_attentions=True` requests weights from backends
that provide them; the default does not retain an attention matrix in the
output. MLA currently expands K/V; it is not a latent-cache inference
implementation.

The internal DFlash attention classes retain their construction names, but now
use this context/result interface. Default algorithm model forwards, checkpoint
parameter names and parameter ordering are preserved;
`tests/test_modeling/test_layers/test_refactor_parity.py` checks this bit for bit
against the pre-refactor implementation. Production DFlash-family layers use
`build_attention`; compatibility attention constructors are not a second
production dispatch table.

## Residual configuration

DFlash, DFlash2, Domino and DSpark accept a `draft_residual` object in their
Hugging Face draft config. Omission retains the standard residual and checkpoint
layout. The new variants are opt-in architectures:

```json
{"draft_residual": {"kind": "gr", "num_streams": 4, "gate_rank": 320}}
```

```json
{"draft_residual": {"kind": "mhc", "num_streams": 4,
                    "sinkhorn_iters": 20, "sinkhorn_eps": 1e-6}}
```

Both expand the algorithm's initial hidden vector into parallel streams.
Attention and FFN receive single-stream inputs and have independent residual
parameters. DFlash2's convolutions remain between residual read and write.
GR normalizes each stream before its gated read; its learned final readout
replaces the ordinary model norm. mHC reads the incoming `pre_mix`, computes
coefficients for its write and the **next** read, and carries that next mix
through every sublayer. Its initial mix selects stream zero; the last mix
collapses the streams before the ordinary final norm.

No forward state is stored on modules. Hyper-connection checkpointing defaults
to `use_reentrant=False` because the stream state is nested and differentiable;
explicit reentrant checkpointing is rejected. Save/load includes the residual
configuration and weights. Resume contracts record non-parameter semantics,
including Sinkhorn iterations and epsilon.

These are differentiable PyTorch implementations of the following residual
math, not whole-model imports or inference-kernel ports:

- Qwen3.8-Flash-Next GR: [SGLang GatedResidual](https://github.com/sgl-project/sglang/blob/635430d07378e2b12381353781a23b0704969909/python/sglang/srt/layers/hyperconnection.py),
  with per-branch normalization as selected by `qwen4_exp.py`.
- DeepSeek-V4.1-Flash single-pass mHC: [Block](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/model.py)
  and [hc_split_sinkhorn](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/kernel.py).

mHC's coefficient projection, normalization, Sinkhorn and residual mixing use
FP32 computation; streams return to the update dtype. Fresh mHC parameters use
small dynamic scales and identity-biased residual routing; the upstream inference
code loads these parameters and does not define training initialization.
GR linear weights use the draft model's usual initialization and its Gemma norm
weights start at zero. No claim of bitwise equality with upstream fused kernels
or performance parity is made. Official full-model/MTP checkpoint loading,
target-feature fusion and vendor-specific prediction heads are outside this
component adaptation. New residual checkpoints require a serving implementation
that understands their config and parameter layout.
