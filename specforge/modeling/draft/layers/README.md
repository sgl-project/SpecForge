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
positions. Omit the prefix for ordinary self attention. Use an explicit mask
for algorithm-specific visibility. `is_causal=True` intersects that mask with
causal visibility; `is_causal=False` disables only the causal restriction. A
configured sliding window intersects both. Without explicit positions, queries
align to the **trailing** key positions, including prefix and cached keys.
For packed/non-contiguous layouts, supply `[batch, sequence]` `query_positions`
and `key_positions`; key positions must cover the entire cache as well.
DFlash supplies the actual anchor-based positions for packed training blocks.
Cache state contains already rotated keys.

When an algorithm's mask already applies the layer's causal and window rules
(DFlash builds one mask per layer type that way), it sets
`mask_encodes_visibility=True` and dense backends use the mask as is, instead of
rebuilding an identical mask (a Flex `create_block_mask`, or a dense
`[batch, 1, query, key]` tensor) in every sliding or causal layer.

Eager/SDPA accept four-dimensional boolean allow-masks or additive masks
(`-inf`/`finfo.min` excludes a key). Flex accepts `BlockMask` and composes mask
functions without constructing a dense attention matrix. FlashAttention uses
native right-aligned causality/windows and optional two-dimensional padding
masks; explicit visibility coordinates or arbitrary masks require another
backend. A query with no visible keys contributes zero attention output,
including after output-projection bias. Individual empty heads are zeroed
before projection, while the residual stream remains intact. Flex already
returns zeros for empty rows, so it requests the log-sum-exp needed to detect
them only when `o_proj` has a bias.
`output_attentions=True` requests weights from backends that provide them; the
default does not retain an attention matrix in the output.
MLA currently expands K/V; it is not a latent-cache inference implementation.

The internal DFlash attention classes retain their construction names, but now
use this context/result interface. Default algorithm model forwards, checkpoint
parameter names and parameter ordering are preserved;
`tests/test_modeling/test_layers/test_refactor_parity.py` checks this bit for bit
against the pre-refactor implementation, with one intended difference: SDPA
now zeroes queries with no visible key after `o_proj`, as eager always did.
Production DFlash-family layers use
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
