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
- `ffn/dense.py`: gated FFN, including the original EAGLE tensor-parallel path.
- `config.py`, `factory.py`: typed construction options and explicit factories.
  `LayerKernels` injects the existing native or Liger norm/FFN implementations.
- `decoder.py`: ordinary pre-norm attention/FFN composition. Its result retains
  attention state; it does not require attention state to be a KV cache.
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
