# Draft components

Algorithm models own input preparation, token visibility, target-feature fusion
and prediction heads. This package owns reusable layer computations.

- `attention/base.py`: `AttentionContext`, `AttentionOutput[State]` and the
  `Attention[State]` contract. `DenseAttention` is a separate scaffold for
  explicit Q/K/V and Hugging Face caches.
- `attention/gqa.py`, `attention/mla.py`: projections and dense attention
  implementations. Projection mixins contain no backend dispatch.
- `attention/selection.py`: token/block selections and local SWA branches,
  with explicit KV source names and logical index coordinates.
- `attention/backends/sparse.py`: typed execution protocol for future sparse
  adapters, with implementation-specific inputs and per-head output validity.
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

## Extension boundary

`qsa.py` and `dsa.py` implement opt-in stateless training attention. DSA follows
the DeepSeek-V3.2 lightning indexer over expanded MLA. QSA follows Qwen's
compressed indexer over GQA. Both use floating point scoring, without inference
FP8/FP4 quantization. They do not implement DeepSeek-V4.1 compressed latent
attention, cross-layer IndexShare, paged caches or official checkpoint conversion.
MoE is still outside this change.

SWA is independent of the projection structure: current GQA/expanded MLA use
`AttentionConfig(sliding_window=N, is_causal=True)` for a causal window of N
logical positions including the query itself. Without causality the condition
is `abs(query_position - key_position) < N`. This is a visibility restriction,
not a promise of bounded cache storage or sparse-kernel performance.

Future sparse adapters consume `SparseSelection` without constructing a dense
mask. `IndexedSelection` uses `[B, Hs, Q, K]` source-local token/block indices
and `[B, Hs, Q]` valid lengths; `SlidingWindowSelection` describes a local
branch without enumerating its keys. For example:

```python
selection = SparseSelection(
    indexed=(IndexedSelection(indices, lengths, source="main", block_size=8),),
    windows=(SlidingWindowSelection(size=128, source="main"),),
)
```

This describes selected blocks **union** local keys, intersected with global
causality, algorithm masks and any global window. Leave the global
`AttentionConfig.sliding_window` unset to retain distant selected blocks.
Overlaps within a source count once; different named sources (e.g. raw local
KV and compressed KV) remain distinct entries in one attention normalization.
Each adapter defines its typed input payload, source mappings and supported
combinations. QSA retrieval blocks must first map to attention KV; compressed
indexer keys are not implicitly attention values. These are interface contracts,
executable in the explicit reference backend. The gathered backend currently
accepts one source and indexed token/block selections; global SWA restrictions
are applied before selection. It rejects unsupported multi-source/window-branch
plans rather than silently executing reference attention.

A sparse implementation should implement `Attention` and reuse projections as
appropriate; it must not inherit `DenseAttention.forward`. Its state can differ
from the dense cache. `AttentionOutput.auxiliary_losses` carries named,
unweighted scalar tensors without detaching them. `DecoderOutput.attention`
retains them. DFlash-family models expose `(hidden_states, auxiliary_loss)` when called with
`return_auxiliary_losses=True`. Their online training wrappers consume this loss;
ordinary dense model forwards keep returning tensors. A trainable indexer rejects
calls that would silently discard its auxiliary loss.
The attention/decoder never performs backward or optimizer stepping. Specialized
outputs can subclass `AttentionOutput` to expose additional typed indexer data
when needed. Router objectives likewise need training integration with MoE.
Ring execution is kept
in the existing EAGLE path; no new Ring/MLA combination is claimed.


## Training QSA and DSA

Set `dflash_config.attention_mode` to `dsa` or `qsa`. These structures work with
DFlash, DFlash2, Domino and DSpark, including their target-context prefix and
algorithm masks. EAGLE3's specialized input path has not been adapted.

Choose the backend explicitly in the draft Hugging Face config:

```json
"sparse_attention": {
  "backend": "gathered",
  "indexer_mode": "joint",
  "loss_weight": 1.0,
  "query_chunk_size": 32
}
```

DSA additionally needs the existing MLA dimensions and positive `q_lora_rank`,
plus `index_n_heads`, `index_head_dim` (at least `qk_rope_head_dim`) and
`index_topk`. The indexer uses LayerNorm on keys, weighted ReLU scores and
non-interleaved indexer RoPE independently of the MLA RoPE convention.

QSA needs `indexer_n_heads`, `indexer_kv_heads=1`, `indexer_head_dim` (covering
RoPE), `indexer_compress_ratio>=2` and `indexer_budget` divisible by that ratio.
The budget counts tokens; the indexer chooses budget/ratio complete blocks.
Raw index keys are mean-pooled in FP32, Gemma-normalized, and rotated at the
first position of each block. Attention still reads original GQA K/V.
For contiguous causal inputs, visible unfinished-block tokens are included
outside the budget. With packed/non-contiguous masks, visible fragments that
cannot be safely pooled are included directly, so the actual token count can
exceed budget + ratio - 1. This prevents retrieval from exposing masked content.

Indexer modes:

- `frozen`: do not train the indexer; use a previously trained indexer checkpoint.
- `warmup`: dense attention while training only the indexer. Model construction
  freezes the rest of the draft, and loading reapplies that policy. The online
  objective uses only weighted indexer KL. Switch to joint training by constructing
  a new model with the new config and loading weights, rather than mutating mode
  on a live optimizer. Resume contracts reject accidental mode changes.
- `joint`: train the main model using its existing objective and the indexer
  using KL. Indexer inputs and teacher distributions are detached. DSA uses the
  selected support in joint mode and all visible keys during warmup, following
  [the V3.2 report, section 2.1](https://arxiv.org/html/2512.02556v1).
  QSA uses teacher attention mass aggregated per complete visible block; this is
  a SpecForge distillation objective, not a published Qwen training recipe.

Auxiliary KL averages valid queries per layer and then layers. Padded proposal
blocks are excluded. `loss_weight` is applied by the online algorithm wrapper.
DFlash/Domino incorporate it into their existing numerator/denominator contract;
DSpark adds it to its existing directly normalized loss. Backward, accumulation
and optimizer stepping remain owned by the controller. Gradient checkpointing
uses the non-reentrant path and preserves auxiliary gradients.

`reference` explicitly materializes dense scores and is for small numerical
checks. `gathered` reads selected KV in query chunks with PyTorch autograd. It
avoids a full attention score matrix in the value computation, but is not a
fused sparse kernel: indexer scores/visibility still grow with sequence length,
and teacher distributions are computed in chunks. Shape validation, selection
compaction and dynamic tail sizes can synchronize the device. Neither backend
is advertised as production long-context performance or CUDA-graph compatible.
Dropout, per-head visibility differences and cache mutation are rejected.

QSA score/compression references: [SGLang explanation](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next/)
and the previously pinned SGLang revision
`635430d07378e2b12381353781a23b0704969909`. The existing GQA projection/normalization
is reused; this is not a full Qwen model or output-gating conversion.


### Validation snapshot

On NVIDIA H20 / PyTorch 2.13.0+cu130, a small BF16 component probe included
projection, indexing, KL and backward (B=1, Q=128, S=1024, hidden=128, budget=32).
Five measured iterations after three warm-ups gave:

| Structure | Backend | Forward + backward | Extra peak allocation |
| --- | --- | --- | --- |
| DSA | reference | 7.68 ms | 20.24 MiB |
| DSA | gathered | 10.90 ms | 15.29 MiB |
| QSA | reference | 7.82 ms | 16.94 MiB |
| QSA | gathered | 11.72 ms | 12.51 MiB |

This is a component smoke benchmark, not model throughput or long-context
scaling evidence. Gathered execution saved memory in this case but was slower;
fused kernel work and representative workload measurements are still needed.
