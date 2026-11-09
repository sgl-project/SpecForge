# Customize a training run

Customization starts from a typed YAML config. Pick the closest checked-in
file under
[`examples/configs`](https://github.com/sgl-project/SpecForge/tree/main/examples/configs),
change model and data paths, and keep the same training entry point:

```bash
specforge train --config ./my-run.yaml
```

For one-off changes, use dotted overrides rather than adding another launcher:

```bash
specforge train \
  --config examples/configs/online/disaggregated/external/qwen3-8b-eagle3.yaml \
  model.target_model_path=/models/my-target \
  data.train_data_path=/datasets/my-training-data.jsonl \
  training.learning_rate=5e-5
```

The config is strict: unknown fields, invalid strategy/topology combinations,
and unsupported attention backends are errors. See
[`specforge/config/schema.py`](https://github.com/sgl-project/SpecForge/blob/main/specforge/config/schema.py) and the
[training guide](../basic_usage/training.md) for the accepted fields.

## Chat templates

Register a text chat template in `TEMPLATE_REGISTRY` in
`specforge/data/template.py`, then reference its name with
`data.chat_template`:

```python
TEMPLATE_REGISTRY.register(
    name="your-template-name",
    template=ChatTemplate(
        assistant_header="...",
        user_header="...",
        system_prompt="...",
        end_of_turn_token="...",
    ),
)
```

```yaml
data:
  train_data_path: /datasets/my-training-data.jsonl
  chat_template: your-template-name
  max_length: 4096
```

The server capture runtime accepts text input only. VLM training, including
`qwen2_5_vl`, is not supported. Attention backends are a closed,
strategy-specific set:

| Strategy | Accepted `training.attention_backend` values |
| --- | --- |
| EAGLE3 | `sdpa`, `flex_attention`, `fa`, offline `usp` |
| P-EAGLE | `flex_attention` |
| DFlash, DFlash2, Domino, DSpark | `eager`, `sdpa`, `flex_attention` |

## Target models

For a Hugging Face-compatible target, normally only these model fields need to
change:

```yaml
model:
  target_model_path: organization/model-name
  target_backend: sglang
  trust_remote_code: false
  embedding_key: model.embed_tokens.weight
  lm_head_key: lm_head.weight
```

Every online run uses `model.target_backend: sglang`. Add target-model support
to the SGLang capture server instead of adding an HF/custom target loader to
the trainer. Target TP/EP and model-specific inference stay on that server;
the SpecForge consumer receives only the algorithm's versioned feature schema.
Offline feature training performs no target inference.

EAGLE3 offline sequence parallelism is selected with
`training.attention_backend: usp` plus `training.sp_ulysses_size` and
`training.sp_ring_size`. Evaluation, compact-teacher projection, and experiment
tracking are also config features rather than custom launchers; see the
[training guide](../basic_usage/training.md) for their validated combinations.

## Draft components and architectures

Copy a YAML file from `specforge/config/draft/presets/` and pass its path as
`draft` in the run config. Named `attention_configs` select GQA, MHA or MLA;
`mlp_configs` define FFNs; `layer_configs` composes these nodes and `layer_types`
selects one named layer per position. Attention direction and sliding windows are
explicit on each attention node. Changing components does not require a new
training launcher.

The canonical codec parses complete configurations into frozen nodes before model
construction. Architecture schemas register in `config.draft`; model classes use
`register_draft_model` in `modeling.draft.models`. Component schemas and builders
register separately. `build_draft_model` dispatches on the single `architectures`
entry and constructs the declared components without reading files.

An objective registers independently in `algorithms.objectives`: it declares the
architectures, inputs, outputs and trainable modules it supports. The model owns
learned execution, fusion and heads; the objective owns supervision and loss.
`training.objective_options` contains training-only settings. See
[draft training](../../draft_training.md) for loading and optimization options.

Export uses `specforge export --format sglang`. The converter checks the complete
structure against supported serving implementations and rejects incompatible
component combinations.

## Fused token losses

Import losses from `specforge.core.fused_loss`. Unprefixed functions are eager.

| API pair (eager / fused) | Label input | Per-token objective |
| --- | --- | --- |
| `ce_loss` / `fused_ce_loss` | Teacher logits `[B, T, V]` | CE against the teacher argmax |
| `tv_loss` / `fused_tv_loss` | Teacher logits `[B, T, V]` | `1 - sum(min(p, q))` |
| `ce_l1_loss` / `fused_ce_l1_loss` | Explicit token IDs `[...]`; optional teacher logits `[..., V]` | CE and optional `sum(abs(p - q))`, returned separately |

Here `p = softmax(draft_logits)` and `q = softmax(teacher_logits)`.
The joint CE/L1 API uses the supplied token IDs even when they differ from the
teacher argmax. For normalized distributions, `TV = L1 / 2`. Its backward shares
draft normalization and combines both loss gradients before casting to the
input dtype. This is useful when CE and teacher L1/TV train the same logits.

```python
from specforge.core.fused_loss import fused_ce_l1_loss

logits = lm_head(hidden)                    # [..., V], BF16/FP16/FP32 on CUDA
terms = fused_ce_l1_loss(
    logits,
    target_ids,                             # [...], int64, values in [0, V)
    teacher_logits=teacher_logits,          # optional [..., V], not probabilities
)
per_token_loss = 0.1 * terms.cross_entropy + 0.9 * terms.teacher_l1
loss = (per_token_loss * loss_mask).sum() / loss_mask.sum().clamp_min(1)
loss.backward()
```

The returned `CEL1Loss` contains unreduced FP32 losses. Without teacher logits,
`teacher_l1` is `None`; use `terms.cross_entropy` alone. Both inputs use the same
device and vocabulary. There is no `ignore_index`: masked positions still need
valid token IDs. Teacher logits never receive gradients. Masks, coefficients
and reduction belong to the caller; use `0.5 * terms.teacher_l1` for TV.

`top_k=K` additionally returns differentiable `topk_values` and integer
`topk_ids` with shape `[..., K]`, for DFlash2's selector. `argmax_ids` and the
optional draft/teacher top-1 probabilities are detached diagnostics. The
fused implementation requires CUDA and Triton; use `ce_l1_loss` on CPU.
Importing the public module does not require Triton. The eager reference promotes low-precision
CE to FP32 and preserves explicitly supplied FP64 inputs for numerical checks.

`training.objective_options.fused_logits_loss` is a boolean, defaulting to `true` for all supported
models, including DFlash2 and Domino. Set `false` to use eager losses.
Unsupported devices and dtypes fall back to eager. Domino uses its specialized
CE kernel under the same flag.

`training.objective_options.objective_chunk_blocks` sets the chunk size, and
`training.objective_options.objective_checkpointing` (default `true`) independently controls
recomputation of each chunk's projection and objective. A zero chunk size treats
the whole objective as one chunk; checkpointing still follows its own flag.
Without checkpointing, all chunks can retain
logits until backward: smaller chunks do not bound total saved-logits memory.
DFlash always skips its zero-weight anchor projection in multi-slot objectives;
DSpark supervises every slot. Detailed diagnostics retain the full blocks.

The selector consumes top-k candidates and hidden states, computes its own CE,
and joins the logits backward only through candidate values. Its direct hidden
state gradient is separate; `training.objective_options.selector_stop_gradient` detaches
both inputs.
Weights, masks and loss coefficients are applied outside the logits operators.

## Canonical draft component changes

The canonical component path is still being introduced alongside legacy training.
Its schema version 1 is provisional until the canonical-format cutover (design
step 6). Intermediate configurations must include the current explicit structural
fields, including RoPE `interleaved`, attention `qk_norm_*`, and RMSNorm
`has_weight`; they are not a stable checkpoint interchange format yet. After the
cutover, incompatible changes require a schema migration.

Canonical RoPE computes frequencies, phases, and cosine/sine tables in FP32.
Rotation also computes in FP32 and casts once to the Q/K dtype. This follows
SGLang's base rotary embedding, which keeps an FP32 cosine/sine cache on CUDA
for head sizes 64, 128, 256 and 512, and intentionally changes BF16 numerics
relative to legacy frequency/table rounding. For other head sizes, and for
SGLang's multi-axis rotary embedding, SGLang casts the cache to the activation
dtype before rotating, so such drafts differ from serving by at least that
rounding. Linear scaling divides positions; dynamic NTK
uses a fixed `factor * original_max_position_embeddings`, independent of previous
calls. Legacy EAGLE dynamic configurations remain unsupported. Continued training
can therefore differ numerically; acceptance impact must be measured before the
training cutover.

Runtime attention names are `eager`, `sdpa`, `flex_attention`, and `fa`, matching
training configuration. Canonical `usp` is not implemented and fails explicitly.
Flex defaults to the general TRITON kernel for sparse short-query batches; an
explicit `FlexAttentionBackend(kernel_backend="FLASH")` uses `(256, 128)` blocks.
