# Draft-OPD replay support

**Experimental library support.** This provides the loss and DFlash2 replay
forward for a future on-policy training pipeline. It does not register a new
training strategy or provide a runnable rollout/training launcher. Sliding
attention honors an explicit draft configuration `is_causal=False`; absent or
unspecified causality retains the legacy causal sliding mask.

The objective is an adaptation of the
[Draft-OPD TV variant](https://github.com/Simplified-Reasoning/Draft-OPD/tree/e81a8bc488ef762f178f8708ffbc49bd92c3a93d)
to DFlash2's sparse, predecessor-conditioned selector. It is not a reproduction
of the authors' training setup or a claim of measured acceptance improvement.
The paper's accepted-token forward KL and rejected-suffix reverse KL are a
different objective; this module implements the TV/first-rejection variant.

## Rollout contract

`OPDDFlash2Model` in `specforge.algorithms.common.dflash_opd` accepts one audited
rollout per forward. Its constructor takes the same draft, frozen target head,
frozen target embedding and objective options as `OnlineDFlashModel`. Keep
`selector_stop_gradient=False` for coupled gradients.

The caller supplies `input_ids[1, S]`, committed target taps
`hidden_states[1, H, D]`, a verified-continuation `loss_mask[1, S]`, and a
`replay` dictionary. `H` may equal `S` or `S - 1`: a final residual/bonus token
can be returned before the target has processed it. The missing tap is padded
and must never be used as earlier context.

Let `B` be the number of verification blocks, `L = block_size - 1`, `K` the
selector width, and `T` the complete sparse teacher support width.

| Replay field | Shape | Meaning |
| --- | --- | --- |
| `anchor_positions` | `[B]` | Unique absolute positions of the clean anchor token |
| `proposed_ids` | `[B, L]` | Actual proposals, including the first rejection |
| `accepted_lengths` | `[B]` | Accepted proposal count, excluding anchor and bonus |
| `exposed_lengths` | `[B]` | Proposal exposure after EOS/length censoring |
| `candidate_ids` | `[B, L, K]` | Natural unary candidates in their serving order |
| `q` | `[B, L, K]` | Actual conditional selector probabilities at temperature 1 |
| `target_ids` | `[B, L, T]` | Complete filtered teacher support at each proposal |
| `target_probs` | `[B, L, T]` | Original teacher probabilities, summing to one |

Replay recomputes natural candidates and selector scores, conditioning each
position on the actual preceding proposal. It requires candidate identity and
order to match and the maximum probability difference to be at most `0.01`.
That threshold is a provisional validation gate, not measured backend parity.
The producer must bind every record to the current draft policy and exact
teacher sampling/filtering configuration. Final accepted text alone cannot
reconstruct the rejected proposal or its verifier distribution.

### Serving and replay must describe the same policy

The frozen head supplied to replay must reproduce the head used to generate
the draft proposals. For a quantized serving head, a separate BF16 checkpoint
or a BF16 dequantization is not sufficient evidence of equivalence. Check both
the natural candidate IDs and their conditional probabilities against actual
serving captures. The same requirement applies to attention causality, KV
precision, rotary computation and fused operations. Do not force the captured
candidate IDs or relax the probability gate to hide a forward mismatch.

With `is_causal=False`, draft positions can attend to future positions within
their own block, while the sliding lower bound still excludes old positions.
They cannot attend to another draft block or to target taps at or after the
anchor. Both dense and FlexAttention masks enforce this contract. Full-attention
behavior and the default causal sliding mask are unchanged.

## Objective and normalization

`first_rejection_mask` retains accepted proposals plus the first rejection,
clipped by exposure. Rejected suffixes and residual/bonus tokens are excluded.
Zero-exposure blocks contribute zero to the loss and its denominator.

For each retained position, the overlap is
`A = sum(v in C, min(p(v), q(v)))`, with `C` the natural candidate set.
`1 - A` is the exact total variation distance for a draft supported on `C`.
Teacher mass outside `C` is retained implicitly; do not renormalize `p` into
the candidate set.

For a block with `N` exposed positions, minimize
`1 - sum(cumprod(A)) / N`. With a predecessor-conditioned selector, this is a
replay-path surrogate, not exact unconditional serving acceptance length.
Teacher probabilities are detached. A zero-overlap block remains finite.
Sparse TV can also have positive overlap and zero score gradient: when every
selected `q(v)` exceeds `p(v)`, overlap is the constant teacher mass on the
candidate set. For example, `q=(0.5, 0.5)` and `p(C)=(0.2, 0.2)` give overlap
`0.4` and a one-position loss of `0.6` with zero score gradient. Report teacher
coverage and this plateau frequency when diagnosing a weak learning signal;
the example alone does not establish its frequency or performance impact.

The wrapper returns an **unnormalized block sum**, additive block statistics,
and replay agreement diagnostics. By default it adds `0.1` times the inherited
auxiliary objective, weighted by the record's valid-block count. Configure that
objective through the inherited DFlash/D-PACE, LK and selector options. Set
`auxiliary_coefficient=0` to skip its forward entirely.

Normalize across the complete optimizer accumulation window. For averaged
DDP/FSDP reductions, `global_block_loss` scales local sums by
`world_size / global_valid_blocks`; do not divide by accumulation again.
Alternatively, scale accumulated synchronized gradients by that factor before
clipping. Unequal numbers of blocks across ranks must not become an unweighted
average of local means.

## Integration still required

An end-to-end trainer must supply actual speculative verifier records, fully
audit each generated input, synchronize only draft weights at every optimizer
boundary, and reject stale policies. Checkpoints must recover optimizer,
scheduler, rollout RNG, data cursor and policy identity together. These parts
are not implemented by this module.

CPU tests cover dense-reference values/gradients, first rejection and censoring,
unequal rank/accumulation counts, sparse-support plateaus, selector/backbone
gradients, and prevention of future-context leakage, including noncausal sliding
blocks. Passing these tests does not establish native backend parity,
distributed optimizer recovery, performance or held-out acceptance gains for
an integration using this library.
