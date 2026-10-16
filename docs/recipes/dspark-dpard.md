# D-PARD for DSpark and DSpark-KV

This opt-in objective follows the [D-PARD reference](https://d-pard-web.readthedocs.io/en/latest/#implementation).
It changes the training objective, not the draft architecture, attention direction,
conditioning source, Markov head, or serving format. Existing recipes retain CE+L1.

Add these fields to an existing DSpark or DSpark-KV recipe:

```yaml
training:
  strategy: dspark_kv
  dspark_objective: dpard
  dspark_dpard_rho: 0.5
  loss_type: dflash
  lk_loss_type: null
  dspark_confidence_head_alpha: 0.0
```

Use `strategy: dspark` for hidden-state conditioning. To match the previous
no-confidence experiments, also keep `enable_confidence_head: false` in the draft
model config. This objective does not enable convolution or change causal attention.
Keep other recipe settings unchanged and omit `max_checkpoints` to retain exports.

## Objective

Let P be the frozen target softmax and Q the Markov-corrected draft softmax.
For each valid block position i:

- Acceptance: `a_i = sum_v min(P_iv, Q_iv) = 1 - TV(P_i, Q_i)`.
- Smoothed survival: `s_i = rho + (1 - rho) * a_i`.
- Detached credit: `w_i = stop_gradient(sum_{j >= i} product_{k <= j} s_k)`.
- Local divergence: `D_i = -2 * logsumexp_v(0.5 * (log P_iv + log Q_iv))`.
- Draft loss: `sum_valid(w_i * D_i) / number_of_valid_positions`.

The denominator is the valid-token count across data-parallel ranks in each
forward, not the sum of credits or the number of sequences. For fully valid
blocks this matches the reference's `renyi.numel()`. Existing gradient accumulation
averages microbatch losses; unequal valid-token counts across microbatches are not
reweighted into a single concatenated-batch token mean.

There is no CE/TV warmup, LK mixing, fixed position decay, or hard-label Q(y)
in the D-PARD draft objective. `dspark_ce_loss_alpha`, `dspark_l1_loss_alpha`, and
`loss_decay_gamma` do not scale that objective. Combining it with DPaCE or LK is
rejected rather than silently changing its meaning.

Target logits and position credits are detached. Log-softmax/logsumexp are
computed in FP32 for BF16 training. Padding neither reduces survival nor adds
continuation credit. Chunking splits anchors, never the positions within a block.
The existing frozen target head projects `target_last_hidden_states`; no full
target probability tensor needs to be stored on disk or transferred by producers.

If a confidence head is enabled, its existing BCE auxiliary objective, fixed
position weights, normalization and coefficient remain unchanged. Set its loss
coefficient to zero for pure D-PARD. CE/L1 metrics remain diagnostic; the new
`train/dpard_loss` reports the actual draft objective. Existing accuracy and
acceptance proxy metrics keep their definitions.

## Validation and resume

```bash
python -m unittest tests.test_modeling.test_dspark_dpard -v
```

Tests cover the reference formula and analytic gradients, detached teachers and
credit, masking, extreme logits, hidden/KV forward-backward, chunk equivalence,
distributed normalization, configuration wiring and resume contracts.

The objective, rho and normalization are recorded in the resume contract. A
CE+L1 checkpoint is not an exact optimizer-state resume into D-PARD; use an explicit
weight-only warm start for that change. No SGLang serving changes are required.
