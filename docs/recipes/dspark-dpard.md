# DSpark D-PARD

Adapted from [SpecForge PR #973](https://github.com/sgl-project/SpecForge/pull/973),
commit `25899d3`. This port adds the objective to the existing hidden-state DSpark
training path; it does not import the PR stack's KV conditioning or convolution.

```yaml
training:
  strategy: dspark
  dspark_objective: dpard
  dspark_dpard_rho: 0.5
  loss_type: dflash
  lk_loss_type: null
```

For target distribution P and Markov-corrected draft distribution Q, acceptance
is `a_i = sum_v min(P_iv, Q_iv)`. Smoothed survival is
`s_i = rho + (1-rho) * a_i`; detached position credit is
`w_i = stop_gradient(sum_{j>=i} product_{k<=j} s_k)`.
The local divergence is `D_i = -2 * logsumexp_v((log P_iv + log Q_iv)/2)`.
The draft objective is `sum_valid(w_i * D_i) / number_of_valid_tokens`, pooled
across data-parallel ranks. Padding neither reduces survival nor adds credit.
The teacher is frozen; logarithms and probability reductions use FP32.

DPARD replaces the CE/L1 draft objective. `dspark_ce_loss_alpha`,
`dspark_l1_loss_alpha`, and fixed position decay do not scale DPARD. CE/L1 remain
diagnostics; `dpard_loss` reports the new objective. DPACE and LK combinations
are rejected. Existing DPACE recipes remain supported through the default
`dspark_objective: ce_l1`.

The confidence BCE auxiliary loss retains its coefficient, position decay, and
token normalization. The DeepSeek recipe below preserves its coefficient 1.0;
set `dspark_confidence_head_alpha: 0.0` for pure DPARD. Captured target final hidden
states are required even when CE/L1/confidence coefficients are all zero.

The new recipe is
`examples/configs/online/disaggregated/managed-local/deepseek-v4-flash-dspark-dpard-h20.yaml`.
It has separate run/output/control/store namespaces from the DPACE recipe. It
uses the same GPUs and capture port; run the two recipes separately.
It omits `max_checkpoints` (default 0 retains all checkpoints).

```bash
specforge train -c examples/configs/online/disaggregated/managed-local/deepseek-v4-flash-dspark-dpard-h20.yaml --plan
python -m unittest discover -s tests/test_utils -p test_dspark_dpard.py -v
```

The resume contract records the objective, rho and valid-token normalization.
Switching objectives is not an exact optimizer-state resume; use a new run or
an explicit weight-only warm start. Existing gradient accumulation averages
microbatch losses rather than recomputing one token mean across microbatches.

Tests check formula/analytic gradients, detached credits/teacher, masks, extreme
logits, chunking, DP normalization, configuration wiring and confidence parity.
Full DeepSeek training quality, throughput and peak memory remain unmeasured.
This port uses the reference eager DPARD implementation, not a fused DPARD kernel.
