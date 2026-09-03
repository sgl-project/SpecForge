# D-PARD

D-PARD replaces DSpark's CE/L1 actor with Rényi-half divergence and detached
position weights derived from exact rejection-sampling acceptance.

For target distribution `p_t` and draft distribution `q_t` at temperature 1:

```text
R_t = -2 log sum_v sqrt(p_t(v) q_t(v))
a_t = sum_v min(p_t(v), q_t(v))
s_t = alpha + (1 - alpha) a_t
W_t = stop_gradient(sum_{k=t}^D product_{i=1}^k s_i)
```

The actor is `sum_t W_t R_t / global_sequence_count`.
It has no calibration, valid-position, or weight-mass denominator.
The confidence head remains trainable with the existing static gamma weights,
exact-overlap target, and static-weight-mass normalization.

## Offline training

From the repository root:

```bash
specforge train --config examples/configs/offline/colocated/qwen3-4b-dspark-dpard-offline.yaml
```

Set `data.hidden_states_path` to a SpecForge DSpark cache containing
`input_ids`, `loss_mask`, `hidden_states` for layers `[1, 17, 33]`, and
`target_last_hidden_states`. Speculators-native caches are a different format.

The example uses Qwen3-4B, three full-attention draft layers, B16, 512 anchors
per sequence, gamma 7 for confidence, alpha 0.5, and seed 42. It trains for six
epochs on two GPUs with batch size 1 and accumulation 2. Attention, optimizer,
and data loading use the native SpecForge paths.

Enable the objective with `training.loss_type: dpard` and set both
`dspark_ce_loss_alpha` and `dspark_l1_loss_alpha` to zero. `dpard_alpha` must be
strictly between zero and one. D-PARD cannot be combined with LK loss.
The default `loss_type: dflash` preserves the standard DSpark CE/L1 loss.

`dpard_loss` reports the weighted actor mean. `dpard_credit_position` reports
the mean detached weight at each block position (zero-based). Objective and
smoothing settings are checked when resuming a checkpoint.
