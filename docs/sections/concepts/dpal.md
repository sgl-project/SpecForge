# D-PAL for DFlash

D-PAL trains a selector-free DFlash draft using teacher–draft agreement to weight positions in each proposal block. Choose the per-token loss under `training.loss.terms.token` and enable the position weights under `training.weighting`. Both settings work with the existing DFlash preset.

| D-PAL variant | Token loss configuration | Per-position loss |
|---|---|---|
| Rényi | `{kind: renyi_half}` | −2 log Σ√(pq) |
| KL | `{kind: kl}` | KL(p ∥ q) |
| LA | `{kind: neg_log_overlap}` | −log Σ min(p, q), evaluated in log space without the epsilon floor used by `nla` |
| CE+TV | `{kind: ce_l1, options: {ce_weight: 0.1, l1_weight: 0.45}}` | 0.1 CE + 0.9 TV, where TV is half the L1 distance |

Here p is the frozen teacher distribution and q is the draft distribution. CE uses the recorded token. All four use the same position weighting:

```yaml
training:
  loss:
    terms:
      token: {kind: renyi_half}
  weighting: {kind: dpal, alpha: 0.3}
```

The smoothing floor `alpha` (ρ in the paper) defaults to 0.3. Each valid slot contributes a factor ρ + (1 − ρ) Σ min(p, q). Its weight is the sum of cumulative prefix products ending at that slot or later. Masked slots are skipped. Weights are detached. Losses sum within an anchor, average over valid anchors within each sequence, then average over valid sequences across the optimizer window and data-parallel ranks.

## Teacher features

For offline training, use the same edited DFlash training YAML for [capture](/basic_usage/offline_capture) and training. If a cache contains only auxiliary features, capture again into a fresh directory and set `data.hidden_states_path` to that directory. Online training requests final teacher hidden states automatically.

[Wu et al., Beyond the Point Mass](https://arxiv.org/abs/2610.28355).
