# EAGLE3 offline disaggregated parity

This page preserves a historical correctness check from commit `40d8feff`.
It is evidence that the offline producer/consumer path and colocated path were
numerically aligned in that experiment, not a current-release performance or
quality claim.

## Qwen2.5-7B, two H200 nodes

The comparison used the same Qwen2.5-7B model build, precomputed feature
tensors, and random seed for both paths. The disaggregated consumer and the
colocated baseline assembled the EAGLE3 runtime through the same training
implementation. Feature tensors were byte-identical; the remaining metric
differences were approximately `1e-6` to `1e-8` GPU numerical noise.

| Step | Metric | Disaggregated | Colocated |
| ---: | --- | ---: | ---: |
| 20 | `acceptance_rate` | 0.0013300 | 0.0013300 |
| 20 | `ploss` | 5.386736 | 5.386740 |
| 20 | `acc` | 0.0272590 | 0.0272590 |
| 120 | `acceptance_rate` | 0.0223610 | 0.0223505 |
| 180 | `acceptance_rate` | 0.0337013 | 0.0336982 |

Both accuracy and the training-time acceptance proxy increased during the run.
Per-step values remained noisy because the experiment used batch size 1 over
64 diverse samples. This proxy is useful for path parity; it is not the same
measurement as serving-time accepted length.

## Unified entry for a new run

The current equivalent recipe is
`examples/configs/qwen2.5-7b-eagle3-offline-disaggregated.yaml`. To place the
offline producer and consumer on different nodes, run one checked-in command
through the cluster launcher:

```bash
rcli exec --per-node <job> \
  'CONFIG=examples/configs/qwen2.5-7b-eagle3-offline-disaggregated.yaml bash examples/disagg/run_offline_2node.sh'
```

Rank 0 invokes `specforge train --role producer`; rank 1 invokes
`specforge train --role consumer`. Both nodes must resolve the config's
`control_dir`, `store_root`, hidden-state input, and vocabulary mapping to the
same data. Use a fresh attempt directory, then compare against the colocated
`examples/configs/qwen2.5-7b-eagle3-offline.yaml` recipe with the same inputs,
seed, and training overrides.
