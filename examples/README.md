# SpecForge Training Examples

All training methods use the same typed entry point. Pick a colocated run config,
update its model and data paths, and launch it directly on one GPU:

```bash
specforge train --config examples/configs/qwen3-8b-eagle3-online.yaml
```

The representative Qwen3-8B configs are:

| Config | Mode | Strategy |
| --- | --- | --- |
| `configs/qwen3-8b-eagle3-online.yaml` | Online target capture | EAGLE3 |
| `configs/qwen3-8b-eagle3-offline.yaml` | Precomputed features | EAGLE3 |
| `configs/qwen3-8b-eagle3-offline-disaggregated.yaml` | Disaggregated precomputed features | EAGLE3 |
| `configs/qwen3-8b-dflash-online.yaml` | Online target capture | DFlash |
| `configs/qwen3-8b-dflash-disaggregated.yaml` | Disaggregated server capture | DFlash |
| `configs/qwen3-8b-domino-online.yaml` | Online target capture | Domino |
| `configs/qwen3-8b-peagle-online.yaml` | Online target capture | P-EAGLE |
| `configs/qwen3-4b-dspark-disaggregated.yaml` | Disaggregated server capture | DSpark |

Online configs point `data.train_data_path` at raw conversation data. The
offline config expects hidden-state checkpoints in `data.hidden_states_path`
and a matching precomputed target-to-draft vocabulary map in
`model.vocab_mapping_path`.

Colocated and offline inputs are currently single-rank. The online
disaggregated consumer is the supported multi-GPU data-parallel topology. Use
the dedicated [`run_online.sh`](./disagg/run_online.sh) and
[`run_offline.sh`](./disagg/run_offline.sh) topology wrappers; the complete
environment contract is in the
[disaggregated training guide](../docs/basic_usage/disaggregated_training.md).

Online resume is not supported in this PR. Use `training.resume_from` only for
the single-rank local offline path documented in the
[training guide](../docs/basic_usage/training.md).

Paths are resolved from the directory where the command is run. The checked-in
values assume the repository root. Datasets and generated features live under
`cache/`; checkpoints are written under `outputs/`.
