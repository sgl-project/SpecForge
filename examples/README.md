# SpecForge Training Examples

All training methods use the same typed entry point. Pick a run config, update
its model and data paths, and launch it directly; multi-process topology is
already recorded in the YAML:

```bash
specforge train --config examples/configs/qwen3-8b-eagle3-disaggregated.yaml
```

The representative configs are below. The complete recipe catalog, including
NPU, offline, and managed/external-service variants, is in
[`examples/configs/README.md`](./configs/README.md).

| Config | Mode | Strategy |
| --- | --- | --- |
| `examples/configs/qwen3-8b-eagle3-disaggregated.yaml` | Disaggregated SGLang server capture | EAGLE3 |
| `examples/configs/qwen3-8b-eagle3-offline.yaml` | Precomputed features | EAGLE3 |
| `examples/configs/qwen3-8b-eagle3-offline-disaggregated.yaml` | Disaggregated precomputed features | EAGLE3 |
| `examples/configs/qwen2.5-7b-eagle3-offline-disaggregated.yaml` | Disaggregated precomputed features | EAGLE3 |
| `examples/configs/qwen3-8b-dflash-online.yaml` | Disaggregated SGLang server capture | DFlash |
| `examples/configs/qwen3-8b-dpace-online.yaml` | Online D-PACE objective | DFlash |
| `examples/configs/qwen3-8b-dflash-disaggregated.yaml` | Disaggregated server capture | DFlash |
| `examples/configs/qwen3-8b-dflash-1server-dp7-disaggregated.yaml` | Managed local one capture server + DP7 | DFlash |
| `examples/configs/qwen3-8b-domino-online.yaml` | Disaggregated SGLang server capture | Domino |
| `examples/configs/qwen3-8b-domino-disaggregated.yaml` | Disaggregated server capture | Domino |
| `examples/configs/qwen3-8b-domino-1server-dp7-disaggregated.yaml` | Managed local one capture server + DP7 | Domino |
| `examples/configs/qwen3-8b-domino-multiserver-disaggregated.yaml` | Managed local Mooncake + two capture servers | Domino |
| `examples/configs/qwen3-8b-peagle-disaggregated.yaml` | Disaggregated SGLang server capture | P-EAGLE |
| `examples/configs/qwen3-4b-dspark-disaggregated.yaml` | Disaggregated server capture | DSpark |
| `examples/configs/qwen3.6-27b-dflash-multiserver-disaggregated.yaml` | Managed local Mooncake + two capture servers | DFlash |
| `examples/configs/qwen3.6-27b-dflash-1server-dp2-disaggregated.yaml` | Managed local one capture server + DP2 | DFlash |
| `examples/configs/qwen3.5-4b-dflash-online-npu.yaml` | Disaggregated NPU SGLang capture | DFlash |
| `examples/configs/qwen3.5-4b-domino-online-npu.yaml` | Disaggregated NPU SGLang capture | Domino |

Online configs point `data.train_data_path` at raw conversation data. The
offline config expects hidden-state checkpoints in `data.hidden_states_path`.
Local offline EAGLE3 derives and caches its vocabulary map when no path is set;
disaggregated EAGLE3 requires one explicit shared `model.vocab_mapping_path`.

Online training always uses an external or managed SGLang capture server and
the disaggregated producer/consumer data plane. Colocated online target loading
and the HF/custom online backends are intentionally unsupported. Online capture
is text-only: VLM training, including Qwen2.5-VL, is not supported. Online
evaluation is also not supported.

The same CLI owns offline DP, EAGLE3 offline USP, and managed capture-server
topology. Trainer `tp_size` remains 1; target TP belongs to SGLang capture
servers, and non-USP trainer ranks consume disjoint data. The optional
[`run_online.sh`](./disagg/run_online.sh) and
[`run_offline.sh`](./disagg/run_offline.sh) scripts are thin single-node
delegates to `specforge train`. The
[`run_offline_2node.sh`](./disagg/run_offline_2node.sh) wrapper only maps the
cluster-provided node rank to the same CLI's producer or consumer role. Launch
topology remains in YAML. The complete
environment contract is in the [disaggregated training
guide](../docs/basic_usage/disaggregated_training.md).

Offline feature training supports EAGLE3, DFlash, and Domino, including local
and disaggregated consumers. Optional config sections provide online/offline
evaluation with `<run_id>-best` selection, compact teacher projection for
offline text EAGLE3, and W&B, TensorBoard, SwanLab, or MLflow tracking. See the
[training guide](../docs/basic_usage/training.md) for the full capability
matrix, ROCm installation, and Ascend NPU/HCCL launch example.

Local offline and disaggregated offline resume are supported.
Disaggregated online resume is consumer-only and requires the retained SQLite
ledger, channel/inboxes, Mooncake data, and an exactly matching checkpoint; the
producer is never resumed.

Paths are resolved from the directory where the command is run. The checked-in
values assume the repository root. Datasets and generated features live under
`cache/`; checkpoints are written under `outputs/`.
