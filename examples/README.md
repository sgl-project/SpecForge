# SpecForge Training Examples

All training methods use the same typed entry point. Pick a run config, update
its model and data paths, and launch it directly or through `torchrun`:

```bash
specforge train --config examples/configs/qwen3-8b-eagle3-online.yaml
```

The representative Qwen3-8B configs are below. The complete 51-recipe catalog,
including VLM, NPU, offline, and disaggregated variants, is in
[`examples/configs/README.md`](./configs/README.md).

| Config | Mode | Strategy |
| --- | --- | --- |
| `examples/configs/qwen3-8b-eagle3-online.yaml` | Online target capture | EAGLE3 |
| `examples/configs/qwen3-8b-eagle3-offline.yaml` | Precomputed features | EAGLE3 |
| `examples/configs/qwen3-8b-eagle3-offline-disaggregated.yaml` | Disaggregated precomputed features | EAGLE3 |
| `examples/configs/qwen3-8b-dflash-online.yaml` | Online target capture | DFlash |
| `examples/configs/qwen3-8b-dpace-online.yaml` | Online D-PACE objective | DFlash |
| `examples/configs/qwen3-8b-dflash-disaggregated.yaml` | Disaggregated server capture | DFlash |
| `examples/configs/qwen3-8b-domino-online.yaml` | Online target capture | Domino |
| `examples/configs/qwen3-8b-domino-disaggregated.yaml` | Disaggregated server capture | Domino |
| `examples/configs/qwen3-8b-peagle-online.yaml` | Online target capture | P-EAGLE |
| `examples/configs/qwen3-4b-dspark-disaggregated.yaml` | Disaggregated server capture | DSpark |
| `examples/configs/qwen2.5-vl-7b-eagle3-online.yaml` | Online multimodal target capture | EAGLE3 |
| `examples/configs/qwen2.5-vl-32b-eagle3-online.yaml` | Online multimodal target capture + target TP | EAGLE3 |
| `examples/configs/qwen3.5-4b-dflash-online-npu.yaml` | Ascend NPU online target capture | DFlash |
| `examples/configs/qwen3.5-4b-domino-online-npu.yaml` | Ascend NPU online target capture | Domino |

Online configs point `data.train_data_path` at raw conversation data. The
offline config expects hidden-state checkpoints in `data.hidden_states_path`.
Local offline EAGLE3 derives and caches its vocabulary map when no path is set;
disaggregated EAGLE3 requires one explicit shared `model.vocab_mapping_path`.

The Qwen2.5-VL 7B and 32B configs use the same `specforge train` entry and
EAGLE3 strategy. Set `model.input_modality: qwen2_5_vl`; image pixels are
prepared inside rollout and only M-RoPE position IDs are retained as training
features. VLM supports colocated online capture with a batch size of one; the
32B recipe uses target TP. Each raw VLM record currently carries one image.

The same CLI owns target TP + target-DP, offline DP, and EAGLE3 offline USP.
Online disaggregated consumers use every trainer rank for DP. Use the dedicated
[`run_online.sh`](./disagg/run_online.sh) and
[`run_offline.sh`](./disagg/run_offline.sh) topology wrappers; they only encode
deployment topology and still dispatch to `specforge train`. The complete
environment contract is in the [disaggregated training
guide](../docs/basic_usage/disaggregated_training.md).

Offline feature training supports EAGLE3, DFlash, and Domino, including local
and disaggregated consumers. Optional config sections provide online/offline
evaluation with `<run_id>-best` selection, compact teacher projection for
offline text EAGLE3, and W&B, TensorBoard, SwanLab, or MLflow tracking. See the
[training guide](../docs/basic_usage/training.md) for the full capability
matrix, ROCm installation, and Ascend NPU/HCCL launch example.

Colocated online/offline and disaggregated offline resume are supported.
Disaggregated online resume is consumer-only and requires the retained SQLite
ledger, channel/inboxes, Mooncake data, and an exactly matching checkpoint; the
producer is never resumed.

Paths are resolved from the directory where the command is run. The checked-in
values assume the repository root. Datasets and generated features live under
`cache/`; checkpoints are written under `outputs/`.
