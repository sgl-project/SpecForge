# SpecForge Training Examples

All training methods use the same typed entry point. Pick a run config, update
its model and data paths, and launch it directly; multi-process topology is
already recorded in the YAML:

```bash
specforge train --config examples/configs/online/disaggregated/external/qwen3-8b-eagle3-disaggregated.yaml
```

The directory path identifies feature mode, topology, and—in the online
disaggregated case—service ownership. Start with one of these categories, then
choose the model and strategy inside it:

| Category | Use it when | Representative config |
| --- | --- | --- |
| Offline colocated | Feature checkpoints already exist and the trainer can read them directly | [`offline/colocated/qwen3-8b-eagle3-offline.yaml`](./configs/offline/colocated/qwen3-8b-eagle3-offline.yaml) |
| Offline disaggregated | A producer must ingest existing features for a separate trainer pool | [`offline/disaggregated/qwen3-8b-eagle3-offline-disaggregated.yaml`](./configs/offline/disaggregated/qwen3-8b-eagle3-offline-disaggregated.yaml) |
| Online disaggregated, external | Mooncake and patched SGLang are started by the user or scheduler | [`online/disaggregated/external/qwen3-8b-eagle3-disaggregated.yaml`](./configs/online/disaggregated/external/qwen3-8b-eagle3-disaggregated.yaml) |
| Online disaggregated, managed-local | One command should own a single-node Mooncake/SGLang/trainer stack | [`online/disaggregated/managed-local/qwen3-8b-dflash-1server-dp7-disaggregated.yaml`](./configs/online/disaggregated/managed-local/qwen3-8b-dflash-1server-dp7-disaggregated.yaml) |

The complete model, strategy, CUDA/ROCm/Ascend, and resource-layout catalog is
documented in [`examples/configs/README.md`](./configs/README.md). The
`online/colocated` directory is reserved for a topology that the current
runtime does not implement.

Online configs point `data.train_data_path` at raw conversation data. Offline
configs expect strategy-specific feature checkpoints in
`data.hidden_states_path`.
Offline colocated EAGLE3 derives and caches its vocabulary map when no path is set;
disaggregated EAGLE3 requires one explicit shared `model.vocab_mapping_path`.

Online training always uses the disaggregated producer/consumer data plane and
an external or managed-local SGLang capture server. Here `external` means that
SpecForge does not own the service lifecycle; the server may still be on the
same host. Colocated online target loading and the HF/custom online backends are
intentionally unsupported. Online capture is text-only: VLM training,
including Qwen2.5-VL, is not supported. Online evaluation is also unsupported.

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

Offline feature training supports EAGLE3, DFlash, Domino, and DSpark, including
local and disaggregated consumers. Optional config sections provide
online/offline evaluation with `<run_id>-best` selection, compact teacher
projection for offline text EAGLE3, and W&B, TensorBoard, SwanLab, or MLflow
tracking. See the [training guide](../docs/basic_usage/training.md) for the full
capability matrix, ROCm installation, and Ascend NPU/HCCL launch example.

Offline colocated and offline disaggregated resume are supported.
Disaggregated online resume is consumer-only and requires the retained SQLite
ledger, channel/inboxes, Mooncake data, and an exactly matching checkpoint; the
producer is never resumed.

Paths are resolved from the directory where the command is run. The checked-in
values assume the repository root. Datasets and generated features live under
`cache/`; checkpoints are written under `outputs/`.
