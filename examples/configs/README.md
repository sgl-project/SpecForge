# Unified training recipe catalog

Every draft model JSON under `configs/` has at least one typed YAML recipe in
this directory. Run any recipe through the one public training entry:

```bash
specforge train --config examples/configs/qwen3-8b-eagle3-online.yaml
```

`model.draft_model_config` may name a local JSON file, a local model directory,
or a Hugging Face repository. Fresh EAGLE3, P-EAGLE, and DFlash runs may omit it
and derive the draft architecture from the target; see the
[training guide](../../docs/basic_usage/training.md#draft-configuration-and-model-initialization)
for layer/block overrides and the distinction between weights-only
`model.draft_checkpoint_path` and full `training.resume_from`.

For multi-process colocated runs, launch the same entry with the topology
encoded in `training.tp_size` and the required world size:

```bash
torchrun --standalone --nproc_per_node 4 "$(command -v specforge)" \
    train --config examples/configs/qwen3-30b-a3b-eagle3-online.yaml
```

The filename is the index: `*-online.yaml` performs target capture while
training, `*-offline.yaml` consumes precomputed features, `*-disaggregated.yaml`
is launched once per producer/consumer role through the wrappers in
`examples/disagg/`, and `*-npu.yaml` retains the HF + SDPA Ascend recipes.
Qwen2.5-VL has both 7B and 32B online recipes and uses the same EAGLE3 trainer.

Before running a recipe, update model/data paths and create any referenced
offline feature or vocabulary-mapping artifacts. GPU visibility, Mooncake
addresses, and NPU runtime variables remain deployment concerns and are passed
through the environment; they are intentionally not embedded in portable YAML.

## Capability matrix

| Strategy | Colocated online | Offline features | Disaggregated online | Disaggregated offline |
| --- | --- | --- | --- | --- |
| EAGLE3 | target TP + target-DP | DP + USP | consumer DP | consumer DP |
| DFlash | target TP + target-DP | DP | consumer DP | consumer DP |
| Domino | target TP + target-DP | DP | consumer DP | consumer DP |
| DSpark | No | No | consumer DP | No |
| P-EAGLE | batch size 1 | No | No | No |

Qwen2.5-VL 7B and 32B remain supported through colocated online EAGLE3. The
32B recipe sets `training.tp_size: 4`; VLM captures the full target batch and
partitions it locally rather than using `model.shard_target_output`.

`qwen3-8b-dpace-online.yaml` is the D-PACE recipe. It deliberately uses the
shared DFlash strategy with `training.loss_type: dpace`; D-PACE is an objective
selection inside the unified trainer, not another training entry.

Evaluation is enabled by pairing `training.eval_interval` with
`data.eval_data_path` for colocated online training or
`data.eval_hidden_states_path` for offline training. Best checkpoints are
linked as `<run_id>-best`. Offline text EAGLE3 may enable
`training.compact_teacher`; `tracking.report_to` selects `none`, W&B,
TensorBoard, SwanLab, or MLflow.

## Ascend NPU launch

Install a vendor-matched PyTorch and `torch_npu` first. The `*-npu.yaml`
recipes use HF target capture and SDPA. For example:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

torchrun --standalone --nproc_per_node=4 "$(command -v specforge)" \
  train --config examples/configs/qwen3.5-4b-dflash-online-npu.yaml
```

`torchrun` provides rank/world/rendezvous variables and the runtime selects
HCCL when `torch_npu` is active. For AMD GPUs, install
`requirements-rocm.txt`; HF + SDPA is the portable ROCm starting point.

Colocated online/offline and disaggregated offline resume are supported.
Disaggregated online recovery resumes only the consumer against retained
control/data-plane state; capture producers always start a fresh attempt.

Migration notes:

- The former GPT-OSS-120B shell accidentally selected the 20B draft config;
  `gpt-oss-120b-eagle3-online.yaml` points to the matching 120B config.
- The former Qwen3-235B shell accidentally launched Qwen3-Next-80B; the two now
  have separate recipes.
- Qwen3-Next online EAGLE3 retains its batch size of two; only P-EAGLE and the
  current VLM recipes require batch size one.
- The old Qwen3.5-35B offline shell had its training command commented out. The
  YAML records that intended offline trainer configuration after feature
  preparation.
- Qwen3-8B DTA still shares the DFlash trainer, as before. Its specialized
  behavior is encoded by the draft JSON (`training_mode: vp_drafter`); there is
  no second DTA training entry.
