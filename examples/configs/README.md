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

Every recipe records its audited process count under `deployment.trainer`.
Multi-process configs self-launch through torch distributed:

```bash
specforge train -c examples/configs/qwen3-30b-a3b-eagle3-online.yaml
```

The filename is the index: `*-online.yaml` performs target capture while
training, `*-offline.yaml` consumes precomputed features, `*-disaggregated.yaml`
supervises both roles on one trainer node or selects a split role with
`--role`, and `*-npu.yaml` retains the HF + SDPA Ascend recipes.
Qwen2.5-VL has both 7B and 32B online recipes and uses the same EAGLE3 trainer.

The `qwen3-8b-dflash-1server-dp7-disaggregated.yaml`,
`qwen3-8b-domino-1server-dp7-disaggregated.yaml`,
`qwen3-8b-domino-multiserver-disaggregated.yaml`,
`qwen3.6-27b-dflash-1server-dp2-disaggregated.yaml`, and
`qwen3.6-27b-dflash-multiserver-disaggregated.yaml` recipes are opt-in local
full-stack examples. Their typed `managed_local` blocks own Mooncake, one or
two patched SGLang capture servers, and the trainer GPU allocation; the same
`specforge train -c ...` command starts and cleans up each complete stack.
Disaggregated recipes without `managed_local` keep Mooncake and SGLang external
for scheduler- or service-managed deployments.

Before running a recipe, update model/data paths and create any referenced
offline feature or vocabulary-mapping artifacts. GPU visibility, Mooncake
addresses, and NPU runtime variables remain deployment concerns and are passed
through the environment; they are intentionally not embedded in portable YAML.

## Capability matrix

| Strategy | Colocated online | Offline features | Disaggregated online | Disaggregated offline |
| --- | --- | --- | --- | --- |
| EAGLE3 | target TP + target-DP | DP + USP | consumer DP | consumer DP |
| DFlash | SGLang target TP + target-DP; HF replicas + batch partition | DP | consumer DP | consumer DP |
| Domino | SGLang target TP + target-DP; HF replicas + batch partition | DP | consumer DP | consumer DP |
| DSpark | No | No | consumer DP | No |
| P-EAGLE | batch size 1 | No | No | No |

Target TP in the DFlash and Domino rows is SGLang-only. With
`model.target_backend: hf`, every rank loads a complete target replica;
`training.tp_size` coordinates the shared capture batch and each rank's local
batch partition but does not shard target weights.

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

specforge train -c examples/configs/qwen3.5-4b-dflash-online-npu.yaml
```

The unified launcher provides rank/world/rendezvous variables and the runtime
selects HCCL when `torch_npu` is active. For AMD GPUs, install
`requirements-rocm.txt`; HF + SDPA is the portable ROCm starting point.

Colocated online/offline and disaggregated offline resume are supported.
Disaggregated online recovery resumes only the consumer against retained
control/data-plane state; capture producers always start a fresh attempt.

Migration notes:

- The former `run_qwen3_8b_dflash_disagg_1srv_dp7.sh` self-contained topology
  is retained as `qwen3-8b-dflash-1server-dp7-disaggregated.yaml`: one managed
  capture server on GPU 0 and seven DFlash trainer ranks on GPUs 1–7.
- The former `run_qwen3_8b_domino_disagg_1srv_dp7.sh` self-contained topology
  is retained as `qwen3-8b-domino-1server-dp7-disaggregated.yaml`: one managed
  capture server on GPU 0 and seven trainer ranks on GPUs 1–7.
- The former `run_qwen3.6_27b_dflash_disagg.sh` one-server topology is retained
  as `qwen3.6-27b-dflash-1server-dp2-disaggregated.yaml`: one managed capture
  server on GPU 0 and two trainer ranks on GPUs 1–2. The external-service YAML
  remains available for scheduler-managed deployments.
- The latest pre-cleanup `run_qwen3_8b_domino_disagg_multiserver.sh` had been
  reduced to one SGLang server and one URL despite its historical name. The
  managed Qwen3-8B Domino recipe above restores a genuine two-server topology
  without restoring the legacy trainer script; the external-service Domino
  recipe also accepts any number of typed `deployment.disaggregated.server_urls`.
- The former GPT-OSS-120B shell accidentally selected the 20B draft config;
  `gpt-oss-120b-eagle3-online.yaml` points to the matching 120B config.
- The former Qwen3-235B shell accidentally launched Qwen3-Next-80B; the two now
  have separate recipes.
- Qwen3-Next online EAGLE3 retains its batch size of two. P-EAGLE requires
  batch size one; the current VLM recipes keep batch size one as a conservative
  default, while the unified VLM collator also supports larger padded batches.
- The old Qwen3.5-35B offline shell had its training command commented out. The
  YAML records that intended offline trainer configuration after feature
  preparation.
- Qwen3-8B DTA still shares the DFlash trainer, as before. Its specialized
  behavior is encoded by the draft JSON (`training_mode: vp_drafter`); there is
  no second DTA training entry.
