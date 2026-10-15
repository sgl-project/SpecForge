# DSpark-KV with convolution and DPaCE or D-PARD

The DSpark-KV training path supports two separate convolutional recipes:

- `examples/configs/online/disaggregated/external/qwen3-8b-dspark-kv-conv-dpace-online.yaml`
- `examples/configs/online/disaggregated/external/qwen3-8b-dspark-kv-conv-dpard-online.yaml`

Both use `training.strategy: dspark` and `model.conditioning_source: kv`.
Target KV geometry is inferred from the target and saved in the exported draft config.
Both use `configs/qwen3-8b-dspark-kv-conv-causal-noconf-b16.json`: causal attention, target-KV conditioning, de-RoPE/reprojection, context key normalization, a two-tap grouped convolution around attention and MLP, and no confidence head. Convolution starts as an identity mapping. Existing models without convolution settings retain their original layers.

DPaCE uses `dspark_objective: ce_l1` and `loss_type: dpace`. Detached credit is derived from the draft probability of the hard-label continuation and weights the CE and L1 terms. D-PARD instead uses `dspark_objective: dpard` and `loss_type: dflash`; its detached credit comes from target/draft distribution overlap and weights a Renyi-half divergence. These are alternatives, not simultaneously enabled objectives. See `dspark-dpard.md` for the D-PARD formula and normalization.

The independent `is_causal` direction flag works with global or sliding attention. Omitting it preserves legacy mask semantics. DSpark resume metadata records convolution, direction, and objective settings to detect incompatible exact resumes.

## Online capture prerequisites

Online KV capture currently supports Qwen3. Existing offline Qwen3.5 KV capture and the cached/de-RoPE/key-normalization options remain supported. Offline datasets must include `target_last_hidden_states` as well as `target_kv` when the objective needs teacher probabilities.

The online producer needs a compatible SGLang 0.5.18 installation with the base SpecForge capture patch and target-KV patch. In the producer environment, apply them explicitly:

```bash
bash scripts/apply_sglang_spec_capture_patch.sh
bash scripts/apply_sglang_target_kv_patch.sh
```

These commands modify that environment's SGLang installation; copying the repository alone does not apply them. Apply the base patch before the KV extension on a fresh SGLang installation. If both are already installed, do not reapply the base patch over the extension. The target-KV installer can be rerun on its own. Mooncake is required for the supplied online recipes; Liger is required when `use_liger_kernel: true` is retained. The compiled objective optimization is optional and defaults off.

Configure model/data paths, capture servers, and Mooncake endpoints before launching. All example paths are relative to the repository working directory. Checkpoints are unlimited by default; the recipes intentionally omit `max_checkpoints`. The two-dimensional projection and selective frozen-head cache optimizations are enabled in these recipes without changing the objective.

For the Qwen3 non-thinking experiment, `qwen3-instruct` uses an empty system prompt. Serving must use the same template. This affects other users of that template too; select a separate template if an explicit system message is wanted.

These recipes configure training, not speculative serving. Export normalization recognizes the KV architecture, but native SGLang KV/conv serving additionally requires its compatible model adapter and worker hooks; the capture patch alone is not a serving implementation.

## Start a local D-PARD training run

The self-contained recipe is
`examples/configs/online/disaggregated/managed-local/qwen3-8b-dspark-kv-conv-dpard-managed.yaml`.
It reserves GPU 0 for TP1 Qwen3 capture and GPU 1 for training. SpecForge starts
and stops the loopback Mooncake and SGLang services; do not launch them manually.
Use two available GPUs with sufficient memory and unused ports, and activate the
environment containing SpecForge, patched SGLang 0.5.18, Mooncake, and Liger.
Run all commands from the repository root.

Update the model and dataset paths in the YAML, or override them at launch:

```bash
RECIPE=examples/configs/online/disaggregated/managed-local/qwen3-8b-dspark-kv-conv-dpard-managed.yaml
MODEL=shared/checkpoint/Qwen3-8B
DATA=shared/datasets/perfectblend/qwen3-8b/perfectblend_train_regen_temperature0_no_think_1.jsonl
specforge train -c "$RECIPE" --plan \
  "model.target_model_path=$MODEL" "data.train_data_path=$DATA"
specforge train -c "$RECIPE" \
  "model.target_model_path=$MODEL" "data.train_data_path=$DATA"
```

This starts a fresh draft for six epochs, not a continuation of the single-record
experiments. The effective global batch is `1 trainer x 2 samples x 2 accumulation
steps = 4`. It uses pure D-PARD with rho 0.5, causal KV+conv, no confidence loss,
and BF16 training. CE and L1 metrics are diagnostics, not extra D-PARD loss terms.
The YAML does not enable FP32 LM-head serving or install a speculative-serving
adapter. Changing the serving precision alone is not FP32 training.

Checkpoints are saved every 3000 optimizer steps and at the end under
`outputs/qwen3-8b-dspark-kv-conv-dpard-managed/consumer`; all are retained.
TensorBoard events are written below that directory:

```bash
tensorboard --logdir outputs/qwen3-8b-dspark-kv-conv-dpard-managed/consumer \
  --host 127.0.0.1 --port 6006
```

For an additional run, give `run_id`, `output_dir`,
`deployment.disaggregated.control_dir`, and
`deployment.disaggregated.consumer_state_dir` fresh values. To warm-start an
exported draft, also set `model.draft_checkpoint_path=path/to/draft_hf`; this
resets the optimizer, scheduler, and data position. It is not an exact resume.
For a short pipeline check, point `DATA` to a one-record JSONL and override
`training.num_epochs=20 training.max_steps=20 training.total_steps=20
training.batch_size=1 training.accumulation_steps=1 training.num_anchors=32
training.objective_chunk_blocks=32 data.max_length=512` with fresh run paths.
A successful short training run does not establish full speculative acceptance.

## Validation

```bash
python -m unittest tests.test_modeling.test_dspark_conv_dpace tests.test_modeling.test_dspark_dpard tests.test_modeling.test_dspark_target_kv tests.test_runtime.test_target_kv_online -v
```

Tests cover both full KV+conv objectives and nonzero convolution gradients, identity initialization, export/reload, independent loss references, projection caching, capture metadata validation, and the offline path. GPU-only cases are skipped when CUDA is unavailable.
