# SpecForge DataFlow Runtime (M1–M4)

A minimal, DataFlow-centered layer over the existing SpecForge model/data code.
It moves SpecForge from a trainer-centered god-script toward explicit contracts:

```
PromptTask -> RolloutWorker -> SampleRef -> FeatureDataLoader -> TrainBatch -> Trainer
```

The **control plane** moves only metadata; large tensors move only through the
**data plane** (FeatureStore). Online and offline converge at `SampleRef`, so the
trainer path has no online/offline branch. Existing model/data/distributed code
(`specforge/core`, `specforge/modeling`, `specforge/data`, `specforge/distributed`)
is reused, not rewritten — the runtime is plumbing around the existing ops, which
is why the equivalence gates are bit-exact.

## Layout

```
specforge/runtime/
  contracts.py            # PromptTask, FeatureSpec, SampleRef, FeatureHandle,
                          #   TrainBatch, assert_no_tensors
  control_plane/
    controller.py         # DataFlowController (in-process; no-tensor invariant)
  data_plane/
    feature_store.py      # FeatureStore ABC + LocalFeatureStore (mem + file + dump)
    sample_ref_queue.py   # SampleRefQueue (lease / ack / fail / depth)
    offline_reader.py     # OfflineManifestReader (.ckpt -> SampleRef)
    feature_dataloader.py # FeatureDataLoader (SampleRef -> TrainBatch)
  inference/
    capture.py            # CaptureConfig + verify_capture (B7/B8)
    rollout_worker.py     # RolloutWorker (strategy-agnostic)
    sglang_adapter.py     # SGLangAdapter.generate_features (the SpecForge<->engine seam)
    sglang_patch_inventory.md  # patch surface + supported-version matrix (M4)
  training/
    strategy.py           # DraftTrainStrategy ABC + Eagle3/DFlash strategies
    backend.py            # TrainingBackend ABC + FSDPTrainingBackend + ParallelConfig
    trainer.py            # TrainerCore (branch-free) + TrainerController + Checkpoint
  launch.py               # wire the offline-EAGLE3 dataflow from a config
```

## Key decisions honored (ADRs)

- **Target representation (ADR-0001):** `FeatureSpec.target_repr` is a tagged
  union; the *strategy* owns the projection so `TrainerCore` stays branch-free.
  Offline EAGLE3 uses `hidden_state` (re-run `TargetHead`) for equivalence;
  `pruned_logits` is the production default (t2d at rollout).
- **Delivery (ADR-0002):** at-least-once + idempotent effects. `commit_samples`
  dedupes on `sample_id`; queue `put`/`ack`/`release` are idempotent.
- **Storage (ADR-0003):** `LocalFeatureStore` is in-memory on the hot path with
  an opt-in disk/mmap dump that doubles as the capture/replay tap.
- **No-tensor invariant:** `SampleRef`/`PromptTask` are frozen dataclasses with no
  tensor fields; the controller runs `assert_no_tensors` on every record.

## Milestone status & exit-gate tests

This branch implements the local DataFlow spine and the focused tests below. It
does **not** complete every acceptance item from the refactor plan: the
`WeightVersion` serving accept-length gate, full optimizer/scheduler resume, and
moving the production DFlash script onto the shared lifecycle remain follow-up
work.

| M | Gate test (`tests/test_runtime/`) | Where it runs |
|---|---|---|
| **M1** | `test_controller_no_tensor.py::...test_controller_carries_no_tensor` | CPU/CI |
| **M1** | `test_equiv_offline_eagle3.py` (offline bit-exact vs `run_forward`) | GPU (rcli) |
| **M2** | `test_equiv_online_eagle3.py` (online vs legacy, BS=1) | GPU (rcli) |
| **M2** | `test_sample_ref_queue.py`, `test_rollout_worker.py` (lease/ack, commit) | CPU/CI |
| **M3** | `test_equiv_trainer_split.py` (paired single step) | GPU (rcli) |
| **M3** | `test_checkpoint_resume.py` | GPU (rcli) |
| **M3** | `test_trainer.py`, `test_seam_fixes.py` (core/accum/checkpoint/DFlash plug-in) | CPU/CI |
| **M4** | `test_extraction_vs_hf_reference.py::test_extraction_vs_hf_reference` | GPU (rcli) |
| **M4** | `test_capture.py` + `..._reference.py::test_capture_layer_mismatch_fails` | CPU/CI |

Plus contract/store/loader unit tests (`test_contracts.py`, `test_feature_store.py`,
`test_feature_dataloader.py`).

## Running the tests

CPU/CI (control + data plane, capture, trainer core — no GPU, no model download):

```bash
PYTHONPATH=$PWD python -m unittest discover -s tests/test_runtime -p "test_*.py" -v
```

GPU equivalence/extraction (tiny synthetic fixtures, no model download) on the
H200 box via rcli:

```bash
rcli exec --sync-code <job> \
  'cd /workspace/SpecForge && PYTHONPATH=$PWD python -m unittest discover -s tests/test_runtime -p "test_*.py" -v'
```

The GPU-only tests are guarded with `@unittest.skipUnless(torch.cuda.is_available())`,
so the same command is safe on CPU (they skip) and on the GPU box (they run).
