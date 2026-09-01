# Kan's DSpark hidden-state layer ablation

Ablates which target-model layers' hidden states feed the DSpark draft:
**5 layers spread across the model** vs **the last 3 layers**, on two targets
(DeepSeek-V4-Flash-0731, Qwen3.6-27B). Draft is the generic `DSparkDraftModel`
(5 draft layers, block_size 7) in all four runs, trained **from scratch on
ShareGPT for 1 epoch** (120,675 samples) at a **global batch of 128**
(~940 optimizer steps), unified across both targets. The only difference
within each pair is `dflash_config.target_layer_ids` in the draft JSON
(feature width of the input projector follows `len(target_layer_ids)`).

DSv4 training hparams are the r2-stabilized set (lr 3e-4 constant, warmup
0.02, clip 0.5): the r1-style lr 6e-4 / clip 1.0 NaN'd both variants (steps
70/210) with the same grad-norm ramp the official r1 run showed at step 1210.
Qwen keeps its recipe hparams (lr 6e-4, warmup 0.04, clip 1.0).

wandb project: `specforge-kan-ablation`.

| run | target | target_layer_ids | node | GPUs |
|---|---|---|---|---|
| `deepseek-v4-flash-dspark-spread5.yaml` | DSv4-Flash (43L) | `[1, 11, 21, 31, 41]` | 10.13.114.103 | all 8 |
| `deepseek-v4-flash-dspark-last3.yaml`   | DSv4-Flash       | `[40, 41, 42]`       | 10.13.114.105 | all 8 |
| `qwen3.6-27b-dspark-spread5.yaml`       | Qwen3.6-27B (64L)| `[2, 17, 32, 47, 62]`| 10.13.114.102 | 0–3 |
| `qwen3.6-27b-dspark-last3.yaml`         | Qwen3.6-27B      | `[60, 61, 62]`\*      | 10.13.114.102 | 4–7 |

Each Qwen run is 2x TP1 capture servers + 2 FSDP trainer ranks
(accumulation 64, global batch 128).

\* Not `[61, 62, 63]`: for the Qwen3.6 (VL-derived) architecture, patched
SGLang captures layer *i*'s output as the *input* of layer *i+1*, so the final
layer's output (id 63) is uncapturable; 62 is the deepest valid id. DSv4's
capture path collects by id directly, which is why its last-3 set `[40, 41,
42]` can include the final layer.

Set-1 configs are single-node `managed_local`: one `specforge train` command
owns the Mooncake master, the SGLang capture server(s), and the trainer. The
two Qwen runs coexist on one node via disjoint GPUs, capture ports, and
Mooncake ports.

## Launch (from the repo root on the assigned node)

Weights load straight from `/cluster-storage/models` (virtiofs; do NOT copy
them onto the pod overlay — the disk usage can get the pod killed). The
virtiofs risk is *concurrent* opens of the same shard (`fuse_open` deadlock,
wedges GPUs until reboot), so serialize the loads instead:
`SPECFORGE_SERIAL_SERVICE_STARTUP=1` makes the managed launcher bring up
capture servers one at a time, and the two Qwen runs sharing a node must be
staggered — start the second only after the first is publishing and its
trainer has finished its `Loading [...] from model-*.safetensors` reads.

After a pod recreation, first re-apply the SGLang patch and deps on every
node: `bash scripts/apply_sglang_spec_capture_patch.sh --target v0.5.18` and
`pip install --no-deps accelerate tensorboard yunchang`.

```bash
export SPECFORGE_SERIAL_SERVICE_STARTUP=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.9
nohup python3 -m specforge.cli train \
  -c examples/configs/kan-ablations/<config>.yaml \
  > outputs/<run_id>.launch.log 2>&1 &
```

Expected length: ~940 optimizer steps for all four runs (global batch 128).
A retry needs a fresh `control_dir` (delete `outputs/<run_id>/` — the
per-run mooncake master dies with the run, so the store id can be reused).

## Ablation set 2: dense FFN vs MoE FFN (spread5)

Second axis, run after set 1 completes: the spread5 configs above serve as
the **dense-FFN arm**; the `*-spread5-moe.*` configs are identical except the
draft JSON adds the official DSpark MoE shape at a quarter of the expert
count — 64 routed experts + 1 shared, top-6, `moe_intermediate_size` 2048
(activated FFN width 6x2048 ~= the dense 12288), sqrtsoftplus gate,
aux-free balance bias (update rate 1e-3), grouped-GEMM dispatch. The MoE FFN
is the `DSparkV4MoE` block swapped into the generic DFlash backbone (gated on
`n_routed_experts` in the draft JSON; see
`specforge/modeling/draft/dflash.py`). Draft sizes: ~8.5B (DSv4) / ~10.8B
(Qwen) vs ~1B dense.

Both MoE runs keep the dense arm's global batch of 128 (same ~940-step
schedule), scaled onto more GPUs:

| run | target | layout |
|---|---|---|
| `deepseek-v4-flash-dspark-spread5-moe.yaml` | DSv4-Flash | TWO NODES: 10.13.114.103 = mooncake master (lease ttl 5m) + 4x TP2 capture servers (8 GPUs); 10.13.114.105 = 8-rank FSDP trainer (accum 16). Capture side launched by hand (prod `launch_dsv4_b200_2node.sh` pattern: serial loads, `MC_TRANSFER_TIMEOUT=300`, `SGLANG_SPEC_CAPTURE_SINK_CLIENTS=3`); trainer runs this config with `SPECFORGE_MOONCAKE_FETCH_CLIENTS=4`. |
| `qwen3.6-27b-dspark-spread5-moe.yaml` | Qwen3.6-27B | WHOLE NODE 10.13.114.102 managed_local: 4x TP1 capture (GPUs 0-3) + 4-rank trainer (GPUs 4-7), accum 32. |

## Ablation set 3: official DSparkV4 drafter architecture (from scratch)

Third axis, run after set 2: the drafter is the OFFICIAL DeepSeek-V4 DSpark
architecture (`DSparkV4DraftModel`: 3 MoE+mHC stages, 256 routed experts +
1 shared, top-6, kv fake quant; `configs/deepseek-v4-flash-dspark-official-
b200.json`) trained **from scratch on ShareGPT** at the ablation's global
batch of 128 (~940 steps). Two arms are aligned to the other sets (spread5
capture layers, block_size 7); one follows the official config exactly.

| run | target | draft config diff vs official | layout |
|---|---|---|---|
| `deepseek-v4-flash-dspark-official-spread5.yaml` | DSv4-Flash | `target_layer_ids` [1,11,21,31,41], `block_size` 7, stage gradient checkpointing | two nodes (.105 capture aux 1 11 21 31 41 / .103 8-rank trainer, accum 16) |
| `deepseek-v4-flash-dspark-official-exact.yaml` | DSv4-Flash | stage gradient checkpointing only (last3 [40,41,42], block 5 as official) | two nodes (.105 capture aux **40 41 42** / .103 trainer) — runs after the spread5 arm since it needs a different capture stack |
| `qwen3.6-27b-dspark-official-spread5.yaml` | Qwen3.6-27B | Qwen width/vocab/ids (hidden 5120, vocab 248320, mask 248070, 64 target layers), spread5 [2,17,32,47,62], `block_size` 7, stage gradient checkpointing | two nodes (.105 = 8x TP1 Qwen capture, aux 2 17 32 47 62 / .103 8-rank trainer, accum 16) — runs after the two DSv4 arms; the single-node 4+4 layout needed CPU-offloaded Adam and ran at 102 s/step |

Training knobs follow the prod official recipes: `attention_backend: native`
(the V4 model builds its own window/block attention), `fsdp_sharding:
FULL_SHARD`, `fsdp_no_sync_grad_accum: false`, and `stage_gradient_checkpointing`
on every arm: without it the 20B drafter on 8x B200 sat at the 183 GB
ceiling and allocator GC thrash made steps ~45 s (GPU util swinging 100%->0%).
All official-arch arms (DSv4 and Qwen) use the
stabilized hparams (lr 3e-4, warmup 0.02, clip 0.5); the generic Qwen arms of
sets 1-2 keep the Qwen recipe (6e-4, 0.04, 1.0). Drafts are ~19.9B (DSv4) / ~25B (Qwen) params, so
checkpoints are ~250-320 GB each (weights + sharded optimizer state).

Node roles for set 3 are SWAPPED relative to set 2 on purpose: 10.13.114.105's
intra-node NCCL runs at ~12 GiB/s (vs ~600 GiB/s on .102/.103 — same topology,
active NVLinks, OK P2P matrix; a host-level fabric/transport fault we cannot
fix from the pod). An FSDP trainer there spends ~80% of GPU time inside
all-gather/reduce-scatter (40 s/step vs ~8 s on a healthy node), so .105
hosts the TP2 capture servers (little NCCL traffic, idle slack) and .103 the
8-rank trainer. Check with `torchrun --nproc_per_node=8 scripts/n105_nccl_bench.py`.

Qwen official arm, final settings: `fsdp_sharding: FULL_SHARD` + `optimizer_cpu_offload:
true` + `SPECFORGE_FSDP_PREFETCH=0` + gc threshold 0.8 -> 32 s/step with memory flat at
143 GB/rank. Measured alternatives on 8x B200: on-GPU Adam thrashed (100 s/step, memory
swinging 84<->172 GB); FSDP prefetch on with offload 48-52 s/step; SHARD_GRAD_OP 70 s/step
(GPU Adam) or 37 s/step (offload). A single-GPU bench showed the drafter's activations
are only ~4 GiB; the pressure is FSDP's per-stage unsharded param+grad buffers (16.6 GiB
each), so the real fix would be finer FSDP units (chunked expert tensors). It also uses
the stabilized hparams 3e-4 / warmup 0.02 / clip 0.5 (the Qwen recipe's 6e-4 / clip 1.0
spiked to grad-norm 7065 at step 60 and lagged the dense arm 0.175 vs 0.347 acc at step
600). This arm is from scratch (no Qwen analog of the target-layer warm start); it was
STOPPED at step ~50 on 2026-08-26 to free the nodes for a DSv4 rerun.

Earlier Qwen official arm memory notes (superseded): the 24.9B Qwen-width drafter sits at the 183 GB
ceiling on 8 ranks even with stage checkpointing (allocator GC thrash: memory
swinging 84<->172 GB, collectives 10x slower from straggling ranks, 100 s/step).
Its config keeps `optimizer_cpu_offload: true` (fp32 masters/Adam on host RAM)
and the trainer is launched with `SPECFORGE_FSDP_PREFETCH=0` (one unsharded
FSDP block resident at a time; see specforge/training/backend.py) plus
`garbage_collection_threshold:0.8`. None of these change the training math.

### Set 3 rerun: target-layer warm start

The from-scratch official-arch arms trailed badly (spread5 acc 0.194, exact
0.055 at step 940 vs 0.32-0.34 for the generic dense drafters). The H200
reference (`specforge-deepseek-v4-flash/qxjld09f`, acc 0.328 at step 940) was
NOT from scratch: its stages were initialized from the target's own layers
40-42 (and it ran lr 6e-4). Both DSv4 official arms were therefore rerun as SEPARATE configs
(`deepseek-v4-flash-dspark-official-{spread5,exact}-init.yaml`, run ids / wandb
names suffixed `-init`) with `draft_checkpoint_path` pointing at that warm start,
keeping lr 3e-4 / warmup 0.02 / clip 0.5. The plain `*-official-*.yaml` files stay
the from-scratch configs (one config file per ablation variant; never edit a
config that a reported run used):

```bash
python3 scripts/init_dspark_v4_from_target.py --output-dir outputs/dspark-v4-official-init-last3 \
  --draft-config examples/configs/kan-ablations/deepseek-v4-flash-dspark-official-exact.json
python3 scripts/init_dspark_v4_from_target.py --output-dir outputs/dspark-v4-official-init-spread5 \
  --draft-config examples/configs/kan-ablations/deepseek-v4-flash-dspark-official-spread5.json --stage-layers 40,41,42
```

`--stage-layers` (new) decouples the stage source layers from the captured
features: the spread5 draft still initializes its 3 stages from target layers
40-42, with the identity `main_proj` placed on the last captured feature
(layer 41's slot of the 5-layer input).

## Results so far (train/acc at step 940, single runs, final-step snapshots)

| target | drafter | capture | acc | pos-0 |
|---|---|---|---|---|
| DSv4 | generic dense | spread5 | 0.322 | 0.605 |
| DSv4 | generic dense | last3 | 0.339 | 0.643 |
| DSv4 | generic MoE-64 | spread5 | 0.293 | 0.583 |
| DSv4 | official, scratch | spread5 | 0.194 | 0.262 |
| DSv4 | official, scratch | last3 (exact) | 0.055 | - |
| DSv4 | official, scratch, rerun (`2qwufsi3`, identical config) | last3 (exact) | 0.179 | - |
| DSv4 | official, target-init (`*-official-spread5-init.yaml`) | spread5 | 0.445 | 0.731 |
| DSv4 | official, target-init (`*-official-exact-init.yaml`) | last3 (exact) | 0.472 | - |
| Qwen | generic dense | spread5 | 0.424 | 0.671 |
| Qwen | generic dense | last3 | 0.406 | 0.658 |
| Qwen | generic MoE-64 | spread5 | 0.436 | 0.687 |

From-scratch official-arch runs are high-variance (0.055 vs 0.179 for the same
config; non-monotonic curves), the warm-started ones smooth and monotonic.

Reference: `specforge-deepseek-v4-flash/qxjld09f` (official arch, target-init,
lr 6e-4, last3) reached acc 0.328 / pos-0 0.504 at step 940 and diverged to NaN
at ~1210-1240. All DSv4 arms here use lr 3e-4 / warmup 0.02 / clip 0.5.
