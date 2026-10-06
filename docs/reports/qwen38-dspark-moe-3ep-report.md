# Qwen3.8-27B DSpark MoE drafter: three-epoch result and reproduction

Status as of 2026-09-17. Everything referenced is on the shared `/personal` volume; node paths refer to the
8x B200 boxes 10.13.114.102 (training) and 10.13.114.103 (eval serving).

## 1. Result

Final drafter: `qwen38-dspark-moe-3ep-cont-step9916` (20.8B total / 1.08B active parameters, DSpark block size 7),
trained for 3 epochs (14,874 optimizer steps at global batch 256 = 3.81M samples) on
`RadixArk/Qwen3.8-27B-Regen-Mixture-v1` against target `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead`.

Accepted drafts per verify step (aggregate accept length minus the bonus token; ceiling 7 for every server), same target,
gamma 7, identical request sets. Accuracy in parentheses where defined. Eval run 2026-09-14 20:56-23:28 UTC.

| dataset | MoE 3 ep (this run) | MoE 1 ep (step 4958) | dense 1 ep (step 2500) | official v1 | 3 ep / official |
|---|---|---|---|---|---|
| GSM8K (1319, 5-shot, non-thinking) | **4.56** (0.963) | 4.18 (0.959) | 4.01 (0.958) | 4.33 (0.952) | 1.05 |
| MATH500 (500, thinking) | **3.67** (0.916) | 3.30 (0.912) | 3.13 (0.926) | 3.61 (0.918) | 1.02 |
| MT-Bench (80 x 2 turns, thinking) | **2.80** | 2.62 | 2.50 | 2.78 | 1.01 |
| AIME26 (30, thinking, 32k) | 3.04 (0.800) | 2.75 (0.867) | 2.78 (0.867) | **3.15** (0.867) | 0.96 |
| training traces (512, thinking) | **2.69** | 2.41 | 2.29 | 2.66 | 1.01 |

- Three epochs add +9-11% accept length over one epoch on every set. The 3-epoch MoE matches or beats the official v1
  drafter (`RadixArk/Qwen3.8-27B-DSpark`, revision b9a5dbdf, 1.86B dense) on four of five sets and trails by 3.5% on
  AIME26, the noisiest set (30 long generations; the 1-epoch MoE measured 2.93 there on 2026-09-09 and 2.75 on 09-14).
- The 1-epoch MoE and dense columns reproduce the 2026-09-09 measurements to within 0.02 except AIME26.
- Accuracy differences are target sampling noise: greedy speculative decoding is lossless.
- Decode throughput on the MoE servers is 10-15% below the official drafter at equal accept length (untuned FusedMoE
  configs for 512 experts of width 512), e.g. MATH500 1485 vs 1720 tok/s.

Baselines: dense 1 ep = step 2500 of run `qwen3.8-27b-dspark-regen-mixture-v1-3ep` in `/personal/SpecForge-qwen38`
(same skeleton, dense FFN 17408, global batch 512, so 2500 steps = 1.28M samples = one epoch); export
`/personal/SpecForge-qwen38/exports/qwen38-dspark-dense-ep1-step2500`. MoE 1 ep = the end of phase A below.

Files: `/personal/SpecForge-qwen38/outputs/launch/eval/results/3ep-moe-vs-1ep-moe-vs-dense-vs-official.{json,log}` plus
per-request records in `…/3ep-moe-vs-1ep-moe-vs-dense-vs-official.requests/`; the results README in that directory holds
every earlier comparison (2026-09-08 and 09-09).

## 2. Artifacts

| what | path | size |
|---|---|---|
| **final training checkpoint, MoE 3 ep** (overall step 14,874; run-local step 4416) | `/personal/SpecForge-qwen38-moe/outputs/qwen3.8-27b-dspark-moe-regen-mixture-v1-cont2ep-restart-step5500-b/qwen3.8-27b-dspark-moe-regen-mixture-v1-cont2ep-restart-step5500-b-step4416` (the `…-latest` symlink in the same directory points here) | 272 GB |
| earlier restart checkpoints (run-local 3500, 4000) | `/personal/SpecForge-qwen38-moe/outputs/qwen3.8-27b-dspark-moe-regen-mixture-v1-cont2ep-restart-step5500-b/qwen3.8-27b-dspark-moe-regen-mixture-v1-cont2ep-restart-step5500-b-step{3500,4000}` | 272 GB each |
| phase B checkpoints (run-local 5000, 5500; 5500 is the phase C warm-start source) | `/personal/SpecForge-qwen38-moe/outputs/qwen3.8-27b-dspark-moe-regen-mixture-v1-cont2ep-from-1ep-v3/qwen3.8-27b-dspark-moe-regen-mixture-v1-cont2ep-from-1ep-v3-step{5000,5500}` | 272 GB each |
| phase A checkpoints (4500, 4958 = the 1-epoch MoE model) | `/personal/SpecForge-qwen38-moe/outputs/qwen3.8-27b-dspark-moe-regen-mixture-v1-1ep-v3/qwen3.8-27b-dspark-moe-regen-mixture-v1-1ep-v3-step{4500,4958}` | 272 GB each |
| **servable export, MoE 3 ep** | `/personal/SpecForge-qwen38-moe/exports/qwen38-dspark-moe-3ep-cont-step9916` | 39 GB |
| servable export, MoE 1 ep | `/personal/SpecForge-qwen38-moe/exports/qwen38-dspark-moe-1ep-v3` | 39 GB |
| **final training checkpoint, dense 1 ep baseline** (step 2500 of the paused 3ep run; branch `kan/qwen38-dspark-ablation`) | `/personal/SpecForge-qwen38/outputs/qwen3.8-27b-dspark-regen-mixture-v1-3ep/qwen3.8-27b-dspark-regen-mixture-v1-3ep-step2500` (earlier: `…-step1500`, `…-step2000` in the same directory) | 25 GB each |
| **servable export, dense 1 ep** | `/personal/SpecForge-qwen38/exports/qwen38-dspark-dense-ep1-step2500` | 3.5 GB |
| official v1 baseline | Hugging Face `RadixArk/Qwen3.8-27B-DSpark`, revision b9a5dbdf | |
| W&B | project `specforge-qwen38-ablation` (https://wandb.ai/yisun0618-nvidia/specforge-qwen38-ablation): MoE runs `…-1ep-v3-b200-dp4`, `…-cont2ep-from-1ep-v3-b200-dp4`, `…-cont2ep-restart-step5500-b-b200-dp4`; dense run `9xmclyf8` | |

A checkpoint directory holds `training_state.pt` (41.6 GB: the full unsharded bf16 draft state dict plus run metadata;
the only file the exporter reads) and `training_state_rank{0-3}.pt` (62.5 GB each: per-rank RNG and CPU-offloaded AdamW
shards). Deletable once exported if the optimizer state is not needed: `qwen3.8-27b-dspark-moe-2node-smoke-knobs/…-step16`,
`…-3ep-v2-step1000` and the partial `…-3ep-v2-step1500` (`.tmp` files only) are leftovers of abandoned runs (575 GB).

## 3. How the model was trained

### 3.1 Drafter

`configs/qwen3.8-27b-dspark-moe-v2.json` in the MoE worktree. Dense DSpark skeleton of the official Qwen3.8 drafter
(5 full-attention layers, hidden 5120, 32 query / 8 KV heads of dim 128, Markov head rank 256, confidence head, mask token
248070, block size 7, target capture layers 5/19/33/47/61 of the 64-layer target) with the FFN of every layer replaced by
an MoE block in the Qwen3.5-MoE style: 512 routed experts of width 512, top-10 softmax routing with renormalisation, one
sigmoid-gated shared expert of width 2048, router centering (`moe_router_center: sample`, momentum 0.99, the fix for the
from-scratch router collapse), aux load-balance loss 0.001, grouped-GEMM dispatch.

**Initialisation.** The draft transformer is trained **from scratch** (normal init, std 0.02; MoE gate and stacked experts via
`MoELayer.reset_parameters`). Nothing is copied from the target's layers. The only target tensors involved are the token
embedding table (`load_target_embedding: true`, key `model.language_model.embed_tokens.weight`, frozen) and the LM head, which the
drafter uses by reference at train and serve time and does not own. Copying target layers is not possible here anyway: the
target is dense (FFN 17408, no experts) with 24 query / 4 KV heads of dim 256 in its 16 full-attention layers, none of which
match the drafter's shapes. The repository's expert warm-start tooling (`scripts/warm_start_moe_drafter.py`,
`specforge/modeling/draft/moe/init.py`) exists only for MoE targets (DeepSeek-V4) and was not used. Phases B and C below are
warm starts from **our own** earlier checkpoints (weights only).

### 3.2 Recipe (identical in all three phases except where noted)

`examples/configs/online/disaggregated/managed-local/qwen3.8-27b-dspark-moe-regen-mixture-v1-1ep-v3.yaml`: single node,
`managed_local` disaggregated mode: 4 TP1 NVFP4 capture servers on GPUs 0-3 (SGLang 0.5.18 + spec-capture patch, triton
attention, 2 running requests, 32k token pool, 100 GiB Mooncake segment each, TCP) feeding a 4-rank FSDP trainer on GPUs 4-7.
DSpark objective (CE 0.1 + L1 0.9 + confidence 1.0, 512 anchors, loss decay gamma 4, 128 objective chunk blocks), flex
attention, `batch_size 4 x accumulation_steps 16 x 4 ranks = global batch 256`, AdamW CPU-offloaded, lr 5e-4 constant after 4%
warm-up, grad clip 1.0, bf16, max_length 8192, chat template `qwen3.5`, seed 42 / prompt_seed 42, checkpoint every 500 steps
(3 kept), `dist_timeout` 15 min. One epoch of the 1,269,290-conversation corpus = 4,958 steps.

### 3.3 The three phases

| phase | run id (`outputs/<run id>/`) | steps | init | wall clock (UTC) | train/acc, loss at end |
|---|---|---|---|---|---|
| A: epoch 1 | `qwen3.8-27b-dspark-moe-regen-mixture-v1-1ep-v3` | 4,958 | scratch | 09-08 01:25 -> 09-09 06:45 (29.3 h) | 0.594, 1.22 |
| B: epochs 2-3, attempt 1 | `…-cont2ep-from-1ep-v3` | planned 9,916; ran 5,620; **checkpoint 5500 used** | weights of A step 4958 | 09-11 06:31 -> 09-12 15:10 (32.7 h) | 0.603, 1.16 |
| C: remainder | `…-cont2ep-restart-step5500-b` | 4,416 (`max_steps`), prompt_seed 44 | weights of B step 5500 | 09-13 17:28 -> 09-14 20:39 (27.2 h) | 0.617, 1.08 |

Total 14,874 steps = 3.00 epochs; about 715 B200-hours including roughly 35 wasted on the two deaths.
Step time 19.7-23 s throughout (capture-side bound at about 11.6 samples/s), peak 128 GB allocated per trainer rank,
`experts_unused_frac` 0 in all phases, `train/acc` on the last logged step 0.617 (peak 0.638 around run-local step 4000 of C).

Why three phases: `managed_local` has no `training.resume_from`, so any interruption becomes a weights-only warm start
(`model.draft_checkpoint_path`) with fresh AdamW moments, a fresh 4% lr warm-up and a fresh data position. Phase B died at
its step 5620 on 2026-09-12 when the shared 49 TB `/personal` volume hit 100% during a checkpoint save (an external writer
was consuming 20-38 GB/min); a first restart from 5500 died the same way three hours later. Phase C, launched after space
was freed, ran to completion. Its prompt_seed 44 gives a shuffle different from both epoch 1 (42) and phase B's second
epoch (43), so a small part of the corpus was seen four times and a similar part twice instead of exactly three times each.

## 4. Reproduction

### 4.1 Code

- Worktree `/personal/SpecForge-qwen38-moe`, branch `kan/moe-3-qwen38`, HEAD `c766217` (2026-09-08). Lineage and key commits are
  in `docs/runbooks/qwen38-27b-dspark-moe-repro.md` section 1. The only uncommitted source change during the runs is an
  opt-in env toggle in `specforge/training/backend.py` (`SPECFORGE_FSDP_SHARDED_GRAD_ACCUM`, default off, so it did not affect
  the runs). The recipes for phases B and C and the helper scripts below are untracked files in the same worktree.
- Serving the MoE drafter needs the SGLang copy `/personal/sglang-0518-moe` (0.5.18 plus `sglang/srt/models/dspark_moe.py`, a
  strict loader for `Qwen3MoeDSparkModel`).

### 4.2 Environment (per node; `/opt` is node-local and disappears on pod rebuild)

```sh
/opt/sglang/bin/python -m pip install --no-deps --target /opt/sglang-0518 /personal/wheels/sglang-0.5.18-*.whl
cd /personal/SpecForge-qwen38-moe && SPECFORGE_SGLANG_ROOT=/opt/sglang-0518 SPECFORGE_SGLANG_VERSION=0.5.18 \
  bash scripts/apply_sglang_spec_capture_patch.sh --target v0.5.18
/opt/sglang/bin/python -m pip install --no-deps tensorboard yunchang
export PYTHONPATH=/opt/sglang-0518:/personal/SpecForge-qwen38-moe HF_HOME=/cluster-storage/models
/opt/sglang/bin/python -c 'import sglang; print(sglang.__version__)'   # 0.5.18
```

Interpreter `/opt/sglang/bin/python` (torch 2.13.0+cu130, transformers 5.12.1, mooncake). W&B credentials in `~/.netrc`.
The target and the official drafter are in the HF cache under `/cluster-storage/models` (their `snapshots/` directories look
empty on this virtiofs mount but the blobs are present, 23 GB and 3.5 GB, and loading by repo id works).

### 4.3 Data

Corpus `RadixArk/Qwen3.8-27B-Regen-Mixture-v1` (1,269,290 conversations, xhigh reasoning). Prepared once into
`/personal/SpecForge-qwen38/cache/dataset/qwen38_regen_mixture_v1_train.jsonl` (40 GB, `{id, conversations}` rows) by
`decompress_shards.py` (zstd shards to JSONL parts) and `slim_rows.py` in that directory; a 1,140-row smoke slice sits next to
it. The tokenised cache `/personal/SpecForge-qwen38/cache/qwen3.8-27b-regen-mixture-v1/` is rebuilt automatically in about
25 minutes whenever the cache key changes; note that the key includes `model.draft_checkpoint_path`, so every warm-start run
tokenises again into a new 72 GB set. Delete stale sets when a run is over.

### 4.4 Training

```sh
cd /personal/SpecForge-qwen38-moe
# phase A: one epoch from scratch (idle 8-GPU node, no other specforge train process)
bash scripts/launch/launch-moe-managed-local.sh qwen3.8-27b-dspark-moe-regen-mixture-v1-1ep-v3
# phase B: two more epochs, warm start from A's final checkpoint (recipe already points at …-1ep-v3-step4958)
bash scripts/launch/launch-moe-managed-local.sh qwen3.8-27b-dspark-moe-regen-mixture-v1-cont2ep-from-1ep-v3
# if B is interrupted: generate a restart recipe from its last complete checkpoint and launch it
/opt/sglang/bin/python scripts/launch/make_restart_recipe.py 5500 -b     # -> …-cont2ep-restart-step5500-b.yaml
bash scripts/launch/launch-moe-managed-local.sh qwen3.8-27b-dspark-moe-regen-mixture-v1-cont2ep-restart-step5500-b
```

The launcher refuses to start on a busy node, sets `PYTHONPATH=/opt/sglang-0518:<checkout>`, `MC_TRANSFER_TIMEOUT=300`,
`OMP_NUM_THREADS=32`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and runs everything under `numactl --preferred=1`
(capture schedulers bind themselves to NUMA node 0; the trainers' host memory must land on node 1). Output goes to
`outputs/<run id>/launch.log` (a `step N: {...}` line every 10 steps), `outputs/<run id>/control/logs/` (capture servers,
Mooncake) and `outputs/<run id>/<run id>-step{500,1000,...}`. A run with `max_steps` also writes a final checkpoint at its
last step. `scripts/launch/watch-moe-run.sh` (set `RUN`/`TOTAL` at the top) prints milestones, checkpoint completions and
errors, and removes the run's own oldest checkpoint before a save if the volume has less than 320 GB free.

Reproducing the exact same model requires the same three-phase schedule, including the two optimizer resets; an
uninterrupted `num_epochs: 3` run of the phase-A recipe is the cleaner experiment and needs about 90 hours plus 292 GB of free
disk per checkpoint save (rotation deletes only after the new save completes).

### 4.5 Export, serve, evaluate

```sh
# export (CPU, ~3 min, needs ~120 GB host RAM); the exporter writes no tokenizer, copy the target's
cd /personal/SpecForge-qwen38-moe && export PYTHONPATH=/opt/sglang-0518:$PWD CUDA_VISIBLE_DEVICES= HF_HOME=/cluster-storage/models
R=outputs/qwen3.8-27b-dspark-moe-regen-mixture-v1-cont2ep-restart-step5500-b; E=exports/qwen38-dspark-moe-3ep-cont-step9916
/opt/sglang/bin/python -m specforge.cli export --to hf --checkpoint $R/$(basename $R)-step4416 \
  --draft-config configs/qwen3.8-27b-dspark-moe-v2.json --output-dir $E
/opt/sglang/bin/python scripts/gates/normalize_dflash_export.py --config $E/config.json --block-size 7
/opt/sglang/bin/python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead', trust_remote_code=True).save_pretrained('$E')"
# config.json must now say architectures ["Qwen3MoeDSparkModel"], block_size 7, num_experts 512, moe_router_bias true

# serve on .103 (target + drafter, gamma 7); dense/official drafters via serve-drafters-on-103.sh (ports 30020/30021)
bash /personal/SpecForge-qwen38/outputs/launch/eval/serve-moe-drafter-on-103.sh 0 30030 /personal/SpecForge-qwen38-moe/$E

# evaluate (from any node that reaches .103); ~2.5 h for four servers
cd /personal/SpecForge-qwen38 && unset SGLANG_SIMULATE_ACC_LEN && PYTHONPATH=/opt/sglang-0518 /opt/sglang/bin/python \
  outputs/launch/eval/eval_drafters.py moe-3ep-step9916=http://10.13.114.103:30030 moe-1ep-step4958=http://10.13.114.103:30031 \
  dense-ep1-step2500=http://10.13.114.103:30020 official-v1=http://10.13.114.103:30021 \
  --datasets gsm8k,math500,mt_bench,aime26,traces --gsm8k-num-examples 1319 --thinking-key enable_thinking \
  --jsonl cache/dataset/qwen38_regen_mixture_v1_train.jsonl --traces-n 512 --concurrency 8 \
  --out outputs/launch/eval/results/<name>.json
```

The runner zeroes the server's speculative counters and flushes the radix cache before each (server, dataset) pair and reads
the forward-step-weighted aggregate accept length from `/server_info`; `drafts/step = aggregate - 1`.

## 5. Things that bit us

- Shared `/personal` volume with no per-user quota: a foreign writer filled it twice in one day and killed two runs during
  checkpoint saves (a save writes 292 GB of `.tmp` files before rotating). Check `df /personal` before launching and keep
  `max_checkpoints` low.
- No resume in `managed_local`: every restart is a warm start with optimizer and lr reset, and a fresh prompt seed is needed to
  avoid replaying the same prefix of the epoch.
- NUMA node 0 pressure: four capture schedulers at about 100 GB RSS each plus the trainers' pinned-memory spill leave node 0 at
  45-60 GB free during saves late in a run; 150 GiB Mooncake segments OOM-kill a scheduler, 100 GiB fit.
- From-scratch MoE routers collapse without input centering; the aux loss alone does not prevent it (fixed in this branch).
- `pgrep -f`/`pkill -f` self-match shells whose command line mentions the pattern; kill by PID or port.
