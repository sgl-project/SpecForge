# Reproducing the Qwen3.8-27B DSpark **MoE** drafter run (one epoch, single node)

Self-contained notes for another session. Everything referenced lives on the shared `/personal` mount.

## 1. Code
- Worktree `/personal/SpecForge-qwen38-moe`, branch `kan/moe-3-qwen38` (this file's commit and its parents).
  Lineage: upstream `main` -> `kan/pr2-dsv4-dspark` (incl. #797 = 68e1820 prefetch/-800 retry) -> `kan/moe-1-skeleton`
  (PR #811 MoE FFN skeleton) -> `kan/moe-2-dsv4` (DeepSeek-V4 MoE) -> `kan/moe-3-qwen38`. Key commits here:
  `9026d75` qwen3_5_moe preset (softmax top-k renorm, sigmoid-gated shared expert, aux-loss controller, Qwen checkpoint
  naming), `71e94ce` router-collapse diagnostics + knobs, `372add6`/`b33f857` router centering (the fix that keeps 512
  experts in use), `2c7c122` fused pinned host round trip for the CPU-offloaded optimizer, `0f1f7ae`/`726d2b2` export
  folds the centering into `gate.bias`, `0899ceb` the one-epoch recipe.
- Optional (not used by the run): `/personal/SpecForge-qwen38-moe-dev` branch `kan/moe-4-fetchpool` (a2f6a1d) adds
  `SPECFORGE_MOONCAKE_FETCH_CLIENTS` for consumers whose features live on another node.

## 2. Environment (per node; /opt is node-local and is wiped when the pod is rebuilt)
- Interpreter `/opt/sglang/bin/python` (torch 2.13.0+cu130, transformers 5.12.1, mooncake). The image ships SGLang 0.5.19
  editable, which the spec-capture patch does not fit, so a patched **0.5.18 overlay** must precede the checkout on PYTHONPATH:
  ```sh
  /opt/sglang/bin/python -m pip install --no-deps --target /opt/sglang-0518 /personal/wheels/sglang-0.5.18-*.whl
  cd /personal/SpecForge-qwen38-moe && SPECFORGE_SGLANG_ROOT=/opt/sglang-0518 SPECFORGE_SGLANG_VERSION=0.5.18 \
    bash scripts/apply_sglang_spec_capture_patch.sh --target v0.5.18
  /opt/sglang/bin/python -m pip install --no-deps tensorboard yunchang
  export PYTHONPATH=/opt/sglang-0518:/personal/SpecForge-qwen38-moe     # verify: python -c 'import sglang; print(sglang.__version__)' -> 0.5.18
  ```
- HF cache `HF_HOME=/cluster-storage/models` (target `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead` is there). W&B login via `~/.netrc`.
- Hardware used: one node with 8x B200 (10.13.114.102), 1.3 TB host RAM in two NUMA nodes.

## 3. Data
- Corpus `RadixArk/Qwen3.8-27B-Regen-Mixture-v1` (1,269,290 conversations, xhigh reasoning), slimmed to `{id, conversations}` JSONL:
  `/personal/SpecForge-qwen38/cache/dataset/qwen38_regen_mixture_v1_train.jsonl` (40 GB); 1,140-row smoke slice next to it.
  Tokenized cache (`chat_template: qwen3.5`, max_length 8192): `/personal/SpecForge-qwen38/cache/qwen3.8-27b-regen-mixture-v1/`
  (rebuilt automatically in ~25 min if the cache key differs).

## 4. The run
- Draft config `configs/qwen3.8-27b-dspark-moe-v2.json`: dense DSpark skeleton of the official Qwen3.8 drafter (5 full-attention
  layers, hidden 5120, target layers 5/19/33/47/61, markov rank 256, mask 248070, **block_size 7**) + MoE FFN mirroring the
  Qwen3.8 MoE family at reduced width: 512 routed experts x 512 wide, top-10, one sigmoid-gated shared expert (2048),
  `moe_preset: qwen3_5_moe`, `moe_router_center: sample` (momentum 0.99), `moe_aux_loss_coeff: 0.001`, `moe_dispatch: grouped_mm`.
  20.8B total / 1.08B active parameters.
- Recipe `examples/configs/online/disaggregated/managed-local/qwen3.8-27b-dspark-moe-regen-mixture-v1-1ep-v3.yaml`:
  1 epoch = 2,479 optimizer steps at global batch 512 (4 ranks x batch_size 4 x accumulation 16), lr 5e-4 constant after 4%
  warmup, FULL_SHARD + `optimizer_cpu_offload`, dist_timeout 15 (minutes), 4 TP1 NVFP4 capture servers on GPUs 0-3 (triton
  attention, 2 running requests, 32k pool, 100 GiB Mooncake segment each) and the 4-rank trainer on GPUs 4-7, in-flight
  768/512, resident caps 300/200/380 GiB. W&B project `specforge-qwen38-ablation`.
- Launcher `scripts/launch/launch-moe-managed-local.sh <run_id>` (copy of outputs/launch/launch-moe-3ep-on-103.sh): refuses to
  start if a train process or busy GPU exists, and sets the env that matters:
  `PYTHONPATH=/opt/sglang-0518:<checkout>`, `MC_TRANSFER_TIMEOUT=300`, `SGLANG_SPEC_CAPTURE_SINK_CLIENTS=4`,
  `OMP_NUM_THREADS=32` (torchrun's default of 1 makes the CPU AdamW single-threaded), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  (without it bs4 ranks fragment to ~180 GB reserved with allocator retries), and runs the whole tree under
  `numactl --preferred=1` (capture schedulers self-bind to NUMA node 0; the trainers' ~130 GB/rank of host memory must land on node 1).
  ```sh
  cd /personal/SpecForge-qwen38-moe && bash scripts/launch/launch-moe-managed-local.sh qwen3.8-27b-dspark-moe-regen-mixture-v1-1ep-v3
  ```
  Output: `outputs/<run_id>/launch.log` (step lines every 10 steps incl. `train/moe/*` router metrics and `train/perf/*`),
  `control/logs/{capture-server-*,mooncake}.log`, checkpoints `<run_id>-step{500,1000,...}` (272 GB each, 3 kept) and a final one.
- Expected numbers (2026-09-08, .102): 22-23 s/step, peak 128 GB allocated per rank, data wait ~1 s, `experts_unused_frac` ~0,
  train/acc 0.55-0.57 at the end of the epoch; ~15.5 h total.
- Managed_local cannot resume (`training.resume_from` is rejected) and a resumed producer would re-capture the trained prefix, so
  treat runs as restart-from-scratch; stop a run only right after a `step N+10` line following a checkpoint (SIGTERM during a save
  leaves `*.pt.tmp` only).

## 5. Export, serve, evaluate
- Export on a node with >=120 GB free host RAM (CPU):
  ```sh
  CUDA_VISIBLE_DEVICES= python -m specforge.cli export --to hf --checkpoint outputs/<run>/<run>-latest \
    --draft-config configs/qwen3.8-27b-dspark-moe-v2.json --output-dir exports/<name>
  CUDA_VISIBLE_DEVICES= python scripts/gates/normalize_dflash_export.py --config exports/<name>/config.json --block-size 7
  # + copy the target tokenizer (AutoTokenizer.from_pretrained('RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead').save_pretrained(...))
  ```
  The normalizer emits `architectures: ["Qwen3MoeDSparkModel"]`, `num_experts`, `moe_router_bias`, and the centering folded into `gate.bias` (fp32).
- Serving needs the SGLang copy `/personal/sglang-0518-moe` (adds `sglang/srt/models/dspark_moe.py`, strict loader):
  `PYTHONPATH=/personal/sglang-0518-moe`, same flags as the dense drafter (`--speculative-algorithm DSPARK --speculative-dspark-block-size 7
  --speculative-draft-model-quantization unquant --attention-backend triton --speculative-draft-attention-backend triton ...`);
  script `/personal/SpecForge-qwen38/outputs/launch/eval/serve-moe-drafter-on-103.sh <gpu> <port> <export-dir>`.
- Eval runner (GSM8K, MATH500, MT-Bench, AIME26, training traces; accept length and accepted-drafts/step per server):
  `/personal/SpecForge-qwen38/outputs/launch/eval/eval_drafters.py` — see `/personal/SpecForge-qwen38/outputs/launch/eval-runbook.md`.
  Results: `/personal/SpecForge-qwen38/outputs/launch/eval/results/`.

## 6. Things that bit us (details in the Claude memory notes under /personal/claude-home/projects/-personal-SpecForge/memory/)
- 4 x 150 GiB Mooncake segments + scheduler baseline exhaust NUMA node 0 (685 GB) -> kernel OOM (schedulers are `bind:0`). 100 GiB segments fit.
- From-scratch MoE router collapses onto 10 experts because the router input has a token-independent common mode; centering fixes it, aux loss alone does not.
- Two-node layouts are fetch-bound: Mooncake TCP reads cap ~0.3-0.4 GiB/s per rank and ~3 GiB/s per host engine; inter-node RDMA fails on this fabric.
- `pkill/pgrep -f <name>` self-matches shells whose command text mentions the name; kill by PID or port.
