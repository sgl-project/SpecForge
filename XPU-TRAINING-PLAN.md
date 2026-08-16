# Qwen3.8-27B drafter training on B70 (XPU) — status & plan

## Done (no GPU used)
1. SpecForge cloned at /home/ryan/specforge.
2. XPU device port applied (5 files, py_compile clean, config loads under vLLM image py3.12):
   - specforge/utils.py            get_device_type/get_local_device: +xpu
   - specforge/distributed.py      backends map: +xpu -> xccl
   - specforge/launch_plan.py      visibility env: +ZE_AFFINITY_MASK
   - specforge/training/backend.py DDP device_ids + RNG state: +xpu
   - specforge/training/trainer.py pin_memory: +xpu
3. configs/qwen3.8-27b-dspark.json copied from the released drafter
   (block_size 7, target_layer_ids [4,16,28,40,52], mask 248077, markov rank 256).
4. examples/configs/qwen3.8-27b-dspark-offline-xpu.yaml written + schema-validated
   (strategy dspark, attention_backend sdpa — flex_attention is CUDA-only).

## Remaining gates (need GPU, in order)
1. Backward smoke test: tiny Linear on xpu, forward/backward/AdamW.step — verify
   autograd + xccl init actually work on this torch 2.11.0+xpu build (~2 min GPU).
2. Capture stage: SpecForge capture is sglang-only (target_backend literal).
   No sglang-XPU image exists on this box yet; llm-scaler sglang/ has the BMG
   patches but no image built. Options: build llm-scaler-sgl:bmg, or write a
   small vLLM hidden-state capture script (vLLM XPU already proven here).
   Output: ./cache/hidden_states/qwen3.8-27b-dspark (~5 layers x 5120 hidden
   x bf16 per token — budget ~50 GB per 1M tokens; disk has ~900 GB).
3. Train: SPECFORGE_DEVICE=xpu specforge train --config
   examples/configs/qwen3.8-27b-dspark-offline-xpu.yaml — single GPU suffices
   (1.36B drafter, target not loaded). Expect hours per run; checkpoint every
   16 steps per config given the xe-reset history on this box.
4. Export via specforge/export/to_hf.py, drop into ~/models/, re-bench dflash
   acceptance vs the 23% off-shelf baseline.

## Why this matters
Off-shelf RadixArk drafter gets ~23% acceptance against our FP8 target
(mismatched training target). A drafter trained on THIS target is the
FP8-quality path past 60 tok/s C1 (INT4 already hits 74.7 but is off the
table per Ryan).

## Addendum 02:39:36Z — XPU_GRAPH rule
Confirmed by claude: VLLM_XPU_ENABLE_XPU_GRAPH=1 hangs xe engines on this
build (bcs reset / CAT error / DEVICE_LOST, persisted across a reboot).
RULE: keep XPU_GRAPH=0 for every serve/capture/training-adjacent run on this
box. Applies to the future capture stage too.

## 2026-08-16T02:56Z — OMP: XPU training capability PROVEN (no training run yet)
Per Ryan: full GPU access granted; stale locks and GPU containers cleared.
Results (single-GPU probes, card0 only):
- Backward/AdamW smoke on B70: PASS (bf16, loss decreasing, 20 steps in 0.29s).
- DSpark drafter (1.36B, released qwen3.8-27b-dspark) fwd+bwd+AdamW on xpu:0:
  PASS, 3 steps, loss 1.254->1.197, grads on all backbone params. Forward
  contract note: position_ids must span ctx+block (rotary covers k = ctx+q).
- Trainer deps in b3 image: OK (datasets/pydantic/accelerate/transformers;
  wandb absent -> report_to: tensorboard|none). specforge.cli imports clean.
- Port applied in /home/ryan/specforge (utils/distributed/launch_plan/backend/
  trainer: xpu detection, xccl backend, ZE_AFFINITY_MASK visibility, DDP/RNG/
  pin_memory branches), py_compile clean; offline XPU YAML schema-validated.
Full plan + evidence: /home/ryan/specforge/XPU-TRAINING-PLAN.md
NOT starting capture/training yet — awaiting Ryan go-ahead. GPU released.
