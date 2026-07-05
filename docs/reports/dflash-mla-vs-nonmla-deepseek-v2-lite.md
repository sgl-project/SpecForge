# DFlash draft-architecture comparison: MLA vs non-MLA on DeepSeek-V2-Lite

Adds the MLA (DeepSeek/Multi-head Latent Attention) DFlash draft
(`deepseek_dflash.py`) as the draft-architecture counterpart to the Qwen3-style
DFlash draft (`dflash.py`) — the same split that `deepseek_eagle3.py` is to
`llama3_eagle.py` for EAGLE3. This report trains **both** DFlash drafts against
the **same** DeepSeek-V2-Lite target so the only variable is the draft's
attention block.

Companion to [`e2e-dataflow-vs-main-qwen05b.md`](e2e-dataflow-vs-main-qwen05b.md)
(that report compares runtimes on a Qwen target; this one compares draft
architectures on an MLA target).

## What "MLA DFlash" is

DFlash's draft attention is block-parallel context+noise: the query stream is
per-block noise embeddings and the keys/values are `cat(context, noise)` along
the sequence axis. The non-MLA draft realizes this with Qwen3 GQA
(`Qwen3DFlashAttention`). The MLA draft (`DeepseekDFlashAttention`) keeps the
context+noise contract byte-for-byte and swaps only the projection geometry:
compressed KV via `kv_a`/`kv_b` LoRA, split nope/rope head dims, DeepSeek
interleaved-pair RoPE (query at block positions, key at the full
`cat(context, noise)` positions), and a YaRN-aware softmax scale. It runs the
sdpa training path only (flex/fa need MLA-shaped kernels).

The DFlash training wrapper (`OnlineDFlashModel`), loss, anchor sampling, and
capture surface are all unchanged — the draft is resolved from the config's
`architectures` through the draft registry, so the two runs share every knob
except the draft class.

## Setup

| | |
|---|---|
| Target | `deepseek-ai/DeepSeek-V2-Lite` (16 heads, kv_lora 512, qk_nope 128, qk_rope 64, v_head 128, 27 layers, YaRN; loaded **native** in transformers 5.12.1, HF backend) |
| MLA draft | `DeepseekDFlashDraftModel`, 3 layers, block 8, `target_layer_ids` [1,12,24], sdpa — `configs/deepseek-v2-lite-dflash.json` |
| non-MLA draft | `DFlashDraftModel` (Qwen3 GQA), 3 layers, block 8, `target_layer_ids` [1,12,24], sdpa — `configs/deepseek-v2-lite-dflash-baseline.json` |
| Data | Nemotron-Post-Training-v2 (`nemotron_400.jsonl`, 400 rows), `deepseek-v2` chat template (added — V2-Lite renders plain-text `User:`/`Assistant:`, not the `<｜Assistant｜>` tokens the `deepseek-v3` template expects) |
| Schedule | 300 steps, batch 1, accum 1, max-len 2048, num-anchors 128, lr 6e-4, seed 0, `mask_token_id` 100002 |
| Hardware | 1×H200 per run (sci-h200 pod), colocated (target capture + draft train in one process) |
| Capture | identical for both: concat of target hidden states at layers [1,12,24] → fc(6144→2048) |

Both drafts consume the **same** captured DeepSeek-V2-Lite features (identical
`set_capture_layers`), so any curve difference is the draft attention alone.

## Correctness gates (before the runs)

`tests/test_runtime`, GPU (H200), transformers 5.12.1:

- `test_mla_draft.py` (EAGLE3 MLA, rebased onto #645): **PASS** — suffix-cache ≡
  causal at step 0; Auto* mapping resolves deepseek_v3; 3-step train smoke
  through the unchanged `Eagle3TrainStrategy`.
- `test_dflash_mla.py` (DFlash MLA, new): **PASS** — registry resolution;
  attention shapes/grads for both q_lora branches; `OnlineDFlashModel` train
  smoke (finite loss, accuracy in [0,1], trainable draft grads).
- `test_dflash_launch.py`, `test_dflash_online_launch.py` (non-MLA regression):
  **PASS**.

## Results

Both drafts train through the identical `OnlineDFlashModel` spine on identical
captured DeepSeek-V2-Lite features. 382 trainable samples, 300 steps, same
throughput (**0.170 s/iter** on both). Loss is per-step (batch 1) so it is
noisy; the mean over a 20-step window is the robust summary.

| Draft | params | loss step 1 | loss head-20 | loss tail-20 | acc tail-20 |
|---|---|---|---|---|---|
| non-MLA (`DFlashDraftModel`, Qwen3 GQA) | 264.7M | 33.49 | 22.96 | **8.65** | 0.047 |
| MLA (`DeepseekDFlashDraftModel`) | 255.6M | 31.33 | 21.77 | **8.92** | 0.043 |

Both fall from ~22 (head-20 mean) to ~8.8 (tail-20 mean) over 300 steps — the
MLA draft learns from real MLA-target features at the same rate as the standard
draft. Per-step spot checks (loss / acc):

| step | non-MLA (Qwen3 GQA) | MLA (DeepSeek) |
|---|---|---|
| 1 | 33.49 / 0.000 | 31.33 / 0.000 |
| 50 | 13.68 / 0.002 | 13.52 / 0.002 |
| 100 | 7.38 / 0.039 | 7.35 / 0.011 |
| 150 | 8.77 / 0.051 | 8.65 / 0.057 |
| 200 | 9.79 / 0.017 | 10.45 / 0.038 |
| 300 | 6.75 / 0.021 | 7.51 / 0.050 |

The two curves interleave step-for-step (neither is consistently above the
other), which is what "same algorithm, different draft attention" should look
like: the draft architecture is not the bottleneck at this scale/step budget.

## Takeaways

- The MLA DFlash draft is **correct and trainable end-to-end** on a real MLA
  target (DeepSeek-V2-Lite): dataset → HF capture of layers [1,12,24] → MLA
  context+noise attention (sdpa) → DFlash loss → optimizer step, on H200.
- It **matches the standard Qwen3-GQA DFlash draft** in loss trajectory and
  throughput on identical features — the compressed-KV / split-rope geometry
  costs nothing here and slightly *reduces* parameters (255.6M vs 264.7M).
- This mirrors the EAGLE3 two-axis result (`deepseek_eagle3.py` vs
  `llama3_eagle.py`): the DFlash *algorithm* is unchanged; only the draft
  *architecture* differs, resolved from the config through the draft registry.

**Scope / follow-ups.** sdpa training path only (flex/fa need MLA-shaped
kernels); `spec_generate` (DynamicCache decode) not yet ported; 300-step
demonstration run (not a converged draft); a longer run + a live-sglang
acceptance gate are the natural next steps. Absolute accuracy is low because
this is an early, short, batch-1 run — the loss trend, not the accuracy value,
is the signal here.

## Reproduction

Correctness gates first: `python -m pytest tests/test_runtime/test_mla_draft.py
tests/test_runtime/test_dflash_mla.py` (GPU).

```bash
# both runs (GPU 0 = MLA, GPU 1 = non-MLA), colocated HF backend, DeepSeek-V2-Lite
# is native in transformers 5.x so no --trust-remote-code needed:
COMMON="--target-model-path deepseek-ai/DeepSeek-V2-Lite --target-model-backend hf \
  --train-data-path nemotron_400.jsonl --chat-template deepseek-v2 \
  --max-length 2048 --batch-size 1 --learning-rate 6e-4 --attention-backend sdpa \
  --num-anchors 128 --seed 0 --num-epochs 2 --max-num-steps 300 --log-interval 1 \
  --save-interval 100000 --mask-token-id 100002"

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc_per_node 1 \
  scripts/train_dflash.py $COMMON \
  --draft-config-path configs/deepseek-v2-lite-dflash.json --output-dir out/mla

CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nnodes 1 --nproc_per_node 1 \
  scripts/train_dflash.py $COMMON \
  --draft-config-path configs/deepseek-v2-lite-dflash-baseline.json --output-dir out/base
```

> `mask_token_id 100002` is a valid embedding row that the V2-Lite tokenizer
> never emits (its ids stop at 100001). The `deepseek-v2` chat template is
> required — V2-Lite renders plain-text `User:`/`Assistant:`, which the
> `deepseek-v3` template's `<｜Assistant｜>` header does not match (→ empty loss
> masks → `total_steps 0`).
