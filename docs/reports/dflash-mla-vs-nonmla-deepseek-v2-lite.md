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
| Data | ShareGPT (`sharegpt_400.jsonl`, 400 rows), `deepseek-v3` chat template |
| Schedule | 300 steps, batch 1, accum 1, max-len 512, lr 6e-4, seed 0, `mask_token_id` 100002 |
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

<!-- FILLED FROM curves.json -->
_(training curves below)_

| Draft | params | loss step 1 | loss step 300 | acc step 1 | acc step 300 (tail-10) |
|---|---|---|---|---|---|
| non-MLA (Qwen3 GQA) | TBD | TBD | TBD | TBD | TBD |
| MLA (DeepSeek) | TBD | TBD | TBD | TBD | TBD |

Per-step spot checks (loss / acc):

| step | non-MLA | MLA |
|---|---|---|
| 1 | TBD | TBD |
| 50 | TBD | TBD |
| 150 | TBD | TBD |
| 300 | TBD | TBD |

## Takeaways

TBD

## Reproduction

```bash
# both runs (GPU 0 = MLA, GPU 1 = non-MLA), colocated HF backend:
COMMON="--target-model-path deepseek-ai/DeepSeek-V2-Lite --target-model-backend hf \
  --train-data-path sharegpt_400.jsonl --chat-template deepseek-v3 \
  --max-length 512 --batch-size 1 --learning-rate 6e-4 --attention-backend sdpa \
  --seed 0 --num-epochs 1 --max-num-steps 300 --log-interval 1 \
  --save-interval 100000 --mask-token-id 100002"

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes 1 --nproc_per_node 1 --rdzv-endpoint localhost:29612 \
  scripts/train_dflash.py $COMMON \
  --draft-config-path configs/deepseek-v2-lite-dflash.json --output-dir out/mla

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes 1 --nproc_per_node 1 --rdzv-endpoint localhost:29613 \
  scripts/train_dflash.py $COMMON \
  --draft-config-path configs/deepseek-v2-lite-dflash-baseline.json --output-dir out/base
```
