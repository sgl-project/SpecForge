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
| Data | Nemotron-Post-Training-v2 (`nemotron_train3k.jsonl`, 3000 rows train → 2887 samples; `nemotron_eval60.jsonl` held out for acceptance), `deepseek-v2` chat template (added — V2-Lite renders plain-text `User:`/`Assistant:`, not the `<｜Assistant｜>` tokens the `deepseek-v3` template expects) |
| Schedule | 4000 steps, batch 1, accum 1, max-len 2048, num-anchors 128, lr 6e-4, seed 0, `mask_token_id` 100002 |
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

## Results — training

Both drafts train through the identical `OnlineDFlashModel` spine on identical
captured DeepSeek-V2-Lite features, 4000 steps, same throughput
(**~0.13 s/iter** on both). Loss is per-step (batch 1) so it's noisy; tail-50
is the robust summary. The two track for ~2000 steps, then the GQA baseline
pulls ahead.

| Draft | params | loss (step 100 → tail-50) | acc (step 100 → tail-50) |
|---|---|---|---|
| non-MLA (`DFlashDraftModel`, Qwen3 GQA) | 264.7M | 12.0 → **5.63** | 0.027 → **0.143** |
| MLA (`DeepseekDFlashDraftModel`) | 255.6M | 11.8 → **6.49** | 0.015 → **0.075** |

Per-step accuracy over training (data-token acc): both climb off zero, but the
GQA draft's accuracy roughly doubles the MLA draft's by step 4000
(0.143 vs 0.075).

## Results — acceptance length

The metric that matters for speculative decoding: run the real DFlash accept
loop (draft proposes a block → target greedy-verifies → accept the matching
prefix + 1 bonus) on 25 held-out prompts, greedy, via one cache-free harness
for both drafts (`scripts/bench_dflash_acceptance.py`, adversarially reviewed).

| Draft | mean accept length | max | accept-length histogram (blocks) |
|---|---|---|---|
| non-MLA (Qwen3 GQA) | **1.478** | 5 | 1:1032 · 2:409 · 3:103 · 4:35 · 5:10 |
| MLA (DeepSeek) | **1.152** | 8 | 1:1791 · 2:173 · 3:64 · 8:1 |

Both **accept** on a real MLA target — the MLA draft is end-to-end functional in
speculative decoding (it proposes tokens the target verifies, occasionally a
full 8-token block). At this 4000-step budget the GQA baseline accepts more
(1.48 vs 1.15), tracking its higher training accuracy. Both are early/modest —
a converged DFlash reaches ~2–4; this is a controlled architecture comparison,
not a tuned draft.

## Takeaways

- The MLA DFlash draft is **correct and functional end-to-end** on a real MLA
  target (DeepSeek-V2-Lite): dataset → HF capture of layers [1,12,24] → MLA
  context+noise attention (sdpa) → DFlash loss → optimizer step → speculative
  decoding that **accepts tokens** (mean 1.15, up to 8/block), on H200.
- At equal 4000-step training the **standard Qwen3-GQA draft outperforms it**
  (acc 0.143 vs 0.075; accept 1.48 vs 1.15). The MLA geometry is not free here —
  whether that's fundamental or a tuning gap (LR, the YaRN mscale carried into
  the fresh draft, more steps) is the open question; the two track for the first
  ~2000 steps before diverging.
- Same EAGLE3 two-axis structure (`deepseek_eagle3.py` vs `llama3_eagle.py`):
  the DFlash *algorithm* is unchanged; only the draft *architecture* differs,
  resolved from the config through the draft registry.

**Bug caught & fixed during this run.** The MLA draft's first acceptance run
read exactly 1.0 (never accepted). Root cause: a hand-rolled `_init_weights` on
the draft ran *after* transformers' weight load (it ignored the
`_is_hf_initialized` guard), so `from_pretrained` silently returned a fresh-init
model — the benchmark had loaded a random draft. Fixed by basing the draft on
`DeepseekV3PreTrainedModel` and inheriting its guard-aware initializer;
`from_pretrained` now round-trips (verified: `fc.weight` norm matches the
checkpoint). This also un-breaks `--resume` for the MLA draft.

**Scope / follow-ups.** sdpa training path only (flex/fa need MLA-shaped
kernels); `spec_generate` (DynamicCache decode) not yet ported (the benchmark is
cache-free); a longer/tuned run + a live-sglang
acceptance gate are the natural next steps.

## Reproduction

Correctness gates first: `python -m pytest tests/test_runtime/test_mla_draft.py
tests/test_runtime/test_dflash_mla.py` (GPU).

```bash
# TRAIN both (GPU 0 = MLA, GPU 1 = non-MLA), colocated HF backend, DeepSeek-V2-Lite
# is native in transformers 5.x so no --trust-remote-code needed:
COMMON="--target-model-path deepseek-ai/DeepSeek-V2-Lite --target-model-backend hf \
  --train-data-path nemotron_train3k.jsonl --chat-template deepseek-v2 \
  --max-length 2048 --batch-size 1 --learning-rate 6e-4 --attention-backend sdpa \
  --num-anchors 128 --seed 0 --num-epochs 2 --max-num-steps 4000 --log-interval 5 \
  --save-interval 100000 --mask-token-id 100002"

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc_per_node 1 \
  scripts/train_dflash.py $COMMON \
  --draft-config-path configs/deepseek-v2-lite-dflash.json --output-dir out/mla_long

CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nnodes 1 --nproc_per_node 1 \
  scripts/train_dflash.py $COMMON --cache-dir cache_base \
  --draft-config-path configs/deepseek-v2-lite-dflash-baseline.json --output-dir out/base_long

# ACCEPTANCE-LENGTH benchmark (held-out prompts), one harness for both drafts:
for d in mla base; do
  python scripts/bench_dflash_acceptance.py \
    --target-model-path deepseek-ai/DeepSeek-V2-Lite \
    --draft-checkpoint out/${d}_long/epoch_2_step_4000 \
    --eval-data-path nemotron_eval60.jsonl --chat-template deepseek-v2 \
    --num-prompts 25 --max-new-tokens 96 --mask-token-id 100002 \
    --json-out out/accept_${d}.json
done
```

> `mask_token_id 100002` is a valid embedding row that the V2-Lite tokenizer
> never emits (its ids stop at 100001). The `deepseek-v2` chat template is
> required — V2-Lite renders plain-text `User:`/`Assistant:`, which the
> `deepseek-v3` template's `<｜Assistant｜>` header does not match (→ empty loss
> masks → `total_steps 0`). The draft's own modeling file is copied into each
> checkpoint; the raw acceptance JSONs are in
> [`data/`](data/dflash-mla-accept_mla.json).
