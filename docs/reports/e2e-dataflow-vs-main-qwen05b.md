# E2E training comparison: DataFlow runtime (PR #645 stack) vs `main`

100-step draft-training curve comparison across the full
`{eagle3, dflash} × {online, offline} × {colocated, disaggregated}` matrix on
the PR #645 stack head (`75f9702`), against `main` (`357a97e`) as the baseline.
Raw per-step losses: [`data/e2e-curves-qwen05b.json`](data/e2e-curves-qwen05b.json).

**Axes.** *online* = the target model runs inference during the job and features
stream to the trainer; *offline* = features precomputed to disk, trainer reads
them. *colocated* = inference and trainer in one process/GPU; *disaggregated* =
rollout/inference pool and trainer pool are separate, sharing only a feature
store + tensor-free control plane.

## Setup

| | |
|---|---|
| Target | `Qwen/Qwen2.5-0.5B-Instruct` (HF backend for capture everywhere) |
| Drafts | 1-layer LlamaForCausalLMEagle3 (`draft_vocab_size` 16000) / 5-layer DFlashDraftModel (block 16, `target_layer_ids` [1,6,11,16,21], `mask_token_id` 151669) |
| Data | ShareGPT (`prepare_data.py --dataset sharegpt`), filtered to 150 rows that **both** branches' renderers tokenize identically (see findings) |
| Schedule | 100 steps, batch 1, accum 1, max-len 512, seed 0; eagle3 lr 1e-4 / ttt 7 / `--total-steps 10000`; dflash lr 6e-4 / `total_steps` pinned to 1000 |
| Hardware | 1 GPU per run (8×H200 pod), runs step-aligned: train-loader shuffling disabled identically in both trees, per-step loss printed from both |
| Offline features | eagle3: HF-forward dump (aux layers [1,11,20] + post-norm last hidden), 130 samples, consumed by main-offline AND ours-offline; dflash: capture dump over the same tokens |
| Token provenance | all "ours" runs consume **main-rendered tokens** (pre-tokenized prompts from the dumps / a main-tree token jsonl), isolating runtime behavior from the branches' data-pipeline divergence |

Correctness gate before the e2e runs: `tests/test_runtime` — 293 OK at PR #643,
299 OK at the #645 stack head (2 skip, 1 xfail).

## Results

Loss = mean of steps 91–100; per-step Δ measured at the same step index
(the 8 zero-loss-mask rows land on identical steps in every aligned run and are
excluded from relative-Δ stats).

### Ours vs `main`

| Family | main | ours colocated | ours disagg | per-step Δ vs main |
|---|---|---|---|---|
| DFlash online | 7.9990 | 7.9990 | 7.9990 | **0.000% — bit-exact, all 100 steps** |
| DFlash offline | — (main has no offline dflash) | 7.9990 | 7.9990 | **0.000% vs main-online — bit-exact** |
| EAGLE3 online | 21.639 | 19.447 | 19.447 | mean 4.6% (per-step pairs ~1–2%) |
| EAGLE3 offline | 21.536 | 19.342 | 19.342 | mean 4.7%, same shape |

EAGLE3 per-step pairs (main / ours): step 1 `19.19/18.88`, step 5 `29.48/29.21`,
step 20 `33.86/33.49`, step 50 `21.48/21.33`, step 80 `26.94/26.84` — parallel
curves with identical zero-mask steps `[16, 32, 65, 72, 75, 79, 92, 100]`. The
~4.6% offset is provenance the runtime does not control: draft-init RNG
consumption order differs between the two scripts, and each branch builds its
16k draft-vocab mapping from its own render of the corpus. DFlash has neither
factor (identical model assembly order, no vocab mapping) and is bit-exact.

### Disaggregated vs colocated (topology invariance)

Single-process disagg (shared-dir store + streaming ref channel) **and** true
two-process placement (producer/inference and consumer/trainer as separate OS
processes on different GPUs; online pairs running concurrently; shared
`SharedDirFeatureStore` + ref-channel file + SQLite metadata store):

| Family | colocated | disagg (1-proc) | disagg (2-proc, 2 GPUs) | max per-step \|Δ\| |
|---|---|---|---|---|
| EAGLE3 online | 19.4471 | 19.4471 | 19.4471 | **0.0 over 100 steps** |
| EAGLE3 offline | 19.3423 | 19.3423 | 19.3423 | **0.0** |
| DFlash online | 7.9990 | 7.9990 | 7.9990 | **0.0** |
| DFlash offline | 7.9990 | 7.9990 | 7.9990 | **0.0** |

Spot-check (identical in all three placements, every step): dflash steps
1/25/50/75/100 = `12.415770 / 8.932111 / 8.614070 / 8.138462 / 7.741909`;
eagle3-online steps 1/25/50 = `18.881990 / 29.006533 / 21.326193`. DFlash
online ≡ offline is also bit-exact (offline features are the online capture's
own dump).

## Findings (pre-existing, follow-ups filed)

1. **Branch data-pipeline divergence.** The stack forked before main's
   conversation-rendering updates; the branch's eagle3 preprocessing renders
   some ShareGPT rows as ~11-token zero-mask stubs **dependent on dataset batch
   composition** (the same jsonl line renders fully in one file and as a stub in
   another). Fix = reconcile `specforge/data/` with main. This is why the
   comparison feeds main-rendered tokens to both sides.
2. **DFlash × interleaved disagg thread-unsafety.** The single-process
   interleaved runner forwards the target in a producer *thread*; DFlash's HF
   capture (`output_hidden_states=True`) trips transformers' non-thread-safe
   hook wrapper while the trainer forwards the draft. EAGLE3 (explicit hooks) is
   unaffected, and the two-process runs show DFlash is fine across *processes* —
   the fix is a capture lock or a hook-based DFlash HF capture.
3. **`TargetHead.load_weights`** requires a sharded `*.index.json` with an
   explicit `lm_head.weight`; single-file tied-embedding checkpoints
   (Qwen2.5-0.5B) need an untied sharded copy (workaround used here).
4. Reproduction gotchas: HF datasets / tree caches key on file *path*, not
   content; loader shuffling draws global RNG at iteration time (disable in both
   trees for step alignment); pin `total_steps` so LR schedules match.

## Reproduction

The full experiment kit (launchers, dumpers, patch scripts, drivers, parser,
draft configs) is committed next to this report in
[`scripts/e2e-qwen05b/`](scripts/e2e-qwen05b/). Layout on the test node:
`$EXP=/root/exp`, `$EXP/sf-main` = main checkout, `$EXP/sf-ours` = this PR's
head; launchers are copied into each tree's `scripts/` so they can reuse the
native model/data builders. Every run is
`CUDA_VISIBLE_DEVICES=<gpu> torchrun --rdzv-backend c10d --rdzv-endpoint
localhost:<port> --nnodes 1 --nproc_per_node 1 <script> <args>` with
`PYTHONPATH=<tree>` and `FLASHINFER_DISABLE_VERSION_CHECK=1`.

### 0. One-time setup

```bash
# experiment-only patches, applied IDENTICALLY where noted so they cancel out
python patch_main.py      $EXP/sf-main   # per-step "CURVE step= loss=" prints; EXP_MAX_STEPS/EXP_TOTAL_STEPS for dflash (main has no --max-num-steps there)
python patch_noshuffle.py $EXP/sf-main   # train-loader shuffle=False (step alignment)
python patch_noshuffle.py $EXP/sf-ours

# model + raw data
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')"
cd $EXP/sf-main && python scripts/prepare_data.py --dataset sharegpt --output-path $EXP/data --sample-size 3000

# untied sharded model copy for the OFFLINE eagle3 runs (TargetHead needs an
# explicit lm_head.weight in a *.index.json; Qwen2.5-0.5B is single-file + tied)
python - <<'PY'
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
src, dst = 'Qwen/Qwen2.5-0.5B-Instruct', '/root/exp/models/qwen2.5-0.5b-untied'
m = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.bfloat16)
w = m.get_output_embeddings().weight
head = nn.Linear(w.shape[1], w.shape[0], bias=False); head.weight = nn.Parameter(w.detach().clone())
m.config.tie_word_embeddings = False; m.set_output_embeddings(head)
m.save_pretrained(dst, max_shard_size='300MB'); AutoTokenizer.from_pretrained(src).save_pretrained(dst)
PY

# keep only jsonl rows BOTH branches render identically (see finding 1);
# run render_stats.py once per tree, then intersect
cd $EXP/sf-ours/scripts && EXP_STATS_OUT=$EXP/data/stats_ours.json PYTHONPATH=$EXP/sf-ours \
  CUDA_VISIBLE_DEVICES=0 torchrun --rdzv-backend c10d --rdzv-endpoint localhost:29430 --nnodes 1 --nproc_per_node 1 \
  render_stats.py --target-model-path Qwen/Qwen2.5-0.5B-Instruct \
  --draft-model-config $EXP/configs/qwen2.5-0.5b-eagle3.json \
  --train-data-path $EXP/data/sharegpt_train.jsonl --chat-template qwen \
  --max-length 512 --batch-size 1 --seed 0 --num-epochs 1 --output-dir $EXP/out/stats
cd $EXP/sf-main/scripts && EXP_STATS_OUT=$EXP/data/stats_main.json PYTHONPATH=$EXP/sf-main \
  CUDA_VISIBLE_DEVICES=0 torchrun ...same args... render_stats.py ...
python intersect_filter.py   # -> $EXP/data/sharegpt_final.jsonl (150 rows)
```

> Caching gotcha: HF `datasets` and the trees' `./cache` key on the data file
> *path*, not content — when regenerating the jsonl, use a NEW filename and
> `rm -rf ~/.cache/huggingface/datasets <tree>/cache <tree>/scripts/cache`.

### 1. Shared feature dumps

```bash
# EAGLE3 offline hidden states (HF forward; consumed by main-offline AND ours-offline)
cd $EXP/sf-main/scripts
EXP_HS_PATH=$EXP/dumps/e3-hs EXP_NUM_SAMPLES=130 PYTHONPATH=$EXP/sf-main \
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv-backend c10d --rdzv-endpoint localhost:29431 --nnodes 1 --nproc_per_node 1 \
  exp_dump_e3_hs.py --target-model-path Qwen/Qwen2.5-0.5B-Instruct \
  --draft-model-config $EXP/configs/qwen2.5-0.5b-eagle3.json \
  --train-data-path $EXP/data/sharegpt_final.jsonl --chat-template qwen \
  --max-length 512 --batch-size 1 --seed 0 --num-epochs 1 --output-dir $EXP/out/e3-dump

# main's dflash token render (so ours consumes identical tokens)
EXP_TOKENS_OUT=$EXP/data/df_tokens.jsonl PYTHONPATH=$EXP/sf-main \
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv-backend c10d --rdzv-endpoint localhost:29440 --nnodes 1 --nproc_per_node 1 \
  dump_df_tokens.py $DF_ARGS --output-dir $EXP/out/df-tokens

# dflash offline features from those tokens (ours capture)
cd $EXP/sf-ours/scripts
EXP_MODE=dump EXP_HS_PATH=$EXP/dumps/dflash-hs EXP_MAX_PROMPTS=130 \
EXP_PROMPTS_FROM=jsonl:$EXP/data/df_tokens.jsonl PYTHONPATH=$EXP/sf-ours \
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv-backend c10d --rdzv-endpoint localhost:29441 --nnodes 1 --nproc_per_node 1 \
  exp_dflash_dataflow.py $DF_ARGS --output-dir $EXP/out/df-dump
```

### 2. The 11-run matrix (`run_rerun7.sh` runs these in GPU-parallel waves)

```bash
E3_ARGS="--target-model-path Qwen/Qwen2.5-0.5B-Instruct \
  --draft-model-config $EXP/configs/qwen2.5-0.5b-eagle3.json \
  --train-data-path $EXP/data/sharegpt_final.jsonl \
  --chat-template qwen --max-length 512 --batch-size 1 --learning-rate 1e-4 \
  --ttt-length 7 --seed 0 --num-epochs 1 --max-num-steps 100 --total-steps 10000 \
  --target-model-backend hf"
DF_ARGS="--target-model-path Qwen/Qwen2.5-0.5B-Instruct \
  --draft-config-path $EXP/configs/qwen2.5-0.5b-dflash.json \
  --train-data-path $EXP/data/sharegpt_final.jsonl \
  --chat-template qwen --max-length 512 --batch-size 1 --learning-rate 6e-4 \
  --seed 0 --num-epochs 1 --mask-token-id 151669 --target-model-backend hf \
  --log-interval 1 --save-interval 1000000"
UNTIED=/root/exp/models/qwen2.5-0.5b-untied

# --- main baselines (cwd $EXP/sf-main, PYTHONPATH=$EXP/sf-main) ---
scripts/train_eagle3.py $E3_ARGS --output-dir $EXP/out/main-e3-online
scripts/train_eagle3.py $E3_ARGS --target-model-path $UNTIED \
  --train-hidden-states-path $EXP/dumps/e3-hs --output-dir $EXP/out/main-e3-offline
EXP_MAX_STEPS=100 EXP_TOTAL_STEPS=1000 \
  scripts/train_dflash.py $DF_ARGS --output-dir $EXP/out/main-df-online
# (main has no offline dflash mode)

# --- ours (cwd $EXP/sf-ours/scripts, PYTHONPATH=$EXP/sf-ours) ---
# eagle3 online: colocate / disagg
EXP_TOPO=colo   EXP_MAX_PROMPTS=130 EXP_PROMPTS_FROM=dump:$EXP/dumps/e3-hs \
  exp_eagle3_dataflow.py $E3_ARGS --output-dir $EXP/out/ours-e3-online-colo
EXP_TOPO=disagg EXP_MAX_PROMPTS=130 EXP_PROMPTS_FROM=dump:$EXP/dumps/e3-hs \
  EXP_STORE_ROOT=$EXP/store \
  exp_eagle3_dataflow.py $E3_ARGS --output-dir $EXP/out/ours-e3-online-disagg
# eagle3 offline: colocate / disagg
EXP_TOPO=colo   exp_eagle3_dataflow.py $E3_ARGS --target-model-path $UNTIED \
  --train-hidden-states-path $EXP/dumps/e3-hs --output-dir $EXP/out/ours-e3-offline-colo
EXP_TOPO=disagg EXP_STORE_ROOT=$EXP/store \
  exp_eagle3_dataflow.py $E3_ARGS --target-model-path $UNTIED \
  --train-hidden-states-path $EXP/dumps/e3-hs --output-dir $EXP/out/ours-e3-offline-disagg
# dflash online: colocate / disagg
EXP_MODE=online EXP_TOPO=colo   EXP_MAX_PROMPTS=130 EXP_MAX_STEPS=100 EXP_TOTAL_STEPS=1000 \
  EXP_PROMPTS_FROM=jsonl:$EXP/data/df_tokens.jsonl \
  exp_dflash_dataflow.py $DF_ARGS --output-dir $EXP/out/ours-df-online-colo
EXP_MODE=online EXP_TOPO=disagg EXP_MAX_PROMPTS=130 EXP_MAX_STEPS=100 EXP_TOTAL_STEPS=1000 \
  EXP_PROMPTS_FROM=jsonl:$EXP/data/df_tokens.jsonl EXP_STORE_ROOT=$EXP/store \
  exp_dflash_dataflow.py $DF_ARGS --output-dir $EXP/out/ours-df-online-disagg
# dflash offline: colocate / disagg
EXP_MODE=offline EXP_TOPO=colo   EXP_HS_PATH=$EXP/dumps/dflash-hs EXP_MAX_STEPS=100 EXP_TOTAL_STEPS=1000 \
  EXP_PROMPTS_FROM=jsonl:$EXP/data/df_tokens.jsonl \
  exp_dflash_dataflow.py $DF_ARGS --output-dir $EXP/out/ours-df-offline-colo
EXP_MODE=offline EXP_TOPO=disagg EXP_HS_PATH=$EXP/dumps/dflash-hs EXP_MAX_STEPS=100 EXP_TOTAL_STEPS=1000 \
  EXP_PROMPTS_FROM=jsonl:$EXP/data/df_tokens.jsonl EXP_STORE_ROOT=$EXP/store \
  exp_dflash_dataflow.py $DF_ARGS --output-dir $EXP/out/ours-df-offline-disagg
```

### 3. Two-process disagg placement (`run_2proc.sh`)

```bash
# eagle3 online: inference pool (GPU0) || trainer pool (GPU1), CONCURRENT
EXP_TOPO=disagg2p EXP_ROLE=producer EXP_STORE_ROOT=$EXP/store2p EXP_DB=$EXP/store2p/e3-online.db \
  EXP_MAX_PROMPTS=130 EXP_PROMPTS_FROM=dump:$EXP/dumps/e3-hs \
  CUDA_VISIBLE_DEVICES=0 torchrun ... exp_eagle3_dataflow.py $E3_ARGS --output-dir $EXP/out/e3-2p-prod
EXP_TOPO=disagg2p EXP_ROLE=consumer EXP_STORE_ROOT=$EXP/store2p EXP_DB=$EXP/store2p/e3-online.db \
  CUDA_VISIBLE_DEVICES=1 torchrun ... exp_eagle3_dataflow.py $E3_ARGS --output-dir $EXP/out/e3-2p-cons

# dflash online: same pattern on GPU2/GPU3 (concurrent across processes)
EXP_MODE=online EXP_TOPO=disagg2p EXP_ROLE=producer EXP_STORE_ROOT=$EXP/store2p EXP_DB=$EXP/store2p/df-online.db \
  EXP_MAX_PROMPTS=130 EXP_MAX_STEPS=100 EXP_TOTAL_STEPS=1000 EXP_PROMPTS_FROM=jsonl:$EXP/data/df_tokens.jsonl \
  CUDA_VISIBLE_DEVICES=2 torchrun ... exp_dflash_dataflow.py $DF_ARGS --output-dir $EXP/out/df-2p-prod
EXP_MODE=online EXP_TOPO=disagg2p EXP_ROLE=consumer EXP_STORE_ROOT=$EXP/store2p EXP_DB=$EXP/store2p/df-online.db \
  EXP_MAX_STEPS=100 EXP_TOTAL_STEPS=1000 EXP_PROMPTS_FROM=jsonl:$EXP/data/df_tokens.jsonl \
  CUDA_VISIBLE_DEVICES=3 torchrun ... exp_dflash_dataflow.py $DF_ARGS --output-dir $EXP/out/df-2p-cons

# offline (both algos): producer PROCESS ingests dump -> store + ref manifest,
# then consumer PROCESS trains from it (same env pattern, EXP_ROLE=producer then consumer)
```

### 4. Collect

```bash
python parse_curves.py /root/exp/logs /root/exp/curves.json   # greps "CURVE step= loss="
```

Correctness gate (run before the e2e matrix):

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 PYTHONPATH=$PWD \
  python -m unittest discover -s tests/test_runtime -p 'test_*.py'
```
