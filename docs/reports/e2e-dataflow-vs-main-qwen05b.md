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

Experiment kit (launchers, patches, drivers, parser) lives at `/root/exp/expkit`
on the pod. Per-run shape:

```bash
# ours, any cell: topology/mode via env, native arg parsers untouched
EXP_MODE=online|offline EXP_TOPO=colo|disagg|disagg2p [EXP_ROLE=producer|consumer] \
EXP_PROMPTS_FROM=dump:<e3-hs>|jsonl:<df_tokens> \
CUDA_VISIBLE_DEVICES=<g> torchrun --rdzv-backend c10d --rdzv-endpoint localhost:<p> \
  --nnodes 1 --nproc_per_node 1 exp_{eagle3,dflash}_dataflow.py <shared args>

# main baselines: scripts/train_eagle3.py / scripts/train_dflash.py with the
# same shared args (+ print-only per-step CURVE patch, shuffle=False patch)
```
