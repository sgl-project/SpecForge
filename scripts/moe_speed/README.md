# MoE drafter serving-speed check (few wide experts vs dense)

Decide whether a DSpark MoE drafter shape costs more per speculative step than
the official dense drafter BEFORE training it. Both drafters are served against
the same target with the accept length pinned (`SGLANG_SIMULATE_ACC_LEN`), so
tok/s differences are step-time differences. The MoE export has random expert
weights copied from the dense drafter's attention / projector / heads: routing
is uniform, which is the worst case for expert bytes.

```bash
# 1. random-expert MoE export from the official dense drafter (~3.9 GB bf16 for 16 x 1024)
python scripts/moe_speed/make_moe_draft_from_dense.py --src RadixArk/Qwen3.8-27B-DSpark \
  --out exports/qwen38-dspark-moe-16x1024-rand --experts 16 --width 1024 --topk 2 --shared 2048

# 2. dense vs MoE at a pinned accept length (one server at a time, same GPU, c = 1 / 8 / 32)
bash scripts/moe_speed/speed_test_moe_draft.sh Qwen/Qwen3.8-27B \
  dense=RadixArk/Qwen3.8-27B-DSpark \
  moe16x1024_k2=exports/qwen38-dspark-moe-16x1024-rand
```

Results land in `results/moe_speed/summary.tsv` (tok/s, mean E2E, TPOT per
concurrency 1 / 8 / 32). Needs stock SGLang >= 0.5.19 with this checkout on
`PYTHONPATH` (the serving classes come from `specforge/serving`, PR #941).
Without a tuned `E=16,N=1024` fused-MoE config SGLang logs "Using default MoE
kernel config"; `patches/sglang/moe_configs/tuning/tune_fused_moe_config.sh`
produces one (set the shape in `tuning/*/config.json`).

## GB300 results (2026-10-08, SGLang 0.5.20, accept length pinned to 5.0)

One GB300 (284 GB) per server, servers run one after another on the same GPU
with identical flags: DSPARK block 7, triton attention and MoE runner (default
kernel config), `--mem-fraction-static 0.85`, mamba cache 160, 32 max running
requests, drafts bf16 (`unquant`), 8 prompts per concurrency level x 512 input
/ 512 output tokens (`sglang.bench_serving`), `SGLANG_SIMULATE_ACC_LEN=5.0`.
Output tok/s; the delta is against the dense drafter at the same concurrency.
Kan's 512 x 512 top-10 drafter is the trained `RadixArk/qwen38-dspark-moe-3ep-cont-step9916`
export (served through the same package), the 16-expert rows are random-expert exports.

### bf16 target `Qwen/Qwen3.8-27B`

| drafter | FFN params / layer | active / token | c=1 | c=8 | c=32 |
|---|---|---|---|---|---|
| official dense v1 (17408 wide) | 267M | 267M | 326.7 | 1815.0 | 3752.4 |
| **MoE 16 x 1024 top-2, shared 2048** | 283M | 63M | 321.4 (-1.6%) | 1780.3 (-1.9%) | 3698.3 (-1.4%) |
| MoE 16 x 2048 top-2, shared 2048 | 535M | 94M | 319.9 (-2.1%) | 1763.5 (-2.8%) | 3660.7 (-2.4%) |
| Kan MoE 512 x 512 top-10, shared 2048 | 4.06B | 110M | 317.5 (-2.8%) | 1655.1 (-8.8%) | 3412.7 (-9.1%) |

### FP8 target `Qwen/Qwen3.8-27B-FP8`

| drafter | FFN params / layer | active / token | c=1 | c=8 | c=32 |
|---|---|---|---|---|---|
| official dense v1 (17408 wide) | 267M | 267M | 399.1 | 2143.0 | 4248.9 |
| **MoE 16 x 1024 top-2, shared 2048** | 283M | 63M | 391.4 (-1.9%) | 2094.8 (-2.2%) | 4206.0 (-1.0%) |
| MoE 16 x 2048 top-2, shared 2048 | 535M | 94M | 389.1 (-2.5%) | 2070.1 (-3.4%) | 4158.9 (-2.1%) |
| MoE 16 x 2048 top-4, shared 2048 | 535M | 157M | 386.2 (-3.2%) | 2041.0 (-4.8%) | 4122.5 (-3.0%) |
| Kan MoE 512 x 512 top-10, shared 2048 | 4.06B | 110M | 385.5 (-3.4%) | 1922.8 (-10.3%) | 3846.6 (-9.5%) |

Reading: with 16 experts a 7-token block x top-2 touches at most 14 experts, so
the per-step expert bytes stay close to one dense FFN (0.28-0.53 GB per layer vs
0.53 GB dense), and the drafter costs 1-2% of throughput on either target. Kan's
shape streams ~1 GB of expert weights per layer per step (67 of 512 experts
touched) and costs 9-10% at c >= 8. Doubling the expert width (16 x 2048) or the
top-k buys nothing at a pinned accept length and costs another 1-2%, so the
training run uses 16 x 1024 top-2.
