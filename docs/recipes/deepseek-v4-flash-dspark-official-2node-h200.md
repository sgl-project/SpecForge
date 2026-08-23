# DeepSeek-V4-Flash DSpark — OFFICIAL drafter architecture, two-node H200

Variant of the [two-node H200 recipe](deepseek-v4-flash-dspark-2node-h200.md)
that trains the **official** DSpark drafter architecture — the same one
bundled with `deepseek-ai/DeepSeek-V4-Flash-0731` — instead of the 5-layer
Qwen3-GQA approximation:

| | approximation (old recipe) | official (this recipe) |
|---|---|---|
| stages | 5 dense Qwen3 GQA layers | 3 DeepSeek-V4 blocks (`mtp.0/1/2`) |
| block type | GQA + dense SwiGLU | MLA-LoRA attn + 256-expert MoE + mHC |
| capture layers | `[1, 11, 21, 31, 41]` | `[40, 41, 42]` |
| block size | 7 | 5 |
| params | ~1B | ~19.9B |

Model: `specforge/modeling/draft/dspark_v4.py` (`DSparkV4DraftModel`), module
tree named so checkpoints natively carry the official `mtp.*` tensor names.
Config: `examples/configs/deepseek-v4-flash-dspark-official-2node-h200.yaml`
with draft config `configs/deepseek-v4-flash-dspark-official.json`.

## Validation gates (run before the full 2-node run)

1. **Parity vs the reference implementation** (CPU only, ~10 min):

   ```bash
   python3 scripts/gates/dspark_v4_parity_check.py            # official weights
   python3 scripts/gates/dspark_v4_parity_check.py --tiny     # fast structural
   ```

   Loads the dequantized official `mtp.*` weights into both
   `DSparkV4DraftModel` and the reference `inference/model.py` block stack
   (tilelang kernels shimmed in torch) and compares hidden states.

2. **Quantizer round-trip** (CPU): official → dequant → requant must be exact:

   ```bash
   python3 scripts/bundle_dspark_v4_official.py --self-test
   ```

3. **Export-path gate**: bundle the *official-dequant* weights and serve them —
   accept length must match the official checkpoint's:

   ```bash
   python3 scripts/init_dspark_v4_from_target.py --from-official \
     --output-dir /tmp/dspark-v4-official-roundtrip
   python3 scripts/bundle_dspark_v4_official.py \
     --draft /tmp/dspark-v4-official-roundtrip \
     --output-dir /cluster-storage/models/dspark-v4-roundtrip-bundle
   sglang serve /cluster-storage/models/dspark-v4-roundtrip-bundle \
     --trust-remote-code --tp 4 --speculative-algorithm DSPARK \
     --moe-runner-backend marlin --mem-fraction-static 0.90 \
     --chunked-prefill-size 4096 --host 0.0.0.0 --port 30000
   ```

## Warm start

Per-run decision (this recipe defaults to it): initialize the three stages
from the target's own layers 40-42, fresh markov/confidence heads:

```bash
python3 scripts/init_dspark_v4_from_target.py \
  --output-dir outputs/dspark-v4-official-init
```

The yaml points `model.draft_checkpoint_path` at that directory. Remove the
key for a pure random init.

## Launch

Identical to the base two-node runbook (same mooncake master, same four TP2
capture servers on 10.220.51.50, same trainer command on 10.220.51.52) with
exactly one capture-server change:

```
--spec-capture-aux-layer-ids 40 41 42
```

Trainer:

```bash
cd /personal/SpecForge
export HF_HOME=/cluster-storage/models MC_TRANSFER_TIMEOUT=300
specforge train -c examples/configs/deepseek-v4-flash-dspark-official-2node-h200.yaml
```

Differences vs the base recipe, all encoded in the yaml:

- `training.attention_backend: native` — the model implements the official
  window/block attention internally (fixed 128-token main-stream window per
  block + bidirectional intra-block attention + per-head sink logit); the
  training wrapper builds no masks.
- `training.fsdp_sharding: FULL_SHARD` — the drafter is ~19.9B params;
  per-rank steady state ≈ 60-65 GB (bf16 shard + grads + fp32 Adam masters).
- `training.warmup_ratio: 0.02` — the baseline run (constant 6e-4, no warmup)
  diverged to NaN at step ~1380.
- feature-store byte caps rescaled for 3 capture layers (~0.25 GiB/8K sample
  vs ~0.4 GiB).

## Training-time comparison vs the baseline run

The baseline `outputs/deepseek-v4-flash-dspark-2node-h200` NaN'd between steps
1370-1380; compare only against its healthy window (steps <= 1370, checkpoint
`-step1280`). Metrics: `train/acc`, `train/ce_loss`,
`train/accuracy_position_{0..4}` (baseline has 7 positions; use 0-4), and the
chained simulated accept length `1 + sum_k prod_{j<=k} acc_pos_j` over the
shared 0-4 horizon.

## Export, serving, eval

```bash
# 1. training checkpoint -> HF draft dir (tensors already named mtp.*)
specforge export --to hf \
  --checkpoint outputs/deepseek-v4-flash-dspark-official-2node-h200/deepseek-v4-flash-dspark-official-2node-h200-step<N> \
  --draft-config configs/deepseek-v4-flash-dspark-official.json \
  --output-dir outputs/dspark-v4-official-hf-step<N>

# 2. quantize + bundle into the official 0731 layout (fp8/fp4 + linked shards)
python3 scripts/bundle_dspark_v4_official.py \
  --draft outputs/dspark-v4-official-hf-step<N> \
  --output-dir /cluster-storage/models/dspark-v4-trained-bundle

# 3. serve exactly like the official checkpoint (single node, TP4)
sglang serve /cluster-storage/models/dspark-v4-trained-bundle \
  --trust-remote-code --tp 4 --speculative-algorithm DSPARK \
  --moe-runner-backend marlin --mem-fraction-static 0.90 \
  --chunked-prefill-size 4096 --host 0.0.0.0 --port 30000

# 4. accept-length eval (GSM8K + AIME25); the bench runner hardcodes an
#    EAGLE3 launch, so point it at the running DSPARK server:
python3 benchmarks/bench_eagle3.py \
  --model-path deepseek-ai/DeepSeek-V4-Flash-0731 --port 30000 \
  --benchmark-list gsm8k:200 aime25:30 --skip-launch-server
```

Run step 3-4 once against the *official* `deepseek-ai/DeepSeek-V4-Flash-0731`
checkpoint first to get the reference accept-length numbers on the same
hardware.
