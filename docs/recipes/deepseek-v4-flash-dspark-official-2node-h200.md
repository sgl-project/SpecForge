# DeepSeek-V4-Flash DSpark — official drafter architecture, two-node H200

Trains the DSpark drafter architecture bundled with
`deepseek-ai/DeepSeek-V4-Flash-0731`: three DeepSeek-V4 blocks (`mtp.0/1/2`:
MLA-LoRA attention, a 256-expert MoE and mHC hyper-connections, about 19.9B
parameters) fed by target layers 40-42, with block size 5.

- Model: `specforge/modeling/draft/dspark_v4.py` (`DSparkV4DraftModel`); its
  module tree carries the official `mtp.*` tensor names.
- Draft config: `configs/deepseek-v4-flash-dspark-official.json`.
- Recipe: `examples/configs/online/disaggregated/external/deepseek-v4-flash-dspark-official-2node-h200.yaml`
  (a capture node with four TP2 servers, a trainer node with eight FSDP ranks).

Weights always come from the 0731 snapshot, whose shards 46-48 bundle the
drafter. The standalone `DeepSeek-V4-Flash-DSpark` repo is a different release
(byte-identical config.json, different target and drafter weights); do not mix
the two.

## Checks before a run

1. Parity against DeepSeek's reference `inference/model.py` (CPU, about 10 min):

   ```bash
   python3 scripts/gates/dspark_v4_parity_check.py            # official weights
   python3 scripts/gates/dspark_v4_parity_check.py --tiny     # fast structural
   ```

2. Quantizer round trip (CPU): official -> dequant -> requant must be exact:

   ```bash
   python3 scripts/bundle_dspark_v4_official.py --self-test
   ```

3. Export path: bundle the dequantized official weights and serve them as in
   "Export and serve" below; the accept length must match the official
   checkpoint's.

   ```bash
   python3 scripts/init_dspark_v4_from_target.py --from-official \
     --output-dir /tmp/dspark-v4-official-roundtrip
   python3 scripts/bundle_dspark_v4_official.py \
     --draft /tmp/dspark-v4-official-roundtrip \
     --output-dir /path/to/models/dspark-v4-roundtrip-bundle
   ```

## Warm start

The recipe starts the three stages from the target's own layers 40-42 with
fresh Markov and confidence heads:

```bash
python3 scripts/init_dspark_v4_from_target.py \
  --output-dir outputs/dspark-v4-official-init
```

`model.draft_checkpoint_path` in the recipe points at that directory. Remove the
key for a random init, or pass `--from-official` to fine-tune the released
drafter instead.

## Launch

Capture node (192.0.2.10 in the recipe): a Mooncake master and four TP2
capture servers on GPUs 0-7 that capture target layers 40-42.

```bash
mooncake_master --enable_http_metadata_server=true \
  --http_metadata_server_host=0.0.0.0 --rpc_port=35551 \
  --http_metadata_server_port=35880 --default_kv_lease_ttl=5m &

export MOONCAKE_MASTER_SERVER_ADDR=192.0.2.10:35551
export MOONCAKE_METADATA_SERVER=http://192.0.2.10:35880/metadata
# The trainer node fetches features from these servers' Mooncake segments,
# so register a routable address (127.0.0.1 would be unreachable).
export MOONCAKE_LOCAL_HOSTNAME=192.0.2.10
export MOONCAKE_PROTOCOL=tcp MC_TRANSFER_TIMEOUT=300
export MOONCAKE_GLOBAL_SEGMENT_SIZE=$((24 << 30)) MOONCAKE_LOCAL_BUFFER_SIZE=$((1 << 30))
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$((2 * i)),$((2 * i + 1)) python -m sglang.launch_server \
    --host 0.0.0.0 --port $((30000 + i)) \
    --model-path deepseek-ai/DeepSeek-V4-Flash-0731 --trust-remote-code \
    --skip-tokenizer-init --tp-size 2 --mem-fraction-static 0.85 \
    --context-length 8704 --max-prefill-tokens 8704 --chunked-prefill-size -1 \
    --max-running-requests 8 --moe-runner-backend marlin --watchdog-timeout 3600 \
    --enable-spec-capture --spec-capture-method dspark \
    --spec-capture-aux-layer-ids 40 41 42 --disable-cuda-graph &
done
```

Trainer node (192.0.2.20):

```bash
export MC_TRANSFER_TIMEOUT=300
# The ~19.9B drafter under FULL_SHARD runs close to the HBM budget; expandable
# segments avoid fragmentation-driven OOM across the micro-batches.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.9
specforge train -c examples/configs/online/disaggregated/external/deepseek-v4-flash-dspark-official-2node-h200.yaml
```

What the recipe encodes:

- `training.attention_backend: native`: the model implements the official
  window/block attention itself (a fixed 128-token main-stream window per
  block, bidirectional attention inside the block and a per-head sink logit),
  so the training wrapper builds no masks.
- `training.fsdp_sharding: FULL_SHARD`: params, grads and optimizer states
  shard across the eight ranks. Measured on 8x H200 with recipe-shaped
  synthetic batches: about 18.3 s per optimizer step and a peak of 117 GiB per
  rank with the allocator settings above. Two ShareGPT epochs (1885 steps)
  take about 9.6 h.
- `training.warmup_ratio: 0.02`: a run with a constant 6e-4 and no warmup
  diverged to NaN at step ~1380.
- Feature-store byte caps sized for three capture layers (about 0.25 GiB per
  8K-token sample).

## Export and serve

```bash
# 1. training checkpoint -> HF draft dir (tensors already named mtp.*)
specforge export --to hf \
  --checkpoint outputs/deepseek-v4-flash-dspark-official-2node-h200/deepseek-v4-flash-dspark-official-2node-h200-step<N> \
  --draft-config configs/deepseek-v4-flash-dspark-official.json \
  --output-dir outputs/dspark-v4-official-hf-step<N>

# 2. quantize + bundle into the official 0731 layout (fp8/fp4 + linked shards);
#    --draft-only writes just the drafter, served with
#    --speculative-draft-model-path next to the official target
python3 scripts/bundle_dspark_v4_official.py \
  --draft outputs/dspark-v4-official-hf-step<N> \
  --output-dir /path/to/models/dspark-v4-trained-bundle

# 3. serve exactly like the official checkpoint (single node, TP4)
sglang serve /path/to/models/dspark-v4-trained-bundle \
  --trust-remote-code --tp 4 --speculative-algorithm DSPARK \
  --moe-runner-backend marlin --mem-fraction-static 0.90 \
  --chunked-prefill-size 4096 --host 0.0.0.0 --port 30000

# 4. accept-length eval (GSM8K + AIME25); the bench runner hardcodes an
#    EAGLE3 launch, so point it at the running DSPARK server:
python3 benchmarks/bench_eagle3.py \
  --model-path deepseek-ai/DeepSeek-V4-Flash-0731 --port 30000 \
  --benchmark-list gsm8k:200 aime25:30 --skip-launch-server
```

Run steps 3-4 once against the official `deepseek-ai/DeepSeek-V4-Flash-0731`
checkpoint first to get reference accept-length numbers on the same hardware.
