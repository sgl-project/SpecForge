# Kimi-K3 DSpark 5×MLA and 4×KDA+1×MLA

These experimental draft backbones train against Kimi-K3 target features with
the disaggregated SpecForge data plane:

- `KimiK3DSpark5MLADraftModel`: five K3-style MLA layers;
- `KimiK3DSpark4KDA1MLADraftModel`: `KDA → KDA → MLA → KDA → KDA`.

Both retain the normal DSpark target projector, Markov head, confidence head,
loss, and seven-token proposal block. They use captured target layers
`[11, 23, 47, 71, 83]` (zero based).

## Architecture contract

MLA uses the K3 target dimensions: 96 heads, Q LoRA rank 1536, KV LoRA rank
512, 128 non-RoPE QK channels, 64 RoPE channels, 128 value channels, and the
K3 output gate.

KDA uses 96 heads of width 128, a bias-free four-token causal depthwise
convolution, a full-rank output gate, and the bounded forget gate with lower
bound `-5`. GPU training uses `fla-core==0.5.1`.

In the hybrid, the central MLA layer is the only layer that reads the captured
target context. Each KDA layer operates on one proposal block at a time and
resets its state between anchors. This preserves DFlash's anchor isolation;
linear recurrence cannot connect two independently sampled proposal blocks.

## Pinned data

The recipes use only this Kimi-K3 Open Perfect Blend regeneration:

| Field | Value |
| --- | --- |
| HF dataset | `skx618/Kimi-K3-OpenPerfectBlend-Regen` |
| Revision | `439c2fdc9fd2ae92e194bde468d26867b36dd660` |
| File | `data.jsonl` |
| Rows | `698316` |
| SHA-256 | `5418f09d1af8ec2e08e8385799f1eeb3c062c669b28407870f7737007bc3eeb9` |

Store the Hugging Face token outside the repository:

```bash
install -d -m 700 /workspace/k3_dspark/secrets
install -m 600 /dev/stdin /workspace/k3_dspark/secrets/hf_token
scripts/prepare_kimi_k3_openperfectblend.sh
```

The downloader verifies both the row count and SHA-256. SpecForge dynamically
renders the conversation schema and right-truncates to 4096 tokens.

## Reference training settings

The full recipes preserve the K3 reference run's training settings:

- 4096 token examples, batch size 8, accumulation 32;
- 10 epochs, learning rate `6e-4`, warmup ratio `0.04`;
- 512 anchors, block size 7, Markov rank 256;
- CE/L1/confidence weights `0.1/0.9/1.0`;
- log every 10 steps and save every 250 steps;
- online W&B project `specforge-dspark`.

The supplied topology is one TP8 K3 capture server and a separate four-rank
trainer. Because the trainer world size is part of the effective global batch,
record any topology override in the W&B run config.

## Smoke then full training

Start Mooncake as described in
[the K3 V1C runbook](kimi-k3-dspark-v1c-disaggregated.md), then launch the
patched latest K3 SGLang server. The capture layer ids are part of this recipe's
contract and intentionally differ from the V1C reproduction:

```bash
export CAPTURE_IP=10.65.0.2
export TARGET_MODEL=/workspace/models/Kimi-K3
export MOONCAKE_MASTER_SERVER_ADDR="$CAPTURE_IP:35551"
export MOONCAKE_METADATA_SERVER="http://$CAPTURE_IP:35880/metadata"
export MOONCAKE_LOCAL_HOSTNAME="$CAPTURE_IP"
export MC_TCP_BIND_ADDRESS="$CAPTURE_IP"
export MOONCAKE_PROTOCOL=tcp
export MOONCAKE_GLOBAL_SEGMENT_SIZE=1099511627776
export MOONCAKE_LOCAL_BUFFER_SIZE=1073741824
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 30000 \
  --model-path "$TARGET_MODEL" \
  --trust-remote-code \
  --skip-tokenizer-init \
  --tp-size 8 \
  --mem-fraction-static 0.76 \
  --context-length 4608 \
  --max-running-requests 8 \
  --max-total-tokens 40960 \
  --prefill-attention-backend flashinfer \
  --decode-attention-backend trtllm_mla \
  --moe-runner-backend marlin \
  --enable-symm-mem \
  --mamba-radix-cache-strategy extra_buffer \
  --max-mamba-cache-size 40 \
  --chunked-prefill-size -1 \
  --enable-spec-capture \
  --spec-capture-method dspark \
  --spec-capture-aux-layer-ids 11 23 47 71 83
```

Use capture IP/ports that match the selected YAML.

Keep the W&B API key in a protected file and export it only in the trainer
shell; do not put it in YAML or a command transcript.

Run a one-step smoke with the same architecture and objective before removing
the overrides:

```bash
WANDB_MODE=online specforge train \
  --config examples/configs/kimi-k3-dspark-5mla-openperfectblend-disaggregated.yaml \
  --role both \
  training.max_steps=1 \
  training.batch_size=1 \
  training.accumulation_steps=1 \
  data.train_data_path=/workspace/k3_dspark/data/kimi-k3-openperfectblend-smoke.jsonl
```

After the smoke produces finite loss, gradients, a checkpoint, and online W&B
telemetry, launch the full run:

```bash
WANDB_MODE=online specforge train \
  --config examples/configs/kimi-k3-dspark-5mla-openperfectblend-disaggregated.yaml \
  --role both
```

Use the corresponding
`kimi-k3-dspark-4kda-1mla-openperfectblend-disaggregated.yaml` config for the
hybrid. The two full jobs need separate output/control directories and should
not share one four-rank trainer allocation concurrently.
