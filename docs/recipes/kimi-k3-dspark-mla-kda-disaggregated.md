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

The exact full-attention recipe preserves the K3 reference run's training
settings and trainer shape:

- 4096 token examples, per-rank batch size 1, accumulation 32, DP16;
- four-batch feature prefetch to overlap Mooncake TCP transfer with trainer
  compute;
- 10 epochs, learning rate `6e-4`, warmup ratio `0.04`;
- 9,173 optimizer steps (`469,695 * 10 // 512`), explicitly pinned so all
  roles share one schedule horizon at startup;
- 512 anchors, block size 7, Markov rank 256;
- CE/L1/confidence weights `0.1/0.9/1.0`;
- log every 10 steps and save every 250 steps;
- retain the latest three assembled checkpoints on each trainer node;
- online W&B project `specforge-dspark`.

The supplied exact topology is two TP8 K3 capture replicas and a separate
two-node, 16-rank trainer. The source job's `BATCH_SIZE=8` was per TP8 target
replica, not per FSDP rank: its two replicas formed a 16-sample optimizer
microbatch. One sample on each of 16 trainer GPUs with accumulation 32 restores
the source global batch of 512 and its per-GPU draft compute. The experimental
MLA/KDA recipes remain portable one-node DP8 configurations with two samples
per rank and the same global batch. Two optimizer windows stay in Mooncake so
target capture for the next update overlaps draft training; record any topology
override in the W&B run config.

## Smoke then full training

Start Mooncake as described in
[the K3 V1C runbook](kimi-k3-dspark-v1c-disaggregated.md), then apply the K3
patch to revision `ee560a2b2df5dafe18fd835d2e546eff019ca5ba`. Launch
`run_kimi_k3_dspark_capture_server.sh` on each of the two capture nodes. The
capture layer ids are part of this recipe's contract and intentionally differ
from the V1C reproduction:

```bash
export MODEL_PATH=/workspace/models/Kimi-K3-cdd2e49a
export MOONCAKE_MASTER_IP=10.65.0.2
export SGLANG_ROOT=/workspace/sglang-kimi-k3-ee560a2-spec-capture
export CAPTURE_IP=10.65.0.5  # use the current node's routable address
examples/disagg/run_kimi_k3_dspark_capture_server.sh
```

Use capture IP/ports that match the selected YAML.

For the exact five-layer full-attention reference architecture, use capture
layers `[7, 23, 51, 67, 83]` and the checked-in full-attention recipe. Run the
producer beside trainer rank 0 so both can use the rank-0 local `control_dir`.
The rank-0 inbox HTTP relay serves only tensor-free `SampleRef` metadata to the
second trainer node; feature tensors still move directly through Mooncake.
Override the placeholder host names with private, trusted-network addresses:

```bash
export AUX_LAYER_IDS="7 23 51 67 83"
specforge train \
  --config examples/configs/kimi-k3-dspark-fullattn-openperfectblend-disaggregated.yaml \
  --role producer \
  deployment.trainer.master_addr=10.65.0.3 \
  deployment.disaggregated.inbox_server_url=http://10.65.0.3:35900

# Trainer node 0
WANDB_MODE=online MOONCAKE_LOCAL_HOSTNAME=10.65.0.3 specforge train \
  --config examples/configs/kimi-k3-dspark-fullattn-openperfectblend-disaggregated.yaml \
  --role consumer --node-rank 0 \
  deployment.trainer.master_addr=10.65.0.3 \
  deployment.disaggregated.inbox_server_url=http://10.65.0.3:35900

# Trainer node 1
WANDB_MODE=online MOONCAKE_LOCAL_HOSTNAME=10.65.0.4 specforge train \
  --config examples/configs/kimi-k3-dspark-fullattn-openperfectblend-disaggregated.yaml \
  --role consumer --node-rank 1 \
  deployment.trainer.master_addr=10.65.0.3 \
  deployment.disaggregated.inbox_server_url=http://10.65.0.3:35900
```

When the two trainer nodes do not share `output_dir`, start
`examples/disagg/sync_distributed_checkpoints.py` on both nodes with local rank
ranges `0-7` and `8-15`, respectively. The
[disaggregated examples guide](../../examples/disagg/README.md#checkpoints-without-shared-storage)
contains the complete commands and private-network boundary.

## Capture throughput contract

The source run used two independent TP8 target replicas. A normal TP8 prefill
and a capture-enabled TP8 prefill should have comparable model-compute speed,
but one replica still provides only half of the source job's aggregate sample
rate. The capture patch therefore:

- performs D2H only on the output TP rank and reuses contiguous per-request
  views instead of concatenating every single-chunk tensor;
- publishes the scheduler batch through Mooncake `batch_put_from` rather than
  one RPC per feature object;
- gives each target endpoint two producer request slots, overlapping one
  bounded background publish with the next TP8 prefill; and
- holds the HTTP completion response until every feature key is durable.

The exact 4,096-token validation captured 128 unique samples (211,715 tokens,
18,214,264,880 feature bytes) in 25.010 seconds after the first request began:
5.118 samples/s per TP8 replica. Two replicas project to 10.236 samples/s, or
71.97 optimizer steps/hour at global batch 512. The source W&B run measured
71.50 steps/hour. Steady target prefill remained approximately 10.9–11.2k
tokens/s, so the remaining scale factor is replica count rather than an SGLang
compute regression.

A DP16 end-to-end smoke completed with finite losses and gradient norms across
all 16 ranks. Both trainer nodes assembled and opened the portable checkpoint:
one shared training-state file plus rank files 0--15. Before prefetch, its warm
step took 56.44 seconds: 35--39 seconds of trainer compute plus up to 21 seconds
waiting for Mooncake TCP feature materialization.

With `data.dataloader_num_workers: 4`, four subsequent warm steps took 49.61,
47.70, 48.52, and 49.25 seconds. Their mean was 48.77 seconds, or 73.82
steps/hour and 10.50 samples/s at global batch 512. This is 3.1% faster than
the source W&B run's 50.35 seconds/step (71.50 steps/hour). The prefetch smoke
is recorded in W&B as run `npia5q21`. Keep prefetch enabled when Mooncake uses
TCP; it overlaps feature transfer with the current optimizer step without
changing capture outputs or the training objective.

The resulting 9,173-step full run is recorded in W&B as run `fth4aze4`. Steps
20 through 70 reported finite losses and gradient norms. Step time fell from
45.53 to approximately 40.3 seconds as feature prefetch warmed; steps 60 and
70 sustained 89.28 and 88.85 steps/hour (about 12.7 samples/s), roughly 24%
above the source run. These measurements are early-run validation; use the W&B
performance panels and assembled checkpoints to monitor the remaining run.

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
hybrid, or the full-attention config above for the exact old architecture. The
full jobs need separate output/control directories and should not share one
trainer allocation concurrently.
