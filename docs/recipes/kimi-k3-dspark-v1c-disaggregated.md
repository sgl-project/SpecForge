# Kimi K3 V1C DSpark disaggregated reproduction

This recipe migrates the prior four-node colocated Kimi K3 V1C continual run
to one TP8 capture node and one four-rank trainer node. It preserves the draft
architecture, weights-only warm start, regenerated agentic prompt order,
effective global batch, constant learning rate, and DSpark loss weights.

## Required source revisions

- SpecForge with configurable LR scheduling, an independent online prompt
  seed, and dedicated DSpark capture support.
- Kimi K3 SGLang revision `f8493a43a6a30d2a1cad6b0034e2c1b362d920d5`.
- The K3 SGLang tree patched with:

  ```bash
  scripts/apply_sglang_spec_capture_patch.sh --target kimi-k3-f8493a4
  ```

The patch makes `--spec-capture-method dspark` call K3's
`set_dspark_layers_to_capture` hook. The generic DFlash capture method is not
equivalent for K3. The same versioned patch carries the three required 64K
correctness guards: 64-bit Triton token offsets, scale-stable residual scoring,
and the Marlin grid.y fallback above 65,535 tokens.

## Artifacts

The checked-in recipe uses the paths already provisioned on the K3 RunPod
pool. Other deployments should override them without editing the recipe:

- target revision `cdd2e49a2c1cf8d4713b513955e415ed75405a72`;
- the weights-only V1C `epoch_0_step_0` draft checkpoint;
- the 462-row regenerated dataset whose SHA-256 is
  `6d50e6bb9ee59095eed91bfba035081efef9fea43bece9ad5dd01c6648a8ef24`.

Do not put Hugging Face or W&B credentials in YAML. Supply `HF_TOKEN` and
`WANDB_API_KEY` through protected node-local files or the process environment.

## Capture node

Start Mooncake with at least a 1 TiB global segment, then start the patched K3
server. Replace `CAPTURE_IP` with the routable address used by both nodes.

```bash
export MOONCAKE_LOCAL_HOSTNAME="$CAPTURE_IP"
export MOONCAKE_GLOBAL_SEGMENT_SIZE=1099511627776
export MOONCAKE_LOCAL_BUFFER_SIZE=1073741824
mooncake_master \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=0.0.0.0 \
  --rpc_port=35551 \
  --http_metadata_server_port=35880 \
  --metrics_port=35903
```

In a second process:

```bash
export MOONCAKE_MASTER_SERVER_ADDR="$CAPTURE_IP:35551"
export MOONCAKE_METADATA_SERVER="http://$CAPTURE_IP:35880/metadata"
export MOONCAKE_LOCAL_HOSTNAME="$CAPTURE_IP"
export MOONCAKE_PROTOCOL=tcp
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 30000 \
  --model-path /workspace/models/Kimi-K3-cdd2e49a \
  --trust-remote-code \
  --skip-tokenizer-init \
  --tp-size 8 \
  --mem-fraction-static 0.76 \
  --context-length 66048 \
  --max-running-requests 1 \
  --max-total-tokens 66048 \
  --attention-backend trtllm_mla \
  --moe-runner-backend marlin \
  --mamba-radix-cache-strategy extra_buffer \
  --max-mamba-cache-size 5 \
  --chunked-prefill-size -1 \
  --enable-spec-capture \
  --spec-capture-method dspark \
  --spec-capture-aux-layer-ids 7 23 51 67 83
```

## Trainer node

Resolve `capture-node` through DNS or override the three endpoint fields with
the capture node's IP. The default CLI role starts one CPU producer and a
four-rank FSDP consumer on the same trainer host.

```bash
export MOONCAKE_LOCAL_HOSTNAME="$TRAINER_IP"
export WANDB_API_KEY="$(< /protected/path/wandb-api-key)"
export WANDB_ENTITY=your-entity
CUDA_VISIBLE_DEVICES=0,1,2,3 specforge train \
  -c examples/configs/kimi-k3-dspark-v1c-disaggregated.yaml \
  --role both \
  "deployment.disaggregated.server_urls=[\"http://$CAPTURE_IP:30000\"]" \
  "deployment.disaggregated.mooncake_metadata_server=http://$CAPTURE_IP:35880/metadata" \
  "deployment.disaggregated.mooncake_master_server_addr=$CAPTURE_IP:35551"
```

For a one-update smoke run, additionally override the pre-tokenized four-row
fixture and shrink the optimizer quantum:

```bash
specforge train \
  -c examples/configs/kimi-k3-dspark-v1c-disaggregated.yaml \
  --role both \
  data.train_data_path= \
  data.prompts_path=/workspace/k3_dspark/k3_specforge/cache/kimi-k3-agentic-regen-9a6ea2c7-v1c-full139264/longest-smoke-pretokenized-4rows-65536.jsonl \
  training.num_epochs=1 \
  training.max_steps=1 \
  training.accumulation_steps=1 \
  tracking.report_to=none \
  runtime.in_flight_high_watermark=4 \
  runtime.in_flight_low_watermark=2
```

Validate in order: config plan, patch dry-run, server health, one captured
sample's tensor shapes/dtypes, one finite optimizer update and checkpoint, then
the full run. A smoke pass does not establish numerical parity; compare the
full run's loss/accuracy/tau trajectory and final checkpoint hashes separately.
