# Prepare offline features with the capture plugin

Offline preparation uses a CPU client, SGLang capture servers, and Mooncake:

`dataset -> plugin prefill -> Mooncake -> CPU client -> .ckpt/.ckpt.gz`

Offline colocated and disaggregated **training** continue to read these files;
they do not need a capture server or a local ModelRunner. The preparation client
does not import SGLang or load target model weights. It still needs the target's
config/tokenizer and the same draft config used for training.

## Start the services

Use an SGLang environment with the native APIs described in the
[plugin installation guide](../../plugins/sglang-spec-capture/README.md), and
install the current plugin there. Use SGLang commit `c757c4bb7c55d1eb69fdb2bdc6e6fb8507acb7a0`; released 0.5.23
still lacks the capture APIs. The server and SpecForge client may use the same
environment with a compatible Mooncake wheel, or separate environments.

For a single-host TCP example, start a Mooncake master in a separate terminal:

```bash
mooncake_master --enable-http-metadata-server=true
```

Export the following in both the server and client terminals:

```bash
export MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:50051
export MOONCAKE_METADATA_SERVER=http://127.0.0.1:8080/metadata
export MOONCAKE_LOCAL_HOSTNAME=127.0.0.1
export MOONCAKE_PROTOCOL=tcp
```

Across hosts, replace loopback addresses with reachable addresses and set each
process's `MOONCAKE_LOCAL_HOSTNAME` to its own reachable address. The offline CPU
client supports TCP/RDMA; NVLink GPU receive buffers are not used here.

For `configs/qwen3-8b-dflash.json`, launch this server in the SGLang environment:

```bash
SPECFORGE_SPEC_CAPTURE=1 SGLANG_SPEC_CAPTURE_GPU_PUT=0 \
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B --skip-tokenizer-init \
  --port 30000 --tp-size 1 \
  --speculative-algorithm DFLASH \
  --enable-return-aux-hidden-states \
  --return-aux-hidden-state-layer-ids 1 9 17 25 33 \
  --return-hidden-states-mode full --chunked-prefill-size -1
```

Choose the method and layers from the resolved draft capture plan. Domino uses
the DFLASH convention; EAGLE3 and DSpark use EAGLE3 and DSPARK respectively.
The client checks every server's `/server_info` for the exact `model_path`,
capture method, and ordered layer IDs. Use the same model identifier/path on
both sides, and reinstall the plugin on existing servers to advertise these
fields. TP size, device selection, attention backend, context length and radix
cache options belong to the server command.

## Capture and resume

From the SpecForge environment:

```bash
python scripts/prepare_hidden_states.py \
  --server-urls http://127.0.0.1:30000 \
  --target-model-path Qwen/Qwen3-8B \
  --strategy dflash --draft-model-config configs/qwen3-8b-dflash.json \
  --data-path ./cache/dataset/sharegpt_train.jsonl \
  --output-path ./cache/hidden_states/qwen3-8b-dflash-sharegpt \
  --chat-template qwen --max-length 3072 --batch-size 32
```

Add `--compress` for `.ckpt.gz`. The output retains the existing
`rows_START-END/data_INDEX.ckpt[.gz]` layout and vocabulary mapping. Algorithms
own the persisted feature names. Input IDs and loss masks remain client-side;
the service only returns hidden-state artifacts.

Multiple `--server-urls` are used in round-robin order. To run concurrent CPU
clients, replace `python` with `torchrun --standalone --nproc_per_node=4`.
Workers partition samples without sampler padding or duplicate submissions;
this worker count is independent of server TP size.

Rerun the same command to skip completed files. Changing the batch size or CPU
worker count preserves sample IDs. Keep dataset contents/order, preprocessing,
strategy, target, draft config, and file-group size unchanged when resuming;
use a new output directory for a different dataset or capture configuration.

Writes use temporary files and atomic rename. The client waits for writes to
finish before releasing temporary Mooncake objects, and attempts cleanup on
request, read, or write errors. A response timeout fails the batch instead of
silently retrying it. This is not a durable job/lease protocol: abrupt process
termination or a server publication arriving after a timed-out request can
leave objects until the store is reclaimed. Size the store for concurrent
batches and use `--capture-timeout` appropriate to the target.

The service still runs SGLang's normal prefill/logits path with
`max_new_tokens=0`; this migration does not bypass the target LM head.
