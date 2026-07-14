# Disaggregated training

Disaggregation is a topology of the canonical `specforge train` command, not a
second trainer. The producer captures or ingests features; the consumer runs the
trainer. Both roles use the same typed YAML, and the launch scripts append the
role override as the final argument.

The checked-in recipes are:

| Workflow | Config | Launcher |
| --- | --- | --- |
| Online DFlash | `examples/configs/qwen3-8b-dflash-disaggregated.yaml` | `examples/disagg/run_online.sh` |
| Online DSpark | `examples/configs/qwen3-4b-dspark-disaggregated.yaml` | `examples/disagg/run_online.sh` |
| Offline EAGLE3 | `examples/configs/qwen3-8b-eagle3-offline-disaggregated.yaml` | `examples/disagg/run_offline.sh` |

The launchers only validate topology-specific environment variables and dispatch
to `specforge train`. Extra dotted config overrides are passed through unchanged.

## Mooncake prerequisite

Online disaggregation always uses Mooncake, and the Mooncake backend is an
optional system dependency rather than a SpecForge package dependency. Install
the matching official wheel on the capture server, producer, and consumer:

```bash
# CUDA earlier than 13
pip install mooncake-transfer-engine

# CUDA 13 or later
pip install mooncake-transfer-engine-cuda13

python -c 'from mooncake.store import MooncakeDistributedStore'
```

For a development deployment, start the master with its embedded HTTP metadata
server in a separate terminal or service unit:

```bash
mooncake_master \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=0.0.0.0 \
  --http_metadata_server_port=8080
```

This provides the two addresses used below: `http://<master>:8080/metadata` and
`<master>:50051`. Production deployments should follow the
[Mooncake deployment guide](https://kvcache-ai.github.io/Mooncake/deployment/mooncake-store-deployment-guide.html)
for service supervision, high availability, networking, and RDMA configuration.

## Online server capture

The online producer sends prompts to a patched SGLang server. The server writes
captured tensors to Mooncake, while the producer publishes tensor-free sample
references through a shared JSONL control path. The consumer reads those
references and trains on one or more GPUs.

### 1. Patch and start SGLang

The checked-in patch targets SGLang 0.5.14. Apply it in the environment that
runs the capture server:

```bash
scripts/apply_sglang_spec_capture_patch.sh
```

Start the Mooncake metadata and master services, then export their addresses on
the capture server. `MOONCAKE_LOCAL_HOSTNAME` must be an address reachable from
the producer and consumer pools.

For the Qwen3 DFlash-family recipes, start the capture server with the layer ids
from the draft config:

```bash
export MOONCAKE_METADATA_SERVER=http://metadata-server:8080/metadata
export MOONCAKE_MASTER_SERVER_ADDR=mooncake-master:50051
export MOONCAKE_LOCAL_HOSTNAME=server-routable-address

python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --skip-tokenizer-init \
  --enable-spec-capture \
  --spec-capture-method dflash \
  --spec-capture-aux-layer-ids 1 9 17 25 33 \
  --chunked-prefill-size -1 \
  --disable-radix-cache \
  --host 0.0.0.0 \
  --port 30000
```

DFlash, Domino, and DSpark use `--spec-capture-method dflash`; EAGLE3 uses
`--spec-capture-method eagle3`. The target model and captured layer ids must
match the selected YAML and draft config. Capture rejects chunked prefill and
the radix cache because either can truncate the hidden-state sequence.

### 2. Choose one fresh attempt id

Choose the attempt id once and use the same value in both pools. The reference
channel must be visible to producer and consumer. The SQLite database is
consumer-only coordination state and must be visible to every consumer rank.

```bash
export ATTEMPT_ID=20260713-001
export CONFIG=examples/configs/qwen3-8b-dflash-disaggregated.yaml
export DISAGG_STORE_ID=qwen3-8b-dflash-${ATTEMPT_ID}
export DISAGG_REF_CHANNEL=/shared/control/${ATTEMPT_ID}.refs.jsonl
export MOONCAKE_METADATA_SERVER=http://metadata-server:8080/metadata
export MOONCAKE_MASTER_SERVER_ADDR=mooncake-master:50051
```

Every attempt must use a new `DISAGG_REF_CHANNEL` and consumer `DISAGG_DB`.
Do not delete or reuse control artifacts from an active or failed run. The
runtime atomically claims the producer path and publishes explicit success and
failure sentinels.

### 3. Launch the producer

On the inference pool, set the capture server URL and this host's routable
Mooncake address:

```bash
export DISAGG_SERVER_URL=http://capture-server:30000
export MOONCAKE_LOCAL_HOSTNAME=producer-routable-address

examples/disagg/run_online.sh producer \
  run_id=qwen3-8b-dflash-${ATTEMPT_ID}
```

Use `DISAGG_SERVER_URLS=url1,url2,...` for several capture servers. Server URLs
may instead be set as `training.server_urls` in the YAML.

### 4. Launch the consumer

On the training pool, set a fresh database path and this host's routable
Mooncake address. `NPROC_PER_NODE` controls single-node data parallelism:

```bash
export DISAGG_DB=/shared/control/${ATTEMPT_ID}.consumer.sqlite
export MOONCAKE_LOCAL_HOSTNAME=trainer-routable-address
export DISAGG_IDLE_TIMEOUT=1800

NPROC_PER_NODE=8 examples/disagg/run_online.sh consumer \
  run_id=qwen3-8b-dflash-${ATTEMPT_ID} \
  output_dir=outputs/qwen3-8b-dflash-${ATTEMPT_ID}
```

The consumer can start before the producer; it waits for references. The
launcher uses `torchrun --standalone`, so `NPROC_PER_NODE` covers GPUs on one
training node. The launcher requires `DISAGG_DB` for multi-rank training or
when `DISAGG_INBOX_DIR` is set, and pins `training.metadata_db_path` to the same
fresh path it checked. A direct `specforge train` invocation can instead set the
field in YAML, but the public launcher keeps each attempt's coordination state
explicit.

Online resume is not supported. Start a new attempt with new control paths,
store id, run id, and output directory.

## Offline EAGLE3 ingestion

Offline disaggregation ingests existing EAGLE3 feature checkpoints, publishes a
manifest, and trains from the resulting feature store. It supports either a
shared directory or Mooncake and is currently single-rank.

### Shared-directory backend

Choose one attempt id for both roles. `DISAGG_MANIFEST` and
`DISAGG_STORE_ROOT` must be visible to producer and consumer.

```bash
export ATTEMPT_ID=20260713-001
export CONFIG=examples/configs/qwen3-8b-eagle3-offline-disaggregated.yaml
export DISAGG_BACKEND=shared_dir
export DISAGG_STORE_ID=qwen3-8b-eagle3-${ATTEMPT_ID}
export DISAGG_MANIFEST=/shared/control/${ATTEMPT_ID}.manifest.json
export DISAGG_STORE_ROOT=/shared/features

examples/disagg/run_offline.sh producer \
  run_id=qwen3-8b-eagle3-${ATTEMPT_ID}

examples/disagg/run_offline.sh consumer \
  run_id=qwen3-8b-eagle3-${ATTEMPT_ID} \
  output_dir=outputs/qwen3-8b-eagle3-${ATTEMPT_ID}
```

The consumer may start first and wait for the producer's completion sentinel.
With the shared-directory backend, the producer may also finish before the
consumer starts.

### Mooncake backend

Set `DISAGG_BACKEND=mooncake` instead of `shared_dir`, omit
`DISAGG_STORE_ROOT`, and export the same Mooncake service addresses used by the
online workflow. Set a routable `MOONCAKE_LOCAL_HOSTNAME` independently in each
pool. The Mooncake producer stays alive until the consumer reports completion,
so launch both roles concurrently.

Every offline attempt must use a new manifest path and store id. The launcher
rejects `NPROC_PER_NODE` values other than 1 for the consumer and never removes
old manifests or store objects.

## Launcher contract

Run either script with `--help` for its environment contract. Both accept dotted
overrides after the role:

```bash
CONFIG=examples/configs/qwen3-8b-dflash-disaggregated.yaml \
  examples/disagg/run_online.sh producer \
  training.batch_size=2 training.role=consumer
```

The command still runs as `training.role=producer`, because the launcher appends
its fixed role after user overrides. This prevents the process topology and
typed config from disagreeing.
