# Disaggregated training

Disaggregation is a topology of the canonical `specforge train` command, not a
second trainer. The producer captures or ingests features; the consumer runs the
trainer. Both roles use the same typed YAML, and the launch scripts append the
role override as the final argument.

The checked-in recipes are:

| Workflow | Config | Launcher |
| --- | --- | --- |
| Online DFlash | `examples/configs/qwen3-8b-dflash-disaggregated.yaml` | `examples/disagg/run_online.sh` |
| Online Domino | `examples/configs/qwen3-8b-domino-disaggregated.yaml` | `examples/disagg/run_online.sh` |
| Online DSpark | `examples/configs/qwen3-4b-dspark-disaggregated.yaml` | `examples/disagg/run_online.sh` |
| Offline EAGLE3 | `examples/configs/qwen3-8b-eagle3-offline-disaggregated.yaml` | `examples/disagg/run_offline.sh` |

The launchers only validate topology-specific environment variables and dispatch
to `specforge train`. Extra dotted config overrides are passed through unchanged.
Online server capture supports EAGLE3, DFlash, Domino, and DSpark; offline
ingestion supports EAGLE3, DFlash, and Domino feature contracts. The table lists
the representative checked-in deployment recipes.

Each invocation starts exactly one role; run producer and consumer separately in
their respective pools. The wrappers do not start, stop, or health-check
Mooncake or SGLang services. This keeps service supervision deployment-specific
while both training roles continue to use the one typed CLI.

## Mooncake prerequisite

Online disaggregation always uses Mooncake, and the Mooncake backend is an
optional system dependency rather than a SpecForge package dependency. Install
the matching official wheel on the capture server, producer, and consumer:

```bash
# CUDA earlier than 13
pip install mooncake-transfer-engine

# CUDA 13 or later
pip install mooncake-transfer-engine-cuda13

python -c 'from mooncake.store import MooncakeDistributedStore as S; s=S(); assert callable(getattr(s, "put_from", None)) and callable(getattr(s, "get_into", None))'
```

SpecForge uses one Mooncake wire contract: a hard-pinned object per tensor,
written with `put_from` and read with `get_into`. Clients without those methods
are rejected at startup with an upgrade command; the serialized object API is
not supported. Run the check above on every capture, producer, and consumer
image before starting an attempt.

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
channel and its sidecars must be on a shared control filesystem visible to the
producer and every consumer node. The SQLite database and per-rank inboxes are
consumer-only coordination state, but every consumer rank must resolve them to
the same shared files. Node-local paths such as `/tmp` are not valid for these
values in a multi-node attempt.

```bash
export ATTEMPT_ID=20260713-001
export CONFIG=examples/configs/qwen3-8b-dflash-disaggregated.yaml
export DISAGG_STORE_ID=qwen3-8b-dflash-${ATTEMPT_ID}
export DISAGG_REF_CHANNEL=/shared/control/${ATTEMPT_ID}.refs.jsonl
export MOONCAKE_METADATA_SERVER=http://metadata-server:8080/metadata
export MOONCAKE_MASTER_SERVER_ADDR=mooncake-master:50051
```

Every fresh attempt must use a new `DISAGG_REF_CHANNEL` and consumer
`DISAGG_DB`. The default inbox directory is
`${DISAGG_REF_CHANNEL}.inboxes`; if `DISAGG_INBOX_DIR` is set, it must also be a
fresh attempt-specific shared directory. Do not delete or reuse the channel,
its sidecars, the database, or inbox artifacts from an active run. A consumer
restart is the one exception: reuse that attempt's channel, database, inboxes,
store id, run id, topology, and output directory with `training.resume_from`.

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

On the training pool, set a fresh shared database path and each host's routable
Mooncake address. For one node, `NPROC_PER_NODE` controls data parallelism and
the launcher preserves the `torchrun --standalone` path:

```bash
export DISAGG_DB=/shared/control/${ATTEMPT_ID}.consumer.sqlite
export MOONCAKE_LOCAL_HOSTNAME=trainer-routable-address
export DISAGG_IDLE_TIMEOUT=1800

NPROC_PER_NODE=8 examples/disagg/run_online.sh consumer \
  run_id=qwen3-8b-dflash-${ATTEMPT_ID} \
  output_dir=outputs/qwen3-8b-dflash-${ATTEMPT_ID}
```

The consumer can start before the producer; it waits for references. The
launcher requires `DISAGG_DB` for every consumer, including a one-process run,
and pins `training.metadata_db_path` to the checked path. It requires a new path
for a fresh invocation and an existing path when `training.resume_from` is
present. A direct `specforge train` invocation can instead set the field in
YAML, but the public launcher keeps each attempt's coordination state explicit.
Every rank uses the same RefDistributor/inbox path; there is no separate
one-process consumer implementation.

For multiple training nodes, export one rendezvous and run the same consumer
command once on every node. Only `NODE_RANK` and the node-local
`MOONCAKE_LOCAL_HOSTNAME` differ:

```bash
# Identical on both nodes.
export NNODES=2
export NPROC_PER_NODE=8
export MASTER_ADDR=trainer-0.example
export MASTER_PORT=29500
export DISAGG_DB=/shared/control/${ATTEMPT_ID}.consumer.sqlite
export DISAGG_INBOX_DIR=/shared/control/${ATTEMPT_ID}.inboxes

# trainer-0
export MOONCAKE_LOCAL_HOSTNAME=trainer-0-routable-address
NODE_RANK=0 examples/disagg/run_online.sh consumer \
  run_id=qwen3-8b-dflash-${ATTEMPT_ID} \
  output_dir=/shared/outputs/qwen3-8b-dflash-${ATTEMPT_ID}

# trainer-1
export MOONCAKE_LOCAL_HOSTNAME=trainer-1-routable-address
NODE_RANK=1 examples/disagg/run_online.sh consumer \
  run_id=qwen3-8b-dflash-${ATTEMPT_ID} \
  output_dir=/shared/outputs/qwen3-8b-dflash-${ATTEMPT_ID}
```

The launcher maps those values to the standard `torchrun --nnodes`,
`--node_rank`, `--master_addr`, `--master_port`, and `--nproc_per_node`
arguments. Node 0 performs the fresh-database check; other nodes cannot safely
repeat it because global rank 0 may already have created the shared SQLite file.
All nodes still require the same explicit database path, and a restart requires
that retained database to exist before any node starts.

### Online runtime contract

The online producer owns prompt scheduling and publication only. Captured
tensors go to Mooncake and `SampleRef` metadata goes directly to
`DISAGG_REF_CHANNEL`; the producer has no training ledger and does not retain a
local training queue.

Consumer rank 0 is the only source-channel reader and the only writer to the
SQLite ledger. The consumer path is fixed for every DP size:

```text
RefDistributor (rank 0)
  -> one InboxChannel per rank
  -> one StreamingRefQueue per rank
  -> FeatureDataLoader
  -> DPAckController at the optimizer boundary
```

Before the producer captures any prompt, rank 0 publishes the global dispatch
quantum:

```text
quantum = dp_size * batch_size * accumulation_steps
```

The producer waits for this handshake. Its ref high watermark must be at least
`quantum`. Flow control is part of the canonical typed config, for example:

```yaml
runtime:
  producer_lease: 8
  in_flight_high_watermark: 256
  in_flight_low_watermark: 192
  resident_high_watermark_bytes: null
  resident_low_watermark_bytes: null
  feature_store_max_resident_bytes: null
```

`producer_lease` bounds the number of prompts each capture worker leases in one
round. `in_flight_high_watermark` pauses all workers when published but
consumer-unacknowledged refs reach the high threshold; they resume only after
refs fall to `in_flight_low_watermark`. The optional resident-byte high/low pair
adds the same hysteresis for Mooncake objects. Resident bytes are computed from
published `SampleRef.estimated_bytes` minus the channel's durably consumed
prefix, rather than Mooncake's process-local health counter, so deletes by a
remote consumer can resume the producer. Resume requires both the ref and byte
counts to be at or below their low thresholds. If the byte high watermark is
set without a byte low watermark, the high value is also the resume threshold.

`feature_store_max_resident_bytes` is a final hard publication cap. A newly
captured batch that would exceed it is aborted before its refs are published.
When both byte controls are enabled, the hard cap must be greater than or equal
to the resident high watermark; leave enough headroom for concurrent workers
already inside a capture call.

Existing deployments may keep the environment compatibility overrides
`DISAGG_IN_FLIGHT_HIGH_WATERMARK`, `DISAGG_IN_FLIGHT_LOW_WATERMARK`,
`DISAGG_RESIDENT_HIGH_WATERMARK_BYTES`, and
`DISAGG_RESIDENT_LOW_WATERMARK_BYTES`. If only the legacy ref-high override is
set, it is also used as the ref resume threshold. New configs should use the
`runtime` fields. `producer_lease` and the feature-store hard cap intentionally
have no environment alias; set them in YAML or with dotted `specforge train`
overrides so they remain in the recorded run config.

The distributor dispatches only complete quantum-sized windows, so every rank
receives exactly `batch_size * accumulation_steps` refs for one optimizer step.

The consumer opens Mooncake with `retain_on_release=true`. Materialization ends
the read lease but does not delete the feature. At an optimizer boundary,
`DPAckController` gathers every rank's ids, commits the ids and durable marker to
SQLite, explicitly removes those features, and only then advances the inbox and
source consumed counters. A crash before that transaction replays the refs; a
crash after it skips them.

If producer EOF leaves a partial quantum, the attempt fails instead of training
an incomplete global step. The distributor fails those refs terminally, aborts
their feature-store objects best-effort, settles their source-channel count,
and sends failure sentinels to every rank inbox.

For an online configuration with `training.num_epochs > 1`, the producer
creates that many prompt passes with fresh task/sample ids and all passes form
one consumer stream. New attempts use a new reference channel, inbox, SQLite
database, store id, run id, and output directory.

To restart an interrupted consumer while the original producer/data plane and
retained Mooncake objects are still available, reuse the original attempt
variables and pass the latest matching checkpoint:

```bash
NPROC_PER_NODE=8 examples/disagg/run_online.sh consumer \
  run_id=qwen3-8b-dflash-${ATTEMPT_ID} \
  output_dir=outputs/qwen3-8b-dflash-${ATTEMPT_ID} \
  training.resume_from=outputs/qwen3-8b-dflash-${ATTEMPT_ID}/qwen3-8b-dflash-${ATTEMPT_ID}-latest
```

Rank 0 reconciles SQLite: optimizer-durable ids are idempotently removed and
skipped, while committed-unacked ids are requeued. The durable marker step must
equal the checkpoint step; a marker ahead of the checkpoint is rejected instead
of silently losing updates. This recovery contract is consumer-only. It does
not restart a producer, reconstruct deleted Mooncake data, or promise recovery
after producer cleanup; `training.resume_from` remains invalid for the producer
role.

## Offline feature ingestion

Offline disaggregation ingests existing EAGLE3, DFlash, or Domino feature
checkpoints, publishes a manifest, and trains from the resulting feature store.
It supports either a shared directory or Mooncake, and consumer ranks can use
data parallelism.

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

NPROC_PER_NODE=8 examples/disagg/run_offline.sh consumer \
  run_id=qwen3-8b-eagle3-${ATTEMPT_ID} \
  output_dir=outputs/qwen3-8b-eagle3-${ATTEMPT_ID}
```

The consumer may start first and wait for the producer's completion sentinel.
With the shared-directory backend, the producer may also finish before the
consumer starts. The producer is a single ingestion process. With `NNODES=1`
(the default), the consumer runs through `torchrun --standalone` and
`NPROC_PER_NODE` is its local DP width.

For a multi-node offline consumer, run the same command on each node. The
manifest, shared-directory store, config, run id, and output directory must be
identical and visible at the same paths:

```bash
# Identical on every consumer node.
export NNODES=2
export NPROC_PER_NODE=8
export MASTER_ADDR=trainer-0.example
export MASTER_PORT=29500

# trainer-0
NODE_RANK=0 examples/disagg/run_offline.sh consumer \
  run_id=qwen3-8b-eagle3-${ATTEMPT_ID} \
  output_dir=/shared/outputs/qwen3-8b-eagle3-${ATTEMPT_ID}

# trainer-1
NODE_RANK=1 examples/disagg/run_offline.sh consumer \
  run_id=qwen3-8b-eagle3-${ATTEMPT_ID} \
  output_dir=/shared/outputs/qwen3-8b-eagle3-${ATTEMPT_ID}
```

### Mooncake backend

Set `DISAGG_BACKEND=mooncake` instead of `shared_dir`, omit
`DISAGG_STORE_ROOT`, and export the same Mooncake service addresses used by the
online workflow. Set a routable `MOONCAKE_LOCAL_HOSTNAME` independently in each
pool. The Mooncake producer stays alive until the consumer reports completion,
so launch both roles concurrently.

Every offline attempt must use a new manifest path and store id. The launcher
uses the same single- or multi-node torchrun contract described above and never
removes old manifests or store objects. For a multi-node Mooncake consumer,
`MOONCAKE_METADATA_SERVER` and `MOONCAKE_MASTER_SERVER_ADDR` are shared service
addresses, while `MOONCAKE_LOCAL_HOSTNAME` must be set to each node's own
routable address. Set `data.eval_hidden_states_path` together with
`training.eval_interval` when the consumer should run offline evaluation.

If that consumer is interrupted, resume the same attempt by reusing its
manifest, feature store, and `run_id`, and pass its latest checkpoint only to
the consumer:

```bash
examples/disagg/run_offline.sh consumer \
  run_id=qwen3-8b-eagle3-${ATTEMPT_ID} \
  output_dir=outputs/qwen3-8b-eagle3-${ATTEMPT_ID} \
  training.resume_from=outputs/qwen3-8b-eagle3-${ATTEMPT_ID}/qwen3-8b-eagle3-${ATTEMPT_ID}-latest
```

The offline producer is an ingestion role, not a trainer, and rejects
`training.resume_from`. Resume is consumer-only: reuse the exact attempt
manifest/store, run id, output directory, model and parallel topology, and
launch every consumer node from the same checkpoint. Changing consumer world
size during exact optimizer-state resume is not part of this contract.

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
typed config from disagreeing. Producers always dispatch directly to
`specforge train`. Consumers use `--standalone` for `NNODES=1`; for
`NNODES>1`, all four values `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT` plus
`NNODES` are required and mapped directly to torchrun. `NPROC_PER_NODE`
defaults to `1` in either case.
