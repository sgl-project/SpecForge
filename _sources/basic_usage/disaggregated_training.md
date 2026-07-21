# Disaggregated training

Disaggregation is a launch topology of the canonical training command:

```bash
specforge train -c run.yaml
```

The producer captures or ingests features and the consumer runs the canonical
trainer. Online training always uses this producer/consumer topology; there is
no colocated target-inference path and no separate Python training entry.

The checked-in recipes are:

| Workflow | Config |
| --- | --- |
| Online EAGLE3 | `examples/configs/qwen3-8b-eagle3-disaggregated.yaml` |
| Online P-EAGLE | `examples/configs/qwen3-8b-peagle-disaggregated.yaml` |
| Online DFlash | `examples/configs/qwen3-8b-dflash-disaggregated.yaml` |
| Online one-server + DP7 Qwen3-8B DFlash | `examples/configs/qwen3-8b-dflash-1server-dp7-disaggregated.yaml` |
| Online Domino | `examples/configs/qwen3-8b-domino-disaggregated.yaml` |
| Online two-server Qwen3-8B Domino | `examples/configs/qwen3-8b-domino-multiserver-disaggregated.yaml` |
| Online DSpark | `examples/configs/qwen3-4b-dspark-disaggregated.yaml` |
| Online Qwen3.6 DFlash | `examples/configs/qwen3.6-27b-dflash-disaggregated.yaml` |
| Online one-server + DP2 Qwen3.6 DFlash | `examples/configs/qwen3.6-27b-dflash-1server-dp2-disaggregated.yaml` |
| Online two-server Qwen3.6 DFlash | `examples/configs/qwen3.6-27b-dflash-multiserver-disaggregated.yaml` |
| Offline EAGLE3 | `examples/configs/qwen3-8b-eagle3-offline-disaggregated.yaml` |
| Offline Qwen2.5-7B EAGLE3 | `examples/configs/qwen2.5-7b-eagle3-offline-disaggregated.yaml` |

## One config owns the launch topology

Process topology and attempt paths live in the same typed run document as the
model and training settings. An online deployment has this shape:

```yaml
deployment:
  mode: disaggregated
  trainer:
    nnodes: 1
    nproc_per_node: 4
  disaggregated:
    control_dir: outputs/qwen3-8b-dflash-disaggregated/control
    consumer_state_dir: outputs/qwen3-8b-dflash-disaggregated/consumer-state
    backend: mooncake
    server_urls:
      - http://127.0.0.1:30000
    mooncake_metadata_server: http://127.0.0.1:35880/metadata
    mooncake_master_server_addr: 127.0.0.1:35551
    mooncake_protocol: tcp
```

`data.train_data_path` or `data.prompts_path` selects online mode;
`data.hidden_states_path` selects offline mode. There is no second scenario
flag to keep synchronized. The consumer world size is
`nnodes * nproc_per_node`; it must satisfy the TP/USP topology in `training`.

The control directory is attempt-scoped. The launcher deterministically derives
the online reference channel and lifecycle markers, or the offline manifest,
beneath that root. Online consumers put the rank-0 SQLite/WAL under
`consumer_state_dir`; that path should be node-local and is required for
multi-node trainers. Rank inboxes stay under the shared `control_dir` when
`deployment.trainer.nnodes > 1`, so remote ranks never need to access the
SQLite filesystem. A fresh attempt requires fresh control and consumer-state
directories. `store_id` defaults to `run_id` and can be set explicitly when the
Mooncake deployment requires another namespace.

Long producer/consumer waits are unbounded unless `idle_timeout_s`,
`peer_wait_timeout_s`, or `producer_hold_s` is explicitly configured. This
avoids terminating healthy training merely because it runs longer than a fixed
wall-clock default. An explicitly configured timeout is terminal: expiration
fails the attempt and runs its feature cleanup.

The file loader translates the former `training.deployment_mode`,
`training.server_urls`, and `training.metadata_db_path` fields at its migration
boundary. New recipes use `deployment`; `training.role` remains the canonical
persisted role selection and normally stays `auto` for a shared config.

## Single-node producer and consumer

With `deployment.trainer.nnodes: 1`, omitting `--role` starts a supervisor for
both SpecForge roles:

```bash
specforge train -c examples/configs/qwen3-8b-dflash-disaggregated.yaml
```

The checked-in online recipes use the local demo endpoints shown above, so the
command and `--plan` need no hidden topology variables. Point those typed fields
at the real services for a remote deployment. Environment values override the
typed Mooncake endpoint fields when the same recipe runs on another node;
`MOONCAKE_LOCAL_HOSTNAME` remains node-local.

The producer is a direct single process. The consumer is automatically launched
with the configured local process count. If either role fails, the supervisor
terminates the owned sibling process group and returns the failing status. A
clean offline shared-directory producer may finish before the consumer without
canceling it.

Inspect the resolved plan without starting processes:

```bash
specforge train -c examples/configs/qwen3-8b-dflash-disaggregated.yaml --plan
```

Plan output redacts secret-shaped overrides and credentials embedded in URLs.

## Multi-server capture

`deployment.disaggregated.server_urls` is an ordered worker topology, not merely
a set of endpoints. The producer creates one capture adapter and one rollout
worker for every list entry. All workers lease disjoint prompts from one
controller, write into the same Mooncake namespace, and publish references to
the same channel concurrently:

```yaml
deployment:
  mode: disaggregated
  trainer:
    nnodes: 1
    nproc_per_node: 2
  disaggregated:
    control_dir: outputs/qwen36-two-capture-servers/control
    backend: mooncake
    server_urls:
      - http://capture-0.example:30000
      - http://capture-1.example:30000
```

Every server must use the same target model, revision, capture method, and
auxiliary layer ids. A server failure returns its leased prompts to the shared
pool; surviving workers can lease and finish them. A worker is removed after
repeated consecutive failures, and the producer fails loudly rather than
publishing a successful partial stream if every worker is lost or a prompt
exhausts its bounded retries.

Repeating one URL is also intentional: each occurrence creates another worker
that can issue a blocking capture request to the same server. This can improve
prefill occupancy without allocating another target replica, subject to that
server's request capacity. For example, the following temporary override uses
four producer workers against one server while keeping the checked-in YAML as
the source of every other setting:

```bash
specforge train -c examples/configs/qwen3-8b-domino-disaggregated.yaml \
  'deployment.disaggregated.server_urls=["http://127.0.0.1:30000","http://127.0.0.1:30000","http://127.0.0.1:30000","http://127.0.0.1:30000"]'
```

In the default external-services mode, listing one or more URLs never starts
those servers. They must already be healthy and attached to the configured
Mooncake deployment. The optional
`deployment.disaggregated.managed_local` profile is a separate, explicit
single-node development convenience: when configured, it owns local Mooncake
and capture-server processes and derives their endpoints. Omitting that block
preserves the external-services behavior exactly. Managed-local launch is for a
fresh online Mooncake run with automatic producer + consumer supervision; it is
not a remote scheduler, resume path, existing-torchrun path, or replacement for
explicit producer and consumer pools.

The historical self-contained DFlash and Domino topologies each record one
TP=1 capture server on GPU 0 and a DP=7 trainer on GPUs 1–7:

```bash
specforge train -c \
  examples/configs/qwen3-8b-dflash-1server-dp7-disaggregated.yaml

specforge train -c \
  examples/configs/qwen3-8b-domino-1server-dp7-disaggregated.yaml
```

The checked-in multi-server Qwen3-8B Domino recipe records two TP=1 capture
servers on GPUs 0–1 and a DP=2 trainer on GPUs 2–3. It directly replaces the
discoverable multi-server behavior of the deleted legacy shell while retaining
the unified training entry:

```bash
specforge train -c \
  examples/configs/qwen3-8b-domino-multiserver-disaggregated.yaml
```

The Qwen3.6 DFlash one-server recipe owns a TP=1 server on GPU 0 and a DP=2
trainer on GPUs 1–2. Its larger sibling records two TP=2 capture servers on
GPUs 0–3 and a DP=2 trainer on GPUs 4–5:

```bash
specforge train -c \
  examples/configs/qwen3.6-27b-dflash-1server-dp2-disaggregated.yaml

specforge train -c \
  examples/configs/qwen3.6-27b-dflash-multiserver-disaggregated.yaml
```

The launcher starts Mooncake first, waits for its metadata and RPC endpoints,
starts every configured capture server and waits for every health endpoint,
then starts the existing producer/consumer plan. A service or role failure
terminates the owned stack; a clean producer exit leaves capture storage alive
until the consumer finishes. Normal completion stops capture servers before
Mooncake. Logs remain under the configured `control_dir/logs/`.

Managed capture derives SGLang `--context-length` as `data.max_length + 7` to
reserve the request headroom required by the capture endpoint. An explicit
`model.sglang_context_length` must be at least that value.

## Split pools and multi-node consumers

The same YAML launches either role explicitly:

```bash
# Inference/ingestion pool
specforge train -c run.yaml --role producer

# Trainer pool
specforge train -c run.yaml --role consumer
```

For offline features split across two physical nodes, the generic wrapper maps
the launcher-provided rank to those same two commands:

```bash
rcli exec --per-node <job> \
  'CONFIG=examples/configs/qwen2.5-7b-eagle3-offline-disaggregated.yaml bash examples/disagg/run_offline_2node.sh'
```

`RCLI_NODE_RANK=0` selects the producer and rank 1 selects the consumer. The
wrapper contains no trainer or transport setup; the shared `control_dir` and
`store_root` remain in YAML and must resolve to the same fresh attempt storage
on both nodes. `NODE_RANK`/`NUM_NODES` are accepted on other schedulers.

For multiple consumer nodes, record the shared topology once:

```yaml
deployment:
  mode: disaggregated
  trainer:
    nnodes: 2
    nproc_per_node: 8
    master_addr: trainer-0.example
    master_port: 29500
  disaggregated:
    control_dir: /shared/control/attempt-001
    backend: mooncake
    server_urls: [http://capture-server:30000]
```

Then provide only the node-local identity on each trainer host:

```bash
# trainer-0
specforge train -c run.yaml --role consumer --node-rank 0

# trainer-1
specforge train -c run.yaml --role consumer --node-rank 1
```

Automatic `--role both` rejects `nnodes > 1`: SpecForge does not SSH to remote
hosts or impersonate a cluster scheduler. An existing torchrun environment is
detected and used as the worker environment rather than nesting another
torchrun. A producer is rejected inside a multi-rank torchrun to prevent
duplicate capture/ingestion roles.

The separate-inference-node Qwen3-8B example keeps that scheduler boundary but
restores full development-stack orchestration:

```bash
export DISAGG_STORE_ID=qwen3-8b-two-node-attempt-001
export DISAGG_RUN_ROOT=/shared/specforge/$DISAGG_STORE_ID
rcli exec --per-node <job> \
  'bash examples/disagg/run_qwen3_8b_dflash_disagg_2node.sh'
```

Rank 0 starts and health-checks Mooncake and patched SGLang, then invokes the
canonical producer role. Rank 1 invokes the canonical consumer role, whose
process count still comes from the YAML. The wrapper coordinates only
node-local services and shared lifecycle markers; it does not contain another
trainer or construct `torchrun`. `NODE_RANK`/`NUM_NODES`/`HEAD_IP` may be used
instead of the corresponding `RCLI_*` variables on another scheduler. Both
nodes must see a fresh `DISAGG_RUN_ROOT`. References stay under its shared
`control` directory, while the wrapper's lifecycle markers stay at the shared
run root. SQLite/WAL and rank inboxes default to the trainer-local
`/tmp/specforge/$DISAGG_STORE_ID/consumer-state`. Set
`DISAGG_CONSUMER_STATE_DIR` or `LOCAL_SCRATCH` to select another node-local
path. This split-state form currently supports one trainer node only.

## External and managed-local services

Without `deployment.disaggregated.managed_local`, the unified supervisor owns
only SpecForge producer and consumer processes. Mooncake and patched SGLang are
external, usually long-lived services managed by Kubernetes, Slurm, systemd, or
the development environment. In this default mode, the training CLI does not
start, stop, or assign GPUs to them.

Install a Mooncake client compatible with the store's `put_from`/`get_into`
wire contract on the producer and consumer images:

```bash
# CUDA earlier than 13
pip install mooncake-transfer-engine

# CUDA 13 or later
pip install mooncake-transfer-engine-cuda13
```

Online configs require Mooncake. Stable endpoints may be supplied by the typed
deployment section or the environment:

```bash
export MOONCAKE_METADATA_SERVER=http://metadata-server:8080/metadata
export MOONCAKE_MASTER_SERVER_ADDR=mooncake-master:50051
export MOONCAKE_LOCAL_HOSTNAME=this-node-routable-address
```

Keep `DISAGG_AUTH_TOKEN`, node-local hostnames, and device visibility out of
checked-in YAML. `MOONCAKE_PROTOCOL` and `MOONCAKE_RDMA_DEVICES` may also be
node-local deployment values.

The online producer sends prompts to the URLs in
`deployment.disaggregated.server_urls`. Start a patched SGLang server separately
with the model, capture method, and auxiliary layer ids matching the draft
config. DFlash, Domino, and DSpark use the DFlash capture contract; EAGLE3 and
P-EAGLE use the EAGLE3 capture contract. Capture rejects chunked prefill and
radix-cache paths that can truncate the captured sequence. Online capture is
text-only: VLM training, including Qwen2.5-VL, is not supported. Online
evaluation is also not supported.

The repository's strict e2e gate remains a full local test-stack orchestrator:

```bash
bash scripts/gates/run_disaggregated_overfit_gate.sh
```

It starts and health-checks Mooncake and SGLang, invokes the canonical training
entry, verifies overfit and serving behavior, and cleans up every process it
owns. This preserves complete automated validation without turning the training
CLI into a production service manager.

## Online runtime contract

Consumer ranks always use one path:

```text
RefDistributor (rank 0)
  -> one InboxChannel per rank
  -> one StreamingRefQueue per rank
  -> FeatureDataLoader
  -> DPAckController at the optimizer boundary
```

Rank 0 publishes a global dispatch quantum before capture:

```text
quantum = consumer_world_size * batch_size * accumulation_steps
```

When both `training.total_steps` and `training.max_steps` are omitted, the
producer publishes the schedule horizon derived from the prepared prompt count,
`training.num_epochs`, and this quantum. The consumer validates that contract,
uses it for optimizer and Domino loss schedules, and still trains until EOF.

The producer waits for that handshake. Configure bounded capture under
`runtime`:

```yaml
runtime:
  producer_lease: 8
  in_flight_high_watermark: 256
  in_flight_low_watermark: 192
  resident_high_watermark_bytes: null
  resident_low_watermark_bytes: null
  feature_store_max_resident_bytes: null
```

High/low thresholds implement hysteresis. The optional hard resident cap rejects
and cleans a new capture before publication. A partial quantum at EOF is settled
without dispatch rather than training an incomplete optimizer step.

Consumer rank 0 is the only source-channel reader and SQLite writer. At an
optimizer boundary, `DPAckController` commits sample ids and the durable marker,
removes their feature objects, then advances the channel counters. A crash
before the transaction replays refs; a crash after it skips them.

## Offline shared-directory and Mooncake stores

The checked-in offline recipe uses a shared directory:

```yaml
deployment:
  mode: disaggregated
  trainer:
    nnodes: 1
    nproc_per_node: 1
  disaggregated:
    control_dir: outputs/qwen3-8b-eagle3-offline-disaggregated/control
    backend: shared_dir
    store_root: outputs/qwen3-8b-eagle3-offline-disaggregated/features
```

Both paths must be visible at the same location from producer and consumer
nodes. The producer ingests existing EAGLE3, DFlash, Domino, or DSpark features
and publishes a fixed manifest. Offline epochs remain re-iterable.

Set `backend: mooncake` instead, omit `store_root`, and provide the Mooncake
endpoints to use the remote store. Also set a positive
`deployment.disaggregated.producer_segment_size`: unlike online server-owned
captures, an offline producer owns the objects it ingests until the consumer
acknowledges them. `MOONCAKE_GLOBAL_SEGMENT_SIZE` remains a compatibility
fallback. The consumer uses a zero-sized client segment. A Mooncake offline
producer remains alive until the consumer reports completion; the single-node
supervisor handles that lifecycle.

## Resume and freshness

For every new online attempt, the consumer database and its WAL/SHM sidecars
must not exist. The launcher checks that rule on global rank 0. Resume requires
the retained database and a matching checkpoint:

```bash
specforge train -c run.yaml --role consumer \
  training.resume_from=outputs/run/run-latest
```

With the default `--role auto`, a disaggregated config containing
`training.resume_from` resolves to consumer-only. Explicit `--role both` is
rejected because producers are never resumed. Reuse the same control directory,
store id, run id, output directory, consumer topology, and Mooncake objects.
The durable optimizer-step acknowledgement must equal the checkpoint step.
Because acknowledgements may advance between periodic checkpoints, a crash in
that interval is deliberately rejected instead of replaying already-consumed
samples or silently skipping optimizer state; use a more frequent save interval
when that recovery window matters.

Offline disaggregated resume also launches only the consumer against the
retained manifest/store. Changing trainer world size during exact optimizer
state resume is outside the current contract.

The optional scripts `examples/disagg/run_online.sh` and
`examples/disagg/run_offline.sh` are thin delegates to the same command. Their
arguments use the CLI form, for example `run_online.sh --role producer`; they do
not parse `NPROC_PER_NODE`, construct torchrun, or manage transport state.
