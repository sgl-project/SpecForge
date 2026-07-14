# Disaggregated examples

The disaggregated examples use the same `specforge train` runtime as colocated
training. The two shell wrappers describe deployment topology only; they do not
implement separate trainers.

| Workflow | Config | Launcher |
| --- | --- | --- |
| Online DFlash | `../configs/qwen3-8b-dflash-disaggregated.yaml` | `run_online.sh` |
| Online Domino | `../configs/qwen3-8b-domino-disaggregated.yaml` | `run_online.sh` |
| Online DSpark | `../configs/qwen3-4b-dspark-disaggregated.yaml` | `run_online.sh` |
| Offline EAGLE3 | `../configs/qwen3-8b-eagle3-offline-disaggregated.yaml` | `run_offline.sh` |

The environment contract differs by workflow. Online capture requires the
patched SGLang server, Mooncake services, a shared reference channel, and a
fresh consumer database. Offline ingestion requires a fresh manifest plus a
shared-directory or Mooncake feature store. Use a single attempt id to derive
all control paths and store ids; never generate it independently in each pool.
Each launcher invocation starts exactly one role. Start Mooncake and any SGLang
capture servers separately, then run producer and consumer in their respective
pools; the wrappers do not create or supervise those external services.

Online consumer ranks always use the same path:
`RefDistributor -> per-rank InboxChannel -> StreamingRefQueue -> DPAckController`.
This includes a one-process consumer; there is no direct-channel fallback.
Global rank 0 requires a fresh `DISAGG_DB` and is the sole durable-ledger
writer. `DISAGG_REF_CHANNEL`, `DISAGG_DB`, and the inbox directory (the default
is `${DISAGG_REF_CHANNEL}.inboxes`) must resolve to the same shared filesystem
state from every consumer node. The reference channel must also be visible to
the producer. Do not use node-local `/tmp` paths for these values in a
multi-node attempt.

Before capture, rank 0 publishes
`quantum = NNODES * NPROC_PER_NODE * training.batch_size *
training.accumulation_steps`. The canonical ref and resident-byte flow-control
watermarks live under the YAML `runtime` section; the ref high watermark must
be at least `quantum`. A partial quantum at EOF fails the attempt and cleans its
undispatched feature objects.

```yaml
runtime:
  producer_lease: 8
  in_flight_high_watermark: 256
  in_flight_low_watermark: 192
  resident_high_watermark_bytes: null
  resident_low_watermark_bytes: null
  feature_store_max_resident_bytes: null
```

The high thresholds pause capture and the low thresholds resume it. Resident
bytes are the sum of published-but-not-durably-consumed ref estimates, so
remote consumer deletes immediately advance the producer's byte accounting.
The hard cap rejects and cleans a newly captured batch before publication
instead of allowing the published set to grow without bound. The legacy
`DISAGG_IN_FLIGHT_{HIGH,LOW}_WATERMARK` and
`DISAGG_RESIDENT_{HIGH,LOW}_WATERMARK_BYTES` environment overrides remain
compatible. Set `producer_lease` and `feature_store_max_resident_bytes` in the
typed config (or as dotted overrides); they have no environment aliases.

For every new online attempt, use fresh `DISAGG_REF_CHANNEL`, `DISAGG_DB`, store
id, run id, inbox, and output paths. To restart only the consumer while that
attempt's producer/data plane and Mooncake objects remain available, reuse those
paths and invoke the same launcher with
`training.resume_from=<matching-latest-checkpoint>`. SQLite requeues unacked refs
and skips optimizer-durable refs. The marker and checkpoint steps must match.
Producer restart and recovery after producer cleanup are not supported.

Offline ingestion supports EAGLE3, DFlash, and Domino feature contracts. The
checked-in EAGLE3 recipe demonstrates both the shared-directory and Mooncake
backends. The producer is one ingestion process. Consumers use
`torchrun --standalone` when `NNODES=1`, or the standard `NNODES`, `NODE_RANK`,
`MASTER_ADDR`, and `MASTER_PORT` rendezvous when `NNODES>1`;
`NPROC_PER_NODE` defaults to `1`. Offline manifests remain fixed and
re-iterable, so offline epochs and consumer-only checkpoint resume retain their
normal semantics.

For example, launch a two-node consumer with the same config, attempt paths,
run id, and output directory on both hosts:

```bash
export NNODES=2 NPROC_PER_NODE=8
export MASTER_ADDR=trainer-0.example MASTER_PORT=29500
export DISAGG_DB=/shared/control/${ATTEMPT_ID}.consumer.sqlite
export DISAGG_INBOX_DIR=/shared/control/${ATTEMPT_ID}.inboxes

# trainer-0
export MOONCAKE_LOCAL_HOSTNAME=trainer-0-routable-address
NODE_RANK=0 examples/disagg/run_online.sh consumer run_id="$ATTEMPT_ID"

# trainer-1
export MOONCAKE_LOCAL_HOSTNAME=trainer-1-routable-address
NODE_RANK=1 examples/disagg/run_online.sh consumer run_id="$ATTEMPT_ID"
```

Use `run_offline.sh consumer` in the same pattern for offline training. The
producer remains one `specforge train` role and does not join the consumer
torchrun group.

See the [disaggregated training guide](../../docs/basic_usage/disaggregated_training.md)
for Mooncake prerequisites plus the full server, transport, and role commands
before running either wrapper. Run a launcher with `--help` for its concise
variable list.
