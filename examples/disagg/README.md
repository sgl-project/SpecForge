# Disaggregated training

Disaggregation is a deployment mode of the same `specforge train` command. It
does not have a separate Python trainer. Producer and consumer use one YAML and
override only `training.role`.

## Online server capture

Online DFlash, Domino, DSpark, and EAGLE3 server capture use Mooncake for
features and a small shared JSONL channel for tensor-free `SampleRef` metadata.
Start the patched SGLang capture server first, then launch one role per pool:

```bash
export CONFIG=examples/configs/qwen3-8b-dflash-disaggregated.yaml
export DISAGG_STORE_ID=qwen3-8b-dflash-disaggregated
export DISAGG_REF_CHANNEL=/shared/control/qwen3-8b.refs.jsonl
export DISAGG_DB=/shared/control/qwen3-8b-consumer.sqlite
export DISAGG_SERVER_URL=http://capture-server:30000
export MOONCAKE_METADATA_SERVER=http://metadata-server:8080/metadata
export MOONCAKE_MASTER_SERVER_ADDR=mooncake-master:50051
export MOONCAKE_LOCAL_HOSTNAME="$(hostname -i | awk '{print $1}')"

ROLE=producer examples/disagg/run_role.sh
ROLE=consumer NPROC_PER_NODE=8 examples/disagg/run_role.sh
```

Every fresh online run must use new, attempt-specific `DISAGG_REF_CHANNEL` and
`DISAGG_DB` paths. The producer fails fast when the channel artifact already
exists, and no online run can be resumed in this PR. The database path must be
visible to every consumer rank and provides durable coordination only within
that attempt. Use `DISAGG_SERVER_URLS=url1,url2,...` for multiple capture servers.
Optional runtime controls include `DISAGG_INBOX_DIR`, `DISAGG_IDLE_TIMEOUT`,
`DISAGG_PEER_WAIT_TIMEOUT`, `MOONCAKE_PROTOCOL`, and
`MOONCAKE_RDMA_DEVICES`. Producer and consumer publish distinct success/failure
sentinels, so either role exits loudly instead of treating a peer crash as a
normal end of data.

Online disaggregation requires `model.target_backend: sglang` and either
`training.total_steps` or `training.max_steps` in the shared YAML. These
requirements apply to DFlash, Domino, DSpark, and EAGLE3 alike.
Both are measured in global optimizer updates: `max_steps` is a stop budget and
also supplies the schedule horizon when `total_steps` is omitted, while
`total_steps` alone does not stop a longer stream.

DSpark is intentionally server-capture-only; its example is
`examples/configs/qwen3-4b-dspark-disaggregated.yaml`.

## Offline EAGLE3

Offline disaggregation ingests existing feature checkpoints into either a
shared directory or Mooncake and publishes one manifest:

```bash
export CONFIG=examples/configs/qwen3-8b-eagle3-offline-disaggregated.yaml
export DISAGG_MANIFEST=/shared/control/eagle3-manifest.json
export DISAGG_STORE_ROOT=/shared/features
export DISAGG_BACKEND=shared_dir

ROLE=producer examples/disagg/run_role.sh
ROLE=consumer NPROC_PER_NODE=1 examples/disagg/run_role.sh
```

Set `DISAGG_BACKEND=mooncake` and the Mooncake variables from the online example
to avoid a shared feature mount. The producer then remains alive until the
consumer signals completion, because the objects live in its Mooncake segment.
Every fresh offline run must use a new `DISAGG_MANIFEST` path; the producer
fails fast when that artifact already exists. Offline inputs are currently
single-rank; increasing `NPROC_PER_NODE` for the offline consumer is rejected.

`run_domino_dflash_serving_gate.sh` is retained as a post-training serving
validation gate; it does not implement training.
