# Disaggregated examples

The disaggregated examples use the same `specforge train` runtime as colocated
training. The two shell wrappers describe deployment topology only; they do not
implement separate trainers.

| Workflow | Config | Launcher |
| --- | --- | --- |
| Online DFlash | `../configs/qwen3-8b-dflash-disaggregated.yaml` | `run_online.sh` |
| Online DSpark | `../configs/qwen3-4b-dspark-disaggregated.yaml` | `run_online.sh` |
| Offline EAGLE3 | `../configs/qwen3-8b-eagle3-offline-disaggregated.yaml` | `run_offline.sh` |

The environment contract differs by workflow. Online capture requires the
patched SGLang server, Mooncake services, a shared reference channel, and a
fresh consumer database. Offline ingestion requires a fresh manifest plus a
shared-directory or Mooncake feature store. Use a single attempt id to derive
all control paths and store ids; never generate it independently in each pool.

Online consumer ranks always use the same path:
`RefDistributor -> per-rank InboxChannel -> StreamingRefQueue -> DPAckController`.
This includes `NPROC_PER_NODE=1`; there is no direct-channel fallback. Rank 0
requires a fresh `DISAGG_DB` and is the sole durable-ledger writer. Before
capture, it publishes
`quantum = NPROC_PER_NODE * training.batch_size * training.accumulation_steps`.
The producer requires `DISAGG_IN_FLIGHT_HIGH_WATERMARK >= quantum`; the default
watermark is `256`. A partial quantum at EOF fails the attempt and cleans its
undispatched feature objects.

Online streams are consume-once: there is no resume and no second consumer pass
over old refs. Always use fresh `DISAGG_REF_CHANNEL`, `DISAGG_DB`, store id,
run id, inbox, and output paths for a retry. Offline manifests remain fixed and
re-iterable, so offline epochs and checkpoint resume retain their normal
semantics.

See the [disaggregated training guide](../../docs/basic_usage/disaggregated_training.md)
for Mooncake prerequisites plus the full server, transport, and role commands
before running either wrapper. Run a launcher with `--help` for its concise
variable list.
