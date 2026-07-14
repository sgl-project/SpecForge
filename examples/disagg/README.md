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

See the [disaggregated training guide](../../docs/basic_usage/disaggregated_training.md)
for Mooncake prerequisites plus the full server, transport, and role commands
before running either wrapper. Run a launcher with `--help` for its concise
variable list.

`run_domino_dflash_serving_gate.sh` is a post-training serving validation gate;
it is not a training entry point.
