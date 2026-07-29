# Disaggregated examples

Disaggregated training uses the same typed config and public command as every
other run:

```bash
specforge train -c examples/configs/qwen3-8b-dflash-disaggregated.yaml
```

For a single trainer node, that command supervises the SpecForge producer and
consumer together. The producer is one direct process; the consumer topology
comes from `deployment.trainer` and is launched through torch distributed when
`nproc_per_node > 1`.

The scripts in this directory are optional thin examples. The single-node
wrappers only add the config path and forward arguments to `specforge train`;
they contain no second trainer, torchrun construction, or transport validation:

```bash
CONFIG=examples/configs/qwen3-8b-dflash-disaggregated.yaml \
  examples/disagg/run_online.sh

CONFIG=examples/configs/qwen3-8b-eagle3-offline-disaggregated.yaml \
  examples/disagg/run_offline.sh
```

For offline ingestion and training on separate physical nodes, invoke the same
rank dispatcher on both nodes. The cluster launcher supplies
`RCLI_NODE_RANK=0` for the producer and `RCLI_NODE_RANK=1` for the consumer:

```bash
rcli exec --per-node <job> \
  'CONFIG=examples/configs/qwen2.5-7b-eagle3-offline-disaggregated.yaml bash examples/disagg/run_offline_2node.sh'
```

This wrapper only maps rank to `specforge train --role`; the YAML still owns
the consumer process count and the shared `control_dir`/`store_root`. Both
nodes must resolve those paths to the same storage and start from a fresh
attempt directory. `NODE_RANK`/`NUM_NODES` are accepted when the scheduler does
not expose the corresponding `RCLI_*` variables.

## Explicit roles

Use the same YAML when producer and consumer belong to different pools:

```bash
specforge train -c examples/configs/qwen3-8b-dflash-disaggregated.yaml \
  --role producer

specforge train -c examples/configs/qwen3-8b-dflash-disaggregated.yaml \
  --role consumer
```

For a multi-node consumer, `deployment.trainer.nnodes`,
`nproc_per_node`, `master_addr`, and `master_port` remain identical on every
node. Supply only the node-local rank at invocation time:

```bash
# Trainer node 0
specforge train -c run.yaml --role consumer --node-rank 0

# Trainer node 1
specforge train -c run.yaml --role consumer --node-rank 1
```

Automatic `--role both` is deliberately single-node. The CLI does not SSH into
other machines or start scheduler jobs.

For the common two-physical-node demonstration (inference services + producer
on rank 0, one DP trainer pool on rank 1), run the checked-in infrastructure
wrapper once per node through the cluster launcher:

```bash
export DISAGG_STORE_ID=qwen3-8b-two-node-attempt-001
export DISAGG_RUN_ROOT=/shared/specforge/$DISAGG_STORE_ID

# The launcher supplies RCLI_NODE_RANK, RCLI_NUM_NODES, and RCLI_HEAD_IP.
rcli exec --per-node <job> \
  'bash examples/disagg/run_qwen3_8b_dflash_disagg_2node.sh'
```

The wrapper owns Mooncake/SGLang readiness and cross-node lifecycle only. Both
roles still execute `specforge train -c ...`; it contains no legacy Python
trainer and constructs no `torchrun` command. Set `SERVER_GPUS`, `TRAINER_GPUS`,
`TRAINER_NPROC`, and `CONFIG` when the allocation differs. Both nodes must see
the fresh `DISAGG_RUN_ROOT` path. References stay under its shared `control`
directory, while the wrapper's lifecycle markers stay at the shared run root.
The consumer's SQLite/WAL and rank inboxes default to the trainer-node-local
`/tmp/specforge/$DISAGG_STORE_ID/consumer-state`; override
`DISAGG_CONSUMER_STATE_DIR` or `LOCAL_SCRATCH` when `/tmp` is unsuitable.
Node-local consumer state currently supports one trainer node only.

For the GLM-5.2 DSpark 120K recipe, use TP8 on the target node and USP8/FSDP
on the trainer node. H200 needs a lower SGLang static-memory fraction than the
Qwen default so decode-graph capture and the long-context feature payload both
retain headroom:

```bash
export CONFIG=examples/configs/glm-5.2-dspark-online-120k-usp.yaml
export TARGET_MODEL_PATH=zai-org/GLM-5.2-FP8
export SERVER_GPUS=0,1,2,3,4,5,6,7 SERVER_TP=8
export SERVER_MEM_FRACTION=0.75
export SERVER_DISABLE_CUDA_GRAPH=1
export SERVER_MAX_TOTAL_TOKENS=120064
export CAPTURE_LAYER_IDS="1 19 38 57 76"
export TRAINER_GPUS=0,1,2,3,4,5,6,7 TRAINER_NPROC=8
export MOONCAKE_GLOBAL_SEGMENT_SIZE=137438953472
export MOONCAKE_DEFAULT_KV_LEASE_TTL=600000
```

Then invoke `run_qwen3_8b_dflash_disagg_2node.sh` once per node as above. The
wrapper name is historical; `CONFIG`, `TARGET_MODEL_PATH`, and the topology
variables select the GLM/DSpark stack. The token cap preserves one 64-token
page of KV-cache margin. Disabling CUDA graphs releases the prefill/decode
graph state for the much larger capture payload; this one-step validation
recipe does not need graph amortization. The patched SGLang request receiver
uses its bounded same-node message queue for TP spec-capture traffic, so keep
`SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=1` (the v0.5.14 default) for TP8.

## External and managed-local services

By default, online capture requires an already-running Mooncake deployment and
patched SGLang capture server. Those long-lived services are not started or
stopped by `specforge train`. Put stable, non-secret topology in the typed
`deployment.disaggregated` section; inject authentication tokens, each node's
Mooncake hostname, and device visibility through the deployment environment.
The checked-in external-service recipes point at the standard local demo ports;
replace those endpoint fields, or override them with environment values, for a
remote deployment.

For a self-contained single-node development run, the managed-local recipes
record Mooncake, one or more capture servers, their GPU placement, and the DP
trainer in one YAML:

```bash
specforge train -c \
  examples/configs/qwen3-8b-dflash-1server-dp7-disaggregated.yaml

specforge train -c \
  examples/configs/qwen3-8b-domino-1server-dp7-disaggregated.yaml
```

These recipes preserve the old DFlash and Domino one-server + DP7
self-contained topologies. The genuine two-server Domino recipe is:

```bash
specforge train -c \
  examples/configs/qwen3-8b-domino-multiserver-disaggregated.yaml
```

Qwen3.6 DFlash has both one-server and larger two-TP2-server managed recipes:

```bash
specforge train -c \
  examples/configs/qwen3.6-27b-dflash-1server-dp2-disaggregated.yaml

specforge train -c \
  examples/configs/qwen3.6-27b-dflash-multiserver-disaggregated.yaml
```

The first command preserves the historical one-server + DP2 self-contained
topology; the second owns two TP=2 servers plus the DP2 trainer.

That opt-in profile starts, health-checks, and cleans up the owned local
services. It does not change the default external-service boundary or attempt
to schedule services on remote hosts.

The strict e2e gate at
`scripts/gates/run_disaggregated_overfit_gate.sh` retains full local test-stack
automation: it starts and health-checks Mooncake and SGLang, runs the unified
producer/consumer entry, verifies training and serving, and cleans up owned
processes. That test harness is not the production service supervisor.

Online configs use Mooncake. Offline configs may use either a typed
`shared_dir` store or Mooncake. `deployment.disaggregated.control_dir` is the
one attempt root from which the launcher derives the reference channel or
manifest and lifecycle markers. An online deployment places the consumer's
rank-0 SQLite/WAL in the node-local
`deployment.disaggregated.consumer_state_dir`. Multi-node trainers require this
field and keep their rank inboxes under the shared `control_dir`. Always choose
fresh control and consumer-state directories for a new attempt. An offline
Mooncake producer must also set a positive
`deployment.disaggregated.producer_segment_size`; online roles and the offline
consumer use zero because they do not own feature allocations.

Use `specforge train -c run.yaml --plan` to inspect the resolved role and
process commands without starting workers. Secret-looking override values and
URL userinfo are redacted.

See the [disaggregated training guide](../../docs/basic_usage/disaggregated_training.md)
for service prerequisites, recovery rules, and the online/offline data-plane
contracts.
