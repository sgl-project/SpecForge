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

The two scripts in this directory are optional thin examples. They only add the
config path and forward arguments to `specforge train`; they contain no second
trainer, torchrun construction, or transport validation:

```bash
CONFIG=examples/configs/qwen3-8b-dflash-disaggregated.yaml \
  examples/disagg/run_online.sh

CONFIG=examples/configs/qwen3-8b-eagle3-offline-disaggregated.yaml \
  examples/disagg/run_offline.sh
```

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
the fresh `DISAGG_RUN_ROOT` path.

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
record Mooncake, two capture servers, their GPU placement, and the DP trainer
in one YAML. The Qwen3-8B Domino recipe restores the genuinely two-server
behavior that existed before the legacy shell was reduced to a one-server
launch:

```bash
specforge train -c \
  examples/configs/qwen3-8b-domino-multiserver-disaggregated.yaml
```

The larger Qwen3.6 DFlash topology uses two TP=2 capture servers:

```bash
specforge train -c \
  examples/configs/qwen3.6-27b-dflash-multiserver-disaggregated.yaml
```

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
manifest, consumer SQLite database, and inbox directory. Always choose a fresh
control directory for a new attempt. An offline Mooncake producer must also set
a positive `deployment.disaggregated.producer_segment_size`; online roles and
the offline consumer use zero because they do not own feature allocations.

Use `specforge train -c run.yaml --plan` to inspect the resolved role and
process commands without starting workers. Secret-looking override values and
URL userinfo are redacted.

See the [disaggregated training guide](../../docs/basic_usage/disaggregated_training.md)
for service prerequisites, recovery rules, and the online/offline data-plane
contracts.
