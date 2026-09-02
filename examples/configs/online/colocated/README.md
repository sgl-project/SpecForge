# Online colocated recipes

Every trainer rank loads one SGLang target shard next to its FSDP draft shard
and captures hidden states in process; no producer role, feature transport, or
second GPU pool is involved. `training.tp_size` is the target-TP island width,
and the trainer world size must be divisible by it.

| Recipe | Target | Topology |
| --- | --- | --- |
| [`qwen3-8b-dspark-colocated.yaml`](qwen3-8b-dspark-colocated.yaml) | Qwen3-8B | 8 islands of TP1 on one 8xH200 node |
| [`kimi-k3-dspark-colocated.yaml`](kimi-k3-dspark-colocated.yaml) | Kimi K3 | 4 islands of TP8 with `HYBRID_SHARD` on 4x8 B300 |

See [Colocated online training](../../../../docs/basic_usage/colocated_training.md)
for memory sizing and scaling guidance.
