---
name: launch-qwen3-4b-dspark-disagg
description: >-
  Launch SpecForge DSpark training for Qwen3-4B from
  examples/configs/qwen3-4b-dspark-disaggregated.yaml, either the full
  10000-step run or a short 4-step debug run. Use when launching,
  smoke-testing, or studying this disaggregated recipe.
disable-model-invocation: true
---

# Launch Qwen3-4B DSpark disaggregated training

One command owns the whole stack. The recipe's
`deployment.disaggregated.managed_local` block makes `specforge train` start
and supervise, in order:

1. a Mooncake master (loopback TCP; RDMA loopback is broken in this KVM guest),
2. one SGLang capture server on GPU 0 (TP=1, port 30000, `mem_fraction_static: 0.7`),
3. a CPU producer that streams captured features into Mooncake,
4. a 7-rank data-parallel trainer on GPUs 1–7 (global batch 7).

All 8 GPUs must be idle before launch. A failure in any role tears down the
whole owned stack.

## Prerequisites

- `specforge` comes from the repo venv, not the system PATH:

```bash
cd /personal/SpecForge && source .venv/bin/activate
```

- Dataset at `./cache/dataset/sharegpt_train.jsonl` (already prepared in this
  checkout).
- `WANDB_API_KEY` in the environment for the full run (the recipe sets
  `tracking.report_to: wandb`; never put the key in YAML). The debug run below
  disables W&B instead.
- Fresh attempt storage: the launcher rejects a run whose control directory
  already contains `consumer.sqlite` (or its WAL/SHM sidecars). Archive or
  delete the previous attempt's directories first — see each section.

Inspect the resolved process plan without starting anything:

```bash
specforge train -c examples/configs/qwen3-4b-dspark-disaggregated.yaml --plan
```

## Full run (10000 steps)

`training.max_steps: 10000` binds long before the `num_epochs: 6` cap.
Checkpoints land every 1000 steps under
`outputs/qwen3-4b-dspark-disaggregated/qwen3-4b-dspark-disaggregated-stepN`
plus a rolling `-latest`.

```bash
cd /personal/SpecForge && source .venv/bin/activate
export WANDB_API_KEY=...   # from the protected environment, not YAML

RUN=outputs/qwen3-4b-dspark-disaggregated
[ -e "$RUN" ] && mv "$RUN" "$RUN-prev-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN"

specforge train -c examples/configs/qwen3-4b-dspark-disaggregated.yaml \
  |& tee "$RUN/supervisor.log"
```

## Debug run (4 steps)

Runs the identical stack but stops after 4 optimizer steps, logs every step,
skips W&B, and keeps all state under a separate `outputs/qwen3-4b-dspark-debug`
tree so the real run's checkpoints are untouched. `run_id` also namespaces the
Mooncake store, so a debug run cannot collide with a concurrent real attempt's
objects (the GPUs still conflict — run one at a time).

```bash
cd /personal/SpecForge && source .venv/bin/activate

DEBUG=outputs/qwen3-4b-dspark-debug
rm -rf "$DEBUG"
mkdir -p "$DEBUG"

specforge train -c examples/configs/qwen3-4b-dspark-disaggregated.yaml \
  run_id=qwen3-4b-dspark-debug \
  output_dir="$DEBUG" \
  deployment.disaggregated.control_dir="$DEBUG/control" \
  training.max_steps=4 \
  training.log_interval=1 \
  tracking.report_to=none \
  |& tee "$DEBUG/supervisor.log"
```

The `rm -rf` makes the debug loop re-runnable: every attempt needs a fresh
control directory. Expect a few minutes of SGLang/Mooncake startup before the
first step; each step consumes 7 samples (7 DP ranks x batch 1).

## Where to look while it runs

- `tee`'d supervisor output: producer/consumer progress, per-step loss and
  accuracy lines from the trainer (rank 0).
- `<control_dir>/logs/capture-server-0.log` — managed SGLang capture server.
- `<control_dir>/logs/mooncake.log` — managed Mooncake master.
- `<control_dir>/refs.jsonl*` — the reference channel; `.consumed_count`
  advances as the consumer commits optimizer steps.
- Full run only: W&B project `specforge`, run `qwen3-4b-dspark-disaggregated`.

## Common failures

- "consumer database ... must not exist": stale attempt storage; archive or
  delete the run directory as shown above.
- Rejected world size / rendezvous errors: a partial torchrun environment is
  set. `unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT NODE_RANK`
  and relaunch.
- Capture server never healthy: check GPU 0 is free and read
  `capture-server-0.log`; the managed server needs the patched SGLang with
  `--enable-spec-capture` support installed in this environment.
- Managed port already bound: the supervisor owns ports 35551/35880/35903
  (Mooncake) and 30000 (capture server); kill leftovers from a previous
  attempt before relaunching.
