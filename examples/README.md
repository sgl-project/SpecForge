# Run SpecForge Examples

This folder contains the examples of running SpecForge on different models. The scripts can be invoked by the following command:

```bash
bash examples/<script-name>.sh [NUM_GPUS] [TP_SIZE]
```

We use the ShareGPT dataset for all the examples for now, but you can replace it with more robust datasets such as perfectblend, magpie-qwen2.5-pro-1m-v0.1, etc.

## D-PACE

DFlash training also supports the D-PACE loss (Dynamic Position-Aware Cross-Entropy). Add `--loss-type dpace` (optionally `--dpace-alpha`, default 0.5) to any `scripts/train_dflash.py` command. `--loss-decay-gamma` is DFlash-only and is ignored by D-PACE variants. Ablation variants are available via `--loss-type dpace-cumulative-confidence-only` and `--loss-type dpace-continuation-value-only`. A ready-to-run example:

```bash
NUM_GPUS=8 DPACE_ALPHA=0.5 bash examples/run_qwen3_8b_dpace_online.sh
```
