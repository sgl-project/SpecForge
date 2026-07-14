# Benchmarking speculative decoding

The repository keeps the model-quality and serving-performance benchmark suite
under `benchmarks/`. It is independent from training: train and export a draft
through the unified `specforge` CLI, then benchmark that exported artifact
against an SGLang server.

## Run server and benchmarks together

From the repository root:

```bash
python benchmarks/bench_eagle3.py \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --speculative-draft-model-path /path/to/exported-draft \
    --port 30000 \
    --trust-remote-code \
    --mem-fraction-static 0.8 \
    --tp-size 1 \
    --attention-backend fa3 \
    --config-list 1,0,0,0 1,3,1,4 \
    --benchmark-list mtbench gsm8k:5 ceval:5:accountant \
    --dtype bfloat16
```

Each `--config-list` entry is
`batch-size,num-steps,topk,num-draft-tokens`. Benchmark selectors use
`name[:num-prompts[:subset,...]]`. Available datasets include AIME, C-Eval,
FinanceQA, GPQA, GSM8K, HumanEval, LiveCodeBench, MATH-500, MBPP, MMLU,
MMStar, MT-Bench, and SimpleQA.

## Benchmark an existing server

Start SGLang separately, then add `--skip-launch-server`:

```bash
python benchmarks/bench_eagle3.py \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 30000 \
    --config-list 1,3,1,4 \
    --benchmark-list mtbench:5 gsm8k:5 humaneval:5 math500:5 \
    --skip-launch-server
```

Results are written as timestamped JSON under `--output-dir`. The standalone
GPU microbenchmarks `bench_domino_mfu.py`,
`specforge/benchmarks/benchmark_flex_attention.py`, and
`specforge/benchmarks/benchmark_loss.py` cover trainer MFU, attention, and loss
kernel behavior respectively.
