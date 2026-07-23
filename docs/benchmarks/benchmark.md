# Benchmarking inference serving

Use `benchmarks/bench_eagle3.py` to compare EAGLE3 serving configurations and
dataset quality. Use `specforge benchmark` to measure any existing
SGLang deployment without assuming a particular speculative algorithm.

| Runner | Server lifecycle | Measurements | Output |
| --- | --- | --- | --- |
| `python benchmarks/bench_eagle3.py` | Launches SGLang per configuration or uses an existing server | Latency, output throughput, acceptance length, and dataset accuracy when available | Timestamped JSON under `--output-dir` |
| `specforge benchmark` | Uses an existing SGLang server | Aggregate output throughput, acceptance length, and verification count when reported by SGLang | Console and optional `--output-json` |

## EAGLE3 benchmark matrix

### Launch SGLang for each configuration

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

- `--config-list` uses `batch-size,num-steps,topk,num-draft-tokens`; `1,0,0,0` is a target-only baseline.
- `--benchmark-list` uses `name[:num-prompts[:subset,...]]`.
- Supported datasets are AIME, C-Eval, FinanceQA, GPQA, GSM8K, HumanEval, LiveCodeBench, MATH-500, MBPP, MMLU, MMStar, MT-Bench, and SimpleQA.
- The runner starts a fresh server for each configuration, runs every requested dataset, flushes the cache between datasets, and writes results under `--output-dir`.

### Use an existing SGLang server

```bash
python benchmarks/bench_eagle3.py \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --port 30000 \
  --config-list 1,3,1,4 \
  --benchmark-list mtbench:5 gsm8k:5 humaneval:5 math500:5 \
  --skip-launch-server
```

With `--skip-launch-server`, the runner does not change the server's speculative settings. The first `--config-list` entry supplies only the request batch size.

## General SGLang benchmark

The existing-server runner supports target-only and speculative deployments on
GSM8K, MATH-500, HumanEval, MBPP, and MT-Bench.

```bash
specforge benchmark \
  --model Qwen/Qwen3-8B \
  --dataset gsm8k \
  --base-url http://127.0.0.1:30000 \
  --num-prompts 1024 \
  --concurrency 16 \
  --output-json ./qwen3-8b-gsm8k.json
```

- Start SGLang with the target-only or speculative configuration you want to measure; this command does not launch or reconfigure the server.
- The runner flushes the server cache, runs one concurrency-sized warmup batch, excludes warmup from the measurement, and then sends `--num-prompts` requests.
- The report includes output-token throughput and includes average acceptance length and speculative verification count when SGLang returns those fields.

## Comparing results

Measure target-only and speculative decoding with the same target revision,
tokenizer and chat template, prompts, sampling parameters, output length,
hardware, tensor parallelism, and concurrency. For the EAGLE3 matrix, include a
zero-step configuration in one run. With the general SGLang runner, benchmark
matched target-only and speculative servers separately and compute speedup from
their throughput results.

Do not compare absolute throughput across the matrix and existing-server runners
because their batching and request scheduling differ.

## Safety

The EAGLE3 HumanEval and MBPP scorers execute model-generated Python. Run them
only in an isolated environment without credentials or production data. The
general SGLang runner measures decoding and does not execute generated code.
