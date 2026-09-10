# Benchmarking inference serving

`specforge benchmark` measures an SGLang server on one or more tasks and
reports output throughput, acceptance length, and, for scored tasks, accuracy.
It can measure a server you already started, or launch one server per
speculative-decoding configuration and sweep a matrix.

## Quick start

Measure a running server:

```bash
specforge benchmark \
  --model Qwen/Qwen3-8B \
  --base-url http://127.0.0.1:30000 \
  --task gsm8k:200 --task mtbench \
  --concurrency 16
```

Sweep target-only and EAGLE3 configurations from a config file:

```bash
specforge benchmark --config examples/benchmarks/eagle3_matrix.yaml
```

Every run prints a summary table and writes a JSON report to
`output.dir/<output.name>_<timestamp>.json`.

```text
config              task            samples     tokens      tok/s   accept   accuracy
------------------------------------------------------------------------------------
baseline            gsm8k               200     61234      812.3        -      92.5%
eagle3-s3-k1-d4     gsm8k               200     61102     2154.7    3.412      92.0%
```

## Configuration

A benchmark is one validated config. Load it from YAML or JSON with `--config`,
set the common fields with flags, and adjust anything else with dotted
`section.field=value` overrides; later sources win in that order.

```yaml
model: meta-llama/Llama-3.1-8B-Instruct
draft_model: /path/to/exported-draft

tasks:
  - name: mtbench
  - name: gsm8k
    num_samples: 200
  - name: ceval
    num_samples: 50
    subset: [accountant, law]

server:
  launch: true
  base_url: http://127.0.0.1:30000
  args: [--tp-size, "1", --mem-fraction-static, "0.8", --attention-backend, fa3]

matrix:
  - {label: baseline}
  - {steps: 3, topk: 1, draft_tokens: 4}
  - {steps: 5, topk: 4, draft_tokens: 8, batch_size: 8}

sampling:
  temperature: 0.0
  max_new_tokens: 2048

concurrency: 1
output:
  dir: ./benchmark_results
  name: llama3.1-8b-eagle3
```

| Section | Fields |
| --- | --- |
| top level | `model` (tokenizer, chat template, served model), `draft_model`, `concurrency`, `warmup`, `enable_thinking`, `trust_remote_code`, `seed`, `plugins` |
| `tasks[]` | `name`, `num_samples`, `subset`, `max_new_tokens` |
| `server` | `base_url`, `launch`, `launch_timeout_seconds`, `request_timeout_seconds`, `args` (verbatim `sglang.launch_server` flags), `env` |
| `matrix[]` | `label`, `batch_size`, `algorithm`, `steps`, `topk`, `draft_tokens`; `steps: 0` is the target-only baseline |
| `sampling` | `temperature`, `top_p`, `top_k`, `max_new_tokens` (global cap over task defaults) |
| `output` | `dir`, `name` |

`matrix` is only consulted with `server.launch: true`. A running server is
measured as-is and its rows are labeled `server`; the benchmark never changes a
server it did not start.

On the command line, `--task NAME[:N[:SUBSET,...]]` is repeatable and replaces
the file's task list, so `--task ceval:50:accountant,law` is the same as the
YAML entry above. `--list-tasks` prints the registry.

## Tasks

| Task | Dataset | Scored |
| --- | --- | --- |
| `gsm8k` | openai/gsm8k test | final number |
| `math500` | HuggingFaceH4/MATH-500 | boxed answer |
| `aime` | Maxwell-Jia/AIME_2024 (32k-token cap) | integer answer |
| `humaneval` | openai/openai_humaneval | runs the unit tests |
| `mbpp` | google-research-datasets/mbpp sanitized | runs the asserts |
| `ceval` | ceval/ceval-exam validation, `subset` = subjects | choice letter |
| `mmlu` | cais/mmlu test, `subset` = HF config names | choice letter |
| `gpqa` | Idavidrein/gpqa main, seeded choice order | choice letter |
| `mmstar` | Lin-Chen/MMStar val (image input) | choice letter |
| `mtbench` | HuggingFaceH4/mt_bench_prompts, two turns | no |
| `simpleqa` | basicv8vc/SimpleQA | no |
| `financeqa` | AfterQuery/FinanceQA | no |
| `livecodebench` | livecodebench/code_generation | no |

Prompts are rendered client-side with the target tokenizer's chat template and
sent to `/generate`, the endpoint that returns speculative-decoding telemetry.
Multi-turn tasks replay each assistant reply into the next turn. `mmstar` needs
a vision-language chat template and passes the image path as `image_data`.

### Adding a task

Subclass `BenchmarkTask`, yield `Sample` objects from `load`, optionally
implement `score`, and register the class. Built-in tasks live in
`specforge/benchmarks/tasks/`; external ones are imported through `plugins`.

```python
from specforge.benchmarks.tasks import TASKS, BenchmarkTask, Sample
from specforge.benchmarks.tasks.answers import extract_final_number, numbers_equal

@TASKS.register
class MyMathTask(BenchmarkTask):
    name = "mymath"
    description = "in-house math set"
    max_new_tokens = 4096

    def load(self):
        for row in self.limit(load_my_rows()):
            yield Sample(turns=[row["question"]], label=row["answer"])

    def score(self, output, label):
        return numbers_equal(extract_final_number(output), label)
```

```yaml
plugins: [my_package.benchmarks]
tasks:
  - name: mymath
```

## Metrics

- **tok/s** is output tokens divided by the wall-clock time of the timed phase.
  With `warmup: true` (the default) one concurrency-sized batch runs first,
  untimed, and the prefix cache is flushed before measuring.
- **accept** is `output_tokens / spec_verify_ct` summed over all requests, so
  it is length-weighted. It is blank when the server reports no speculation.
- **accuracy** is the fraction of scored samples the task marked correct.

## Comparing results

Measure target-only and speculative decoding with the same target revision,
tokenizer and chat template, tasks, sampling parameters, output cap, hardware,
tensor parallelism, and concurrency. A matrix run guarantees this by
construction; for separately started servers, keep the config identical apart
from `server.base_url`.

## Safety

`humaneval` and `mbpp` execute model-generated Python in a subprocess with a
timeout. That isolates the benchmark from hangs, not from malicious code. Run
scored code tasks only in an isolated environment without credentials or
production data.
