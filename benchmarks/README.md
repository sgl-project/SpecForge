# Benchmarking for Speculative Decoding

## Setup

You can create a new environment and install SGLang with the following command:

```bash
# create virtual env
uv venv sglang -p 3.11
source sglang/bin/activate

# install sglang
uv pip install "sglang[all]>=0.4.9.post2"
```

You can serve your trained model with SGLang with the following command by replacing the `<target-model-path>` and `<draft-model-path>` with the actual path to the target model and draft model.

```bash
python3 -m sglang.launch_server \
    --model <target-model-path>  \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path <draft-model-path> \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --mem-fraction-static 0.75 \
    --cuda-graph-max-bs 2 \
    --tp 1 \
    --context-length 8192 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16
```

## Run Benchmarks

You first need to start the SGLang server:

```bash
python3 -m sglang.launch_server \
    --model <target-model-path>  \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path <draft-model-path> \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --mem-fraction-static 0.75 \
    --cuda-graph-max-bs 2 \
    --tp 8 \
    --context-length 8192 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16
```

Then you can run the benchmarks:

```bash
# GSM8K
python run_gsm8k.py

# MATH-500
python run_math500.py

# MTBench
python run_mtbench.py

# HumanEval
python run_humaneval.py
```
