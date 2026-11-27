# Train Parallel Reconsideration

[Parallel Reconsideration](https://arxiv.org/abs/2611.05724v1) adds a learned
self-query and halt gate to a one-layer EAGLE-3.1 draft. All depths share the
decoder and LM head. Reconsideration keeps the RoPE position fixed, reads the
committed prefix and its own key, and carries the sampled halt feature and KV
together into the next step.

Training starts the ordinary and reconsideration rollouts from the same
target-derived feature at each origin. Each rollout samples its own draft
tokens. The reconsideration rollout keeps earlier sampled KV commitments.
The objective uses hard teacher argmax labels at EAGLE's original aligned
positions. The paper leaves teacher rescoring on diverged prefixes unspecified,
so the recipe scores both rollouts against the teacher on the recorded sequence.
The common seed contributes the one-pass CE once. The routed deeper CE, gate
loss, and minibatch marginal entropy follow Eq. (6).
The training accuracy is measured on teacher tokens in the draft vocabulary.

## One-GPU smoke configuration

The pilot uses Qwen2.5-0.5B-Instruct, a 512-token cap, and three rollout steps.
It retains the paper's two reconsideration passes and balance coefficient of
`0.01`. The EAGLE capture layout is unchanged.

Install the pinned SGLang capture plugin and Mooncake following the
[offline capture guide](../../../../docs/sections/basic_usage/offline_capture.md).
Start the Mooncake master in its own terminal.

```bash
mooncake_master --enable-http-metadata-server=true
```

Export these settings in both the server and client terminals.

```bash
export MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:50051
export MOONCAKE_METADATA_SERVER=http://127.0.0.1:8080/metadata
export MOONCAKE_LOCAL_HOSTNAME=127.0.0.1
export MOONCAKE_PROTOCOL=tcp
```

Start the target in the SGLang environment.

```bash
CUDA_VISIBLE_DEVICES=0 SPECFORGE_SPEC_CAPTURE=1 SGLANG_SPEC_CAPTURE_GPU_PUT=0 \
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct --skip-tokenizer-init \
  --port 30000 --tp-size 1 --speculative-algorithm EAGLE3 \
  --enable-return-aux-hidden-states --return-aux-hidden-state-layer-ids 1 11 20 \
  --return-hidden-states-mode full --chunked-prefill-size -1
```

Capture a bounded dataset from the repository root in the client environment.

```bash
python scripts/prepare_data.py --dataset ultrachat --sample-size 2048
python scripts/prepare_hidden_states.py \
  -c examples/configs/offline/colocated/qwen2.5-0.5b-parallel-reconsideration.yaml \
  --server-urls http://127.0.0.1:30000 \
  --data-path cache/dataset/ultrachat_train.jsonl --chat-template qwen \
  --max-length 512 --batch-size 4 \
  --output-path cache/hidden_states/qwen2.5-0.5b-parallel-reconsideration
```

Stop the target service before training on the same GPU. Save the trainer log
alongside the native checkpoints.

```bash
set -o pipefail
run_dir=outputs/qwen2.5-0.5b-parallel-reconsideration
mkdir -p "$run_dir"
CUDA_VISIBLE_DEVICES=0 bash examples/run_qwen2_5_0_5b_parallel_reconsideration.sh \
  cache/hidden_states/qwen2.5-0.5b-parallel-reconsideration "$run_dir" \
  2>&1 | tee "$run_dir/train.log"
```

Pass `resume=$run_dir` after the two positional arguments to resume a canonical
checkpoint. The saved draft includes the self-query, halt gate, and vocabulary
mapping. Serving the reconsidered tree requires a matching inference runtime.
