# EAGLE-4 stage-1 training

Stage 1 trains the backbone and a dense, full-vocabulary draft head.
`efficient_lm_head_config` specifies inference export only: it does not create
low-rank parameters or change the training loss. The draft creates full-vocabulary
identity buffers internally; omit `model.vocab_mapping_path`. Candidate-factor
training is outside this implementation.

The Qwen3-4B architecture is defined in `configs/qwen3-4b-eagle4.json`: one draft
layer, vocabulary 151,936, auxiliary layers 1/17/32, fusion normalization,
no rotary embeddings, a 512-token attention window, and eight latent ReLU-squared
experts with two selected per token. Sliding-window training supports `sdpa` and
`flex_attention`.

## Recipes

Prepare the ShareGPT training JSONL at the path specified in the recipe, install
SpecForge and its runtime dependencies, and run commands from the repository root.

For online training, use an SGLang build that supports SpecForge streaming feature
capture, together with Mooncake. The managed-local launcher uses GPU 0 for capture
and GPUs 1 through 7 for training:

```bash
specforge train --config examples/configs/online/disaggregated/managed-local/qwen3-4b-eagle4.yaml
```

For an offline smoke test, first run the feature-capture command in the comments
of `examples/configs/offline/colocated/qwen3-4b-eagle4.yaml`, then:

```bash
specforge train --config examples/configs/offline/colocated/qwen3-4b-eagle4.yaml
```

The offline example captures 256 training samples and runs 20 optimizer steps.
Runtime attention kernels are selected by the installed capture backend;
these recipes do not force a hardware-specific kernel.

## Config-driven SVD export

The Qwen3-4B config uses `efficient_lm_head_config` with `type: low_rank`,
`rank: 128`, and `num_candidates: 128`. Runtime training checkpoints contain the
trained dense weights and optimizer state, without low-rank factor tensors.

```bash
specforge export --to sglang \
  --checkpoint ./outputs/qwen3-4b-eagle4-offline \
  --draft-config configs/qwen3-4b-eagle4.json \
  --output-dir ./outputs/qwen3-4b-eagle4-svd \
  --svd-device cuda
```

Export computes an unweighted FP32 SVD of the trained dense head `W`. For
`W = U S Vh`, it writes `B = U[:, :rank] S[:rank]` and `A = Vh[:rank]`, stored in
`W`'s dtype. These are `efficient_lm_head.weight` and
`efficient_lm_head_down_proj.weight`. Retrieval uses `B A h` to select candidates;
the original, unchanged dense head rescores their rows. Omitting the head config
produces a dense export. The SVD device defaults to CPU; `cuda` uses an allocated
GPU and avoids the potentially expensive CPU decomposition.

`--to hf` also derives SVD factors when configured and includes the target
embedding (supply `--embedding-source` if the checkpoint excludes it). These
factor-bearing exports are inference artifacts. Resume dense training from its
runtime checkpoint, not the inference export. Export never modifies that source
checkpoint or adds factors to its optimizer.

`configs/qwen3.8-27b-eagle4.json` reproduces the published
[EAGLE-Corp config](https://huggingface.co/EAGLE-Corp/Qwen3.8-27B-EAGLE4/blob/f58a5985223040552ad404fba98f0dd27f5f676a/config.json).
The offline recipe is
`examples/configs/offline/colocated/qwen3.8-27b-eagle4.yaml`. It requires the matching
`Qwen/Qwen3.8-27B` target, tokenizer, capture layers, and embedding key. It is not interchangeable with
Qwen3-4B: it uses different hidden/vocabulary sizes, RoPE, a dense MLP, and a 4,096-token window.

## Recorded serving smoke tests

The existing stage-1 checkpoint and its rank-128 SVD conversion were tested with
SGLang and static EAGLE-Harness scheduling: BF16, one GPU, TP=1, concurrency=1,
eager execution, three draft steps, top-k=1, and four draft tokens. Decoding was
greedy with thinking disabled and a 256-token output limit. All 33 prompts were
warmed up before timing.

Dense serving completed 33/33 requests at 88.03 output tokens/s and 1.862 tokens
per verification round. SVD serving completed 33/33 at 86.17 tokens/s and 1.839
per round. Acceptance includes the bonus token. These are integration smoke tests,
not optimized throughput measurements or checks of exact target-only equivalence.


## Training-to-serving pipeline validation

A fresh Qwen3-4B smoke run exercised feature capture, 20 optimizer steps of dense
stage-1 training, config-driven rank-128 SVD export, and SGLang serving. The run
used the offline recipe with 32 captured samples, sequence length 512, batch size
1, TTT length 7, and learning rate 1e-4. Runtime checkpoints contained no retrieval
factors; the export preserved every non-embedding trained tensor exactly and added
factors of shapes `[151936, 128]` and `[128, 2560]`.

With the serving settings above and a 64-token output limit, the fresh export
completed 33/33 requests at 49.23 output tokens/s and 1.072 tokens per verification
round (including the bonus token). This validates the pipeline; 20 training steps
do not establish model quality. These results are separate from the trained
checkpoint measurements above.

The matching Qwen3.8-27B pipeline also completed 20 optimizer steps and exported
factors of shapes `[248320, 128]` and `[128, 5120]`, with the same dense-weight
preservation checks. Serving completed 8/8 generic prompts at 21.25 output tokens/s
and 1.039 tokens per verification round, using the same serving settings and
64-token output limit. Its prompt set differs from the 4B test.

Both smoke runs had finite losses and gradients. Four truncated batches per run
contained no supervised tokens; 16 of the 20 optimizer steps had nonzero loss.
