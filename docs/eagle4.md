# EAGLE-4 stage-1 training

Stage 1 trains a dense, full-vocabulary draft head. Candidate-factor training and
SVD conversion are outside this implementation. Do not set
`efficient_lm_head_config` or `model.vocab_mapping_path`: the draft creates its
full-vocabulary identity buffers directly.

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
It is neither the one-epoch training run nor the held-out evaluation described
below. Runtime attention kernels are selected by the installed capture backend;
these recipes do not force a hardware-specific kernel.

## Historical validation and recipe alignment

The archived stage-1 run used the same architecture JSON, one capture GPU and
seven trainer ranks, one ShareGPT epoch, sequence length 2,048, TTT length 7,
batch size 1 per rank, learning rate 0.0001, gradient clipping 0.5, and seed 0.
It completed 16,947 optimizer steps. Its capture configuration explicitly selected
a runtime-specific attention kernel; the current recipe leaves that selection
to the runtime. The old online setup required an external identity vocabulary
mapping; the current implementation creates that mapping internally.

The historical checkpoint predates the correction of the fusion residual to
`z + attention(norm(z))`. No full training run with the corrected residual has
been completed. Its training and evaluation metrics therefore describe that
historical checkpoint, not a fresh execution of the current recipe.

Teacher-forced offline evaluation used a separate set of 256 held-out
conversations, excluded from training by ID and message fingerprint, with sequence
length 2,048 and TTT length 7. A rank-128 SVD conversion retrieving 128 candidates
was evaluated separately; it does not imply candidate-head training support here.

Serving validation was a separate integration smoke test using the existing dense
checkpoint and its SVD conversion with the corrected inference residual. It used
SGLang with static EAGLE-Harness scheduling, BF16, one GPU, tensor parallelism 1,
concurrency 1, eager execution, three draft steps, top-k 1, and four draft tokens.
Decoding was greedy with thinking disabled and a 256-token output limit. All 33
prompts (32 held-out first-user-turn prompts and one long-context prompt) were
warmed up before timing. Dense serving completed 33/33 requests at 88.03 output
tokens/s and 1.862 tokens per verification round; SVD serving completed 33/33 at
86.17 tokens/s and 1.839 tokens per round. Acceptance includes the bonus token.

The training YAML files do not configure this serving protocol. These results
are smoke-test evidence, not an optimized benchmark, exact target-only output
equivalence check, or an end-to-end rerun of the revised recipes.
