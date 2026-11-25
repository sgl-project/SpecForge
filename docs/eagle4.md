# EAGLE-4 training

Stage 1 trains a dense, full-vocabulary draft head and remains the default.
`efficient_lm_head_config` alone describes stage-1 SVD export and does not change
the training objective. Set `eagle4_training_stage: stage2` in the draft JSON to
train candidate factors. Omit `model.vocab_mapping_path`; the draft creates its
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

## Stage-2 candidate training

Stage 2 trains a small head to retrieve vocabulary candidates for a frozen
stage-1 drafter. The full vocabulary head remains the scoring head: the low-rank
head selects which tokens to score, and the corresponding rows of the full head
provide their logits. Training therefore optimizes candidate coverage rather
than requiring the low-rank scores to reproduce every full-head logit.

### Head structure and initialization

For vocabulary size `V`, draft hidden size `H`, rank `r`, and candidate budget `C`,
the model keeps three weight matrices:

- `W = lm_head.weight`, shape `[V, H]`: the frozen full vocabulary head.
- `A = efficient_lm_head.weight`, shape `[V, r]`: the trainable vocabulary factor.
- `B = efficient_lm_head_down_proj.weight`, shape `[r, H]`: the trainable down projection.

For a normalized draft hidden vector `h`, the full-head logits are `z = W h`
and the retrieval scores are `s = A (B h)`. At inference, the top `C` entries of
`s` select candidates; their exact logits are computed using the selected rows
of `W`. The dense head must be retained in the checkpoint.

With `type: low_rank`, missing factor weights are initialized from the dense
head using an FP32 truncated SVD: `W ≈ U_r Σ_r V_r^T`, with
`A = U_r Σ_r` and `B = V_r^T`. Factors are copied into the model's parameter
dtype. If either factor is missing during warm start, both are SVD-initialized;
if both are present, their loaded values are preserved. SVD supplies an initial
matrix approximation; subsequent training specializes the factors for retrieval.

### Frozen drafter and training flow

Only `A` and `B` receive parameter updates. Embeddings, feature projection,
normalization layers, attention, MoE experts and router parameters, and the full
head remain frozen. Router load-balancing bias updates are explicitly disabled:
freezing parameters alone would not prevent this buffer from changing.

Stage 2 uses the same target-feature capture and token/loss-mask format as stage 1.
Offline training reads previously captured target features; online training
receives them from the capture service. At each training-time rollout (TTT) depth:

1. Run the frozen drafter using target features initially and draft hidden-state
   feedback at subsequent depths, with the training sequence's shifted token inputs.
2. Detach the draft hidden states, apply the frozen output normalization, and
   compute both `z` and `s` on the same hidden states.
3. Use the frozen drafter's full head to choose and weight positive vocabulary
   tokens, then compute the cutoff retrieval loss below.
4. Apply the supervision mask and backpropagate into `A` and `B` only.

The full-head teacher here is the **drafter's** `W`, not the target model's LM
head. This is a teacher-forced training rollout, not an online acceptance-length
measurement. Logit normalization and next-depth hidden-state feedback remain
separate; feedback follows the drafter's `norm_output` configuration.
Per-depth masked losses are summed with equal weights for this retrieval
objective, rather than the usual EAGLE depth decay.

### Cutoff retrieval loss

For each supervised hidden state, let `P` be the top `m` tokens under the frozen
full-head logits, where `m = min(32, floor(C / 4))`. Training requires
`4 <= C < V`. With `C = 128`, there are 32 positives.

For each positive token `i`, its weight is `p_i = softmax(z)_i` at temperature 1.
These are probabilities normalized over the **full vocabulary**, not renormalized
over the positive set. Exclude all positives from the retrieval scores, then
choose the `(C - m + 1)`-th highest remaining score as the cutoff `tau`.
For `C = 128`, this is the 97th highest non-positive score.

The loss for one hidden state is:

```text
L(h) = sum over i in P:
         p_i * 1[s_i <= tau] * softplus(tau - s_i)
```

The objective raises positives that fall at or below the candidate boundary and
lowers the competing cutoff score. Positives already above the cutoff contribute
zero. Putting all `m` positives strictly above this cutoff leaves at most `C - m`
non-positive tokens ahead of them, placing the positives within the top `C`.

Positive selection, probability weights, cutoff selection, and the comparison
mask are stop-gradient. The **value** of the selected cutoff score still receives
gradients, as do the positive scores. Each depth's loss is averaged over the
supervised positions using `loss_mask`; padding and unsupervised positions do
not contribute. Training computes full-vocabulary scores to define this loss;
candidate-only dense scoring is the inference path.

### Configuration and running

On this branch, `training.strategy: eagle4` together with
`eagle4_training_stage: stage2` and `efficient_lm_head_config` in the draft JSON
selects candidate-factor training.
For example:

```json
"eagle4_training_stage": "stage2",
"efficient_lm_head_config": {
  "type": "low_rank",
  "rank": 128,
  "num_candidates": 128
}
```

Use [the candidate draft config](../configs/qwen3-4b-eagle4-candidates.json)
with a compatible trained dense drafter. Keep `draft_vocab_size` equal to
`vocab_size`, and omit `model.vocab_mapping_path`: full-vocabulary identity
buffers are created internally. The architecture, vocabulary, and target feature
layers must match the source checkpoint; adding the head configuration does not
convert an incompatible drafter architecture.

Set `model.draft_checkpoint_path` to the actual checkpoint/export directory
containing the trained draft weights. The recipes contain example paths that
must be adjusted to the available checkpoint and data:

```bash
# Offline: reuse stage-1-compatible captured target features.
specforge train --config examples/configs/offline/colocated/qwen3-4b-eagle4-candidates.yaml

# Online: capture target features while training the candidate factors.
specforge train --config examples/configs/online/disaggregated/managed-local/qwen3-4b-eagle4-candidates.yaml
```

`model.draft_checkpoint_path` is a weights-only warm start: it does not restore
optimizer/scheduler state, step counters, data position, or RNG state. It is
different from `training.resume_from`, the full-resume setting; the two settings
are mutually exclusive. Do not treat a weights-only restart as continuation of
the same optimizer trajectory. Keep both learned factors, the frozen dense
drafter, and the matching head configuration together for inference; applying
SVD again would replace the learned retrieval factors.

### Reading the metrics

- `retrieval_hit_1`: fraction of supervised positions where the candidate set
  contains the frozen full head's top-1 token.
- `retrieval_hit_mean`: mean coverage of the full head's top-`m` positives.
- `retrieval_mass`: full-head probability mass contained in the candidate set.
- `eval/per_position_acc`: full-head top-1 retrieval coverage at each TTT depth;
  `eval/avg_acc` reports position 0, despite its generic name.

For this objective, the generic acceptance-rate fields use positive-set coverage,
and `eval/simulated_acc_len` is derived from retrieval coverage. Neither is an
observed speculative decoding acceptance length. Measure serving acceptance
separately with the target verifier. These definitions also differ from checking
whether a dataset's next token appears among the candidates.

Implementation references: [head and loss](../specforge/modeling/draft/eagle4.py),
[positive-count rule](../specforge/modeling/draft/eagle4_structure.py),
[loading and freezing](../specforge/algorithms/model_providers.py), and
[TTT loss integration](../specforge/algorithms/eagle3/model.py).

For serving exported checkpoints, the version-pinned
[EAGLE-Harness compatibility patch](../patches/eagle-harness/3e20798/README.md)
documents dense and low-rank head support, runtime requirements, and application
and source-hash verification instructions.

## Stage-1 and stage-2 export compatibility

Legacy stage-1 draft configs remain dense, including those with export head
metadata. Both `specforge export --to hf` and `--to sglang` support EAGLE4.
For stage 1, export derives FP32 SVD factors from the trained dense head when
`efficient_lm_head_config` is present, retaining the dense weights. For stage 2,
export preserves the two learned factors and does not recompute SVD. Keep the
stage-2 draft JSON with the stage-2 checkpoint when exporting or warm-starting.
