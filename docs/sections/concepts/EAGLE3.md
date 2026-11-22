# EAGLE3

## Overview

In [speculative decoding](./speculative_decoding.md), a small draft proposes
tokens and a larger target verifies them. A separate small language model can
serve as the draft, but finding one that predicts the target well can be
difficult. EAGLE3 instead trains a lightweight draft that reads the target's
hidden states and the tokens generated so far.

The [EAGLE3 paper](https://arxiv.org/abs/2503.01840) introduces multi-layer
feature fusion and training-time test. These let the draft predict tokens
directly while learning to continue from its own intermediate states. The
target stays frozen during draft training and remains responsible for
verification during inference.

## How it works?

<p align="center">
  <img src="https://developer-blogs.nvidia.com/wp-content/uploads/2025/09/speculative-decoding-eagle-drafting-mechanism.gif" alt="EAGLE3"><br>
  <span>Source: <a href="https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/">Blog by NVIDIA</a></span>
</p>

The animation shows how a compact draft extends the target's output. One
draft-and-verify cycle has four steps:

1. **Collect target features.** The target processes the prompt or verifies a
   previous draft. Hidden states from selected shallow, middle, and deep layers
   are concatenated and projected into the draft's feature space.
2. **Predict a candidate.** The draft combines these features with token
   embeddings, runs its decoder, and produces next-token logits.
3. **Continue cheaply.** Target features for unverified candidates are not yet
   available. The draft feeds back its own hidden states and candidate tokens
   to extend the proposal without another target forward.
4. **Verify and refresh.** The target checks the proposal. Accepted tokens are
   committed, rejected continuations are discarded, and the next cycle starts
   from the verified context.

The feature fusion can be written schematically as

$$
g_t = W_f [h_t^{\mathrm{low}}; h_t^{\mathrm{mid}}; h_t^{\mathrm{high}}].
$$

Here, the three hidden states describe the same target position. Their layer
selection and alignment are part of the draft configuration; they must match
the captured training features and the serving runtime.

### Why fuse several target layers?

The original EAGLE uses top-layer target features and trains the draft to
approximate those features. EAGLE3 removes that feature-reconstruction
constraint and supervises token prediction. This permits a richer mixture of
target layers rather than requiring the draft state to reproduce a particular
target representation. See the [paper's architecture discussion](https://arxiv.org/html/2503.01840v1#S3)
for the distinction between EAGLE and EAGLE3.

## Training

Training-time test unrolls several draft steps. The first step reads fused
target features; later steps consume draft-produced hidden states. Token
supervision is applied at multiple rollout depths so the draft learns to
operate after it has left the initial target-feature input.

SpecForge's EAGLE3 objective uses captured target features and teacher outputs
to build this recurrent computation. The canonical default is soft-label
cross-entropy, with a configurable decay across rollout depth.
`ttt_length` controls the number of training rollout steps. This is a training
setting; it is distinct from the inference engine's candidate budget and tree
shape. The attention mask keeps each rollout attached to its own anchor and
the valid prefix, rather than allowing unrelated training rollouts to read
each other.

Training-time test concerns recurrent feature use. It does not mean the
trainer runs the serving engine's full tree search or rejection sampler on
every training batch. In particular, SpecForge's canonical objective gathers
the depth-aligned sequence tokens while feeding back draft hidden states.
The exact computation is visible in the
[EAGLE3 objective](https://github.com/sgl-project/SpecForge/blob/main/specforge/algorithms/objectives/eagle3.py).

## Inference

EAGLE3 can use the context-dependent draft tree introduced by
[EAGLE2](https://arxiv.org/abs/2406.16858). Branches offer alternatives at a
position, while depth offers longer continuations. A serving runtime chooses
which nodes to expand within its draft budget and verifies the resulting tree
with the appropriate attention mask.

Tree expansion is a serving policy, not an additional target training step.
The target's verification and sampling rules determine which path can be
committed. With exact speculative sampling, the target distribution is
preserved; simply selecting a likely draft token is not sufficient to obtain
that guarantee. See the
[verification explanation](./speculative_decoding.md#how-it-works) for the
accept/reject procedure.

Longer or wider trees offer more candidates but also increase draft work,
verification work, and cache use. Evaluate both acceptance and end-to-end
latency or throughput for the intended workload instead of choosing a tree
solely by its number of candidates.

## Using EAGLE3 in SpecForge

Start with the [online external recipe](https://github.com/sgl-project/SpecForge/blob/main/examples/configs/online/disaggregated/external/qwen3-8b-eagle3.yaml)
or the [offline colocated recipe](https://github.com/sgl-project/SpecForge/blob/main/examples/configs/offline/colocated/qwen3-8b-eagle3.yaml).
Both use the shared training entry point:

```bash
specforge train --config examples/configs/online/disaggregated/external/qwen3-8b-eagle3.yaml
```

Online training obtains frozen target features through the capture service;
offline training reads previously captured features. Use the same target
checkpoint, tokenizer, feature layers, and vocabulary mapping throughout
capture, training, and serving. A vocabulary mapping changes which draft
logits correspond to target token IDs, so the serving export must preserve it.

The [training guide](../basic_usage/training.md) describes configuration,
parallelism, and metrics. The [export guide](/draft_export.md) describes the
serving formats and their restrictions. Published drafts can be found in
[SpecBundle](/specbundle.md).
