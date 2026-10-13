# Cost-Aware Scheduling

`training.cost_aware_scheduling` regroups the samples of every optimizer step
so that each data-parallel micro-batch round holds samples of similar cost. It
raises training throughput on corpora with mixed sequence lengths or mixed
supervision density without changing which samples a step trains on.

## Why

One optimizer step is `accumulation_steps` rounds. In each round every DP rank
runs one micro-batch of `batch_size` samples, padded to the longest member, and
FSDP synchronizes the ranks after every micro-batch. A round therefore takes as
long as its most expensive micro-batch. Arrival order (online) or a shuffled
order (offline) puts cheap and expensive samples in the same round, so the
cheap ones compute padding while the slowest rank finishes.

The scheduler keeps the sample *set* of the window (one optimizer step by
default) and only changes which rank runs which sample in which round: samples
are sorted by a cost key, consecutive `dp_size * batch_size` samples form a
round, and each rank takes a contiguous slice of similar keys.

## Configuration

```yaml
training:
  cost_aware_scheduling: true   # default: false
  cost_aware_key: auto          # auto | length | anchors
```

| Key | Cost of a sample | Use with |
| --- | --- | --- |
| `length` | sequence length | EAGLE3, P-EAGLE, MTP |
| `anchors` | `min(num_anchors, valid anchors)`: positions with two consecutive supervised tokens | DFlash, DFlash2, DSpark, Domino |
| `auto` | the key the algorithm declares (`anchors` for the DFlash family, `length` otherwise) | default |

The DFlash family runs `num_anchors * block_size` query tokens per sample, so
its cost follows the anchor budget rather than the context length. Ties are
broken by sample id, so a plan is a pure function of the window contents. A
combined key (anchor tokens plus a weighted context length) measured only
1-2% above either single key and is not offered.

## Online runs

The rank-0 `RefDistributor` already waits until a whole optimizer window is
committed before it releases the first round. With scheduling enabled it plans
that window and publishes the rounds to the per-rank inboxes in planned order.
Inboxes are consume-once queues read in publish order, so the acknowledgement,
backpressure and resume accounting are unchanged and no extra lookahead or
resident memory is needed. The length comes from `SampleRef.num_tokens`; the
capture producer records `metadata["valid_anchors"]` from the prompt's loss
mask. A window whose refs lack the anchor count (an older producer) is
scheduled by length, and a window without any usable cost keeps arrival order;
`stats` of the distributor count `planned_windows`, `length_fallback_windows`
and `fifo_windows`.

## What stays the same

With the default one-step window every step trains on the same samples as
without scheduling, and objectives that hand the trainer additive
`loss_terms` (DFlash, DFlash2, Domino, MTP) are normalized by the global token
count of the step, so every token keeps the weight it would have had: the
objective and the expected gradient are unchanged.

The realized gradient is not bit-identical. The DFlash family samples its
anchors per micro-batch from the global RNG, so a sample placed in another
micro-batch draws a different subset of its valid anchors (the number of
anchors per sample, `min(num_anchors, valid anchors)`, does not change). The
gradient is another unbiased draw of the same expectation, as between two
seeds. Only runs whose samples all have at most `num_anchors` valid anchors
(no subsampling) regroup without changing the draw.

DSpark historically normalizes every DP micro-batch round by its own
all-reduced token weight and then averages the rounds. Under regrouping that
would re-weight tokens by round composition, so with scheduling on the model
hands the trainer `loss_terms` and the whole step is averaged by its global
token weight, like DFlash. The mode is chosen automatically and recorded in
the DSpark resume contract; without scheduling the historical normalization
stays bit for bit.

EAGLE3 and P-EAGLE normalize per padded micro-batch, which keeps the
per-token weights unchanged only at `training.batch_size=1`; larger batch
sizes are rejected with scheduling on.

Checkpoints record the policy; a resume must use the same settings.

## Measured effect

Dense 5-layer DSpark draft (`num_anchors=512`), Qwen3-8B target, 2 capture
servers + DP2, batch 2 x 32, 4x H200, `SHARD_GRAD_OP`, profiler off, mean of
the steady windows (steps 11-30):

| Run | samples/s |
| --- | ---: |
| defaults | 19.1 |
| `cost_aware_scheduling: true` (`auto` = anchors) | 29.2 |
| `cost_aware_key: length` | 26.4 |

Part of the gain is DSpark's per-step normalization (no per-micro-batch
all-reduce; 21.6 on its own); the rest comes from rounds that no longer mix
long and short captures.
