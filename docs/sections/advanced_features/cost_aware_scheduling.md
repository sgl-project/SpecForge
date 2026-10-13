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
  cost_aware_window: 0         # offline only; 0 = one optimizer step
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

## Offline runs

The sampler keeps its seeded per-epoch permutation and DP padding, then
regroups each window of `cost_aware_window` micro-batch rounds (default: one
optimizer step) before striding the order over the DP ranks. Windows follow the
global micro-batch count: gradient accumulation spans epochs, so the first
window of an epoch is shortened to end on the trainer's step boundary, and
the trailing partial micro-batch each rank drops (`drop_last`) keeps its
original position so the same samples are dropped as without scheduling. Costs of
`file://` features are read once from `input_ids` and `loss_mask` and cached
in `<output_dir>/cost_aware_index.json`, keyed by the file list with sizes
and mtimes and by `max_length` (delete the file to force a rescan);
disaggregated offline refs carry `num_tokens` and `valid_anchors` from the
producer and need no scan. A window wider than one optimizer step gains
little (about one more percent in the benchmark below) and changes which
samples share a step, so the default keeps the step membership intact.

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

Checkpoints record the policy and, offline, the window, `max_length` and a
digest of the per-sample costs, the inputs that decide the order; a resume
whose costs would differ (another `max_length`, rewritten loss masks,
different ref metadata, key or `num_anchors`) is rejected.

## Measured effect

Dense 5-layer DSpark drafts (`num_anchors=512`), 4x H200, `SHARD_GRAD_OP`,
profiler off, mean of the steady windows (steps 11-30); scheduling rows use
`auto`, under which DSpark normalizes per optimizer step.

| Run | samples/s |
| --- | ---: |
| offline, Qwen3.5-27B target, DP4, batch 2 x 32, defaults (`round`) | 17.0 |
| offline, DSpark per-step normalization alone (no regrouping) | 17.5 |
| offline, `cost_aware_scheduling: true` (`auto` = anchors) | 19.2 |
| offline, `cost_aware_key: length` | 19.3 |
| online, Qwen3-8B target, 2 capture servers + DP2, batch 2 x 32, defaults (`round`) | 19.1 |
| online, DSpark per-step normalization alone (no regrouping) | 21.6 |
| online, `cost_aware_scheduling: true` (`auto` = anchors) | 29.2 |
| online, `cost_aware_key: length` | 26.4 |

`step` alone is faster because the per-micro-batch all-reduce of the DSpark
denominator disappears. The scheduling gain on top of it is the cross-rank
imbalance of anchor slots per round (1.12x the useful work before, 1.01x
after) plus, online, the arrival-order mix of long and short captures.
Within-micro-batch sequence packing does not address it and measured below
the baseline on this corpus.
