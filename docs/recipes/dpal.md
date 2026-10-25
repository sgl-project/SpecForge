# D-PAL: acceptance-derived weights and distributional losses

[D-PAL](https://arxiv.org/abs/2610.28355) trains selector-free DFlash1 drafts
with detached target-draft overlap weights. It uses the frozen target
distribution `p` and the draft distribution `q` at the recorded training prefix.
It does not change the draft architecture, export format, or decoding algorithm.

| `training.loss_type` | Paper instance | Local loss |
| --- | --- | --- |
| `dpal-renyi` | D-PAL[Rényi] | `-2 log sum_v sqrt(p_v q_v)` |
| `dpal-kl` | D-PAL[KL] | `KL(p || q)`, target to draft |
| `dpal-la` | D-PAL[LA] | `-log sum_v min(p_v, q_v)` |
| `dpal-ce-tv` | D-PAL[CE+TV] | `0.1 CE(y*, q) + 0.9 TV(p, q)` |

Here `TV(p,q) = 0.5 sum_v |p_v-q_v|`. The CE+TV coefficient is on TV,
not on the unhalved L1 distance. `y*` is the recorded target token.
Each loss uses the same weights:

```text
acceptance_t = sum_v min(p_t[v], q_t[v])
s_t = rho + (1 - rho) * acceptance_t
w_t = stop_gradient(sum_{k=t}^D product_{i=1}^k s_i)
```

`training.dpal_rho` defaults to `0.3` and lies in `[0, 1]`. It does not use
`training.dpace_alpha`. Masked positions contribute no loss and act as no-ops
in prefix products. The objective sums weighted positions, averages valid
anchors within each sequence, then averages valid sequences across the optimizer
window and data-parallel ranks. It does not divide by weight mass.

## Offline example

Start with a target-generated training JSONL in SpecForge's conversation format.
Capture the final teacher states as well as the draft's auxiliary features:

```bash
torchrun --nproc_per_node=1 scripts/prepare_hidden_states.py \
  --strategy dflash --loss-type dpal-renyi \
  --target-model-path Qwen/Qwen3-4B \
  --draft-model-config configs/qwen3-4b-dpal-b16.json \
  --data-path ./cache/dataset/train.jsonl \
  --output-path ./cache/hidden_states/qwen3-4b-dpal \
  --chat-template qwen --max-length 3072 --tp-size 1 --batch-size 1

specforge train -c examples/configs/offline/colocated/qwen3-4b-dpal.yaml --plan
specforge train -c examples/configs/offline/colocated/qwen3-4b-dpal.yaml
```

All four losses share that capture. Select another instance with a dotted override,
using a separate output directory:

```bash
specforge train -c examples/configs/offline/colocated/qwen3-4b-dpal.yaml \
  training.loss_type=dpal-ce-tv run_id=dpal-ce-tv output_dir=./outputs/dpal-ce-tv
```

Old three-field DFlash caches cannot supply a distributional target; recapture
them into a new directory. D-PAL requires `target_last_hidden_states` on every
training step, even when `training.dflash_teacher_metrics=false` or detailed
logging is off. Online capture retains the same required field. Checkpoints
record both the selected objective and `dpal_rho` for resume validation.

The example uses three draft layers, features `[1,17,33]`, B16, six epochs and
batch size eight. Adjust the device topology and microbatch size to the available
memory; effective batch size is microbatch × accumulation × data-parallel ranks.
The frozen target projection runs once per forward and chunks gather from its
detached `[batch, sequence, vocabulary]` logits. This saves repeated head work
but retains that cache in memory. It is not a cross-batch or cross-epoch cache.
The current D-PAL path excludes DFlash2 selectors, Domino, DSpark and LK mixing;
their existing training defaults remain unchanged.

## Citation

```bibtex
@article{wu2026dpal,
  title={Beyond the Point Mass: Acceptance-Derived Weights and Distributional Losses for Speculative Decoding},
  author={Wu, Tianyu and Wang, Zhuohan and Geuter, Jonathan and Yao, Yu and Jiang, Weifan and Zhang, Chengxi and Li, Ju and Yu, Minlan},
  journal={arXiv preprint arXiv:2610.28355},
  year={2026},
  url={https://arxiv.org/abs/2610.28355}
}
```
