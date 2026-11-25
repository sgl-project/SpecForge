# Qwen3.5-4B MMFlash training

The MMFlash image objective uses DFlash drafts and weights each supervised
position using the image-conditioned target score. SpecForge trains from an
offline cache containing the expanded token IDs, assistant mask, visible-image
target features and quantized visual scores. Both the MMFlash run and the
DFlash control use the same cache, draft geometry and full-block sampler.
[Paper](https://arxiv.org/abs/2611.18482) · [Pinned author code](https://github.com/fscdc/MMFlash/tree/5b3fa3fe28f5bc6d7dfe75934a156c52b865dc8c)

## Single A100 smoke

Use the repository's pinned CUDA-compatible PyTorch and Transformers versions.
Enable image preprocessing by installing the torchvision version that matches
your PyTorch. For the last serving stage, provide a CUDA SGLang environment
that supports DFlash on Qwen3.5-4B. Set `MMFLASH_SERVING_PYTHON` to its Python
when it is installed separately. The converter's serving format is recorded
in the export artifact. [SpecForge draft export](../../docs/sections/basic_usage/training.md#export-a-trained-draft)
[Pinned SGLang setup](../../docs/sections/get_started/installation.md)

```bash
python -m pip install -e .
python -m pip install "torchvision==0.29.2"
bash examples/mmflash/run_a100.sh all
```

The example snapshots the target and ChartQA at fixed revisions. It generates
64 training responses with Qwen3.5-4B and keeps a separate set of 12 held-out
images. Scoring and target feature extraction run on the same card before
trainer startup. The pinned authors' implementation independently replays two
scored rows. The example then trains each arm for 64 steps, exports with
`specforge export --converter sglang`, and serves each export in turn.
`outputs/mmflash-smoke/` holds both arms' logs, exported weights, parity
output and the per-request acceptance and decode timing results.

Each stage can be run separately with `preflight`, `prepare`, `score`,
`parity`, `train`, or `bench`. The `dflash` and `mmflash` stages train and
export the individual arms into fresh run directories.

## Use your own image conversations

Use a scored offline run config with `model.input_modality: multimodal`,
a canonical DFlash draft preset and `training.objective: mmflash`.
The two train arms read the same feature directory. The scorer accepts
one image per line, one user turn, and its target-regenerated assistant
reply with thinking disabled. A leading system turn is supported. Image
paths resolve from the working directory.

```json
{"id":"chart-1","image":"/data/chart-1.png","conversations":[{"role":"user","content":"<image>\nExplain the chart."},{"role":"assistant","content":"A target-regenerated explanation of the chart."}]}
```

```bash
cfg=examples/configs/offline/colocated/qwen3.5-4b-mmflash-a100.yaml
python scripts/score_visual_kl.py --config "$cfg" --input train.jsonl --scores cache/mmflash/scores.jsonl
specforge train --config "$cfg"

# Paired DFlash weighting and the same anchor sampler.
specforge train --config "$cfg" training.objective=dflash \
  training.objective_options.full_block_anchors=true \
  training.objective_options.loss_decay_gamma=7 \
  run_id=dflash output_dir=outputs/mmflash-smoke/dflash
```

The score is the corpus quantile of the forward KL between next-token
distributions, gated by `exp(-H)` using the visible-image entropy. It is
quantized at 1e-4 and aligned to the predicted token. The objective uses
detached prefix products of `s + (1-s)p`, then scales each position by
`1 + alpha * g * (1-p)`. The defaults are `s=0.5` and `alpha=1.0`. The
weighted cross-entropy sum is divided by the batch size. Text-only records
use zero visual boost. Only valid prediction slots enter the prefix product.

Without real data flags, the parity command also has a CPU fixture that
compares scores and the objective with the pinned author checkout.

```bash
python scripts/gates/check_mmflash_parity.py --author-repo /path/to/MMFlash
```
