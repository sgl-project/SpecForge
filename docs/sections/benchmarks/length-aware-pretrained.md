# Pretrained DFlash2 length-scheduling validation

This experiment uses the matching pretrained `Qwen/Qwen3.8-27B` target and
`incoai/Qwen3.8-27B-DFlash2` draft with real ShareGPT assistant responses. It is
a short fine-tuning and trainer-throughput experiment, not a convergence or
serving-acceptance benchmark. The data deliberately cover heterogeneous lengths;
they are not a representative sample of the full ShareGPT length distribution.

## Models, data, and alignment

- Target revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Draft revision: `dedf8df68adfb1afeaf7b7480c0a0243108177b4`.
- Dataset: `anon8231489123/ShareGPT_Vicuna_unfiltered`, revision
  `192ab2185289094fc556ec8ce5ce1e8e587154ca`,
  `ShareGPT_V3_unfiltered_cleaned_split.json`.
- Six distinct conversations are selected for each truncated length
  128/256/512/1024: four for training and two for held-out evaluation. The
  resulting 16 training and 8 held-out source hashes are disjoint.
- The target tokenizer's chat template has thinking disabled. The tokenized
  user/generation prefix is checked against the complete conversation; only
  the assistant suffix is supervised, with at least 48 supervised tokens after
  truncation. Raw conversation text is not included in results.
- Actual target features come from BF16 Transformers SDPA execution, not random
  tensors or generated substitutes. Decoder outputs at target layers
  `[5, 19, 33, 47, 61]` correspond to Hugging Face hidden-state indices
  `[6, 20, 34, 48, 62]`; repository `extract_context_feature` concatenates them.
  None is the final target layer, whose output would include final normalization.
  Training uses the repository's offline normalizer and padded collator.
- The draft checkpoint is loaded without missing, unexpected, or mismatched
  keys. Frozen input embeddings and LM-head weights come from the target's
  actual safetensors checkpoint.

The target load took 56.52 s; the 24 captures took 8.65 s in total. These costs
are excluded from draft-training throughput. Capture used the installed
Transformers Torch fallback for hybrid linear attention; no kernel package was
installed or changed for this test.

## Training and measurement protocol

The draft retains its checkpoint configuration: hidden size 5120, vocabulary
248320, five layers, block size 8, convolution group size 16/kernel size 2,
selector rank 256, and selector top-k 16. Training follows the actual recipe's
BF16, flex attention, 512 native random anchors, objective chunk size 128,
`loss_type=dflash`, loss-decay gamma 7, selector coefficient 1, and disabled
teacher diagnostics. The production `TrainerCore`, DFlash strategy, FSDP
`SHARD_GRAD_OP`, and `BF16Optimizer` execute on two H200 GPUs.

Each rank uses batch size 2 and accumulation 4, so each optimizer window contains
the same 16 conversations and 7,680 valid context tokens. Offline bucketing is
bounded to this one optimizer window. The declared smoke-test variations are
short contexts (at most 1,024 tokens), learning rate `1e-5` rather than the
recipe's `5e-4`, no learning-rate warmup, and only eight optimizer updates.
Gradient clipping remains 1.0. Native anchors may differ after regrouping;
the measured supervised-position counts are retained and used in throughput.

Every variant runs in a **fresh two-rank torchrun process**, reloading the same
pretrained weights and resetting the optimizer and seed. Each run performs
three warmup updates and five measured updates. There are three repeats in
baseline/online/offline, offline/online/baseline, baseline/online/offline order.
GPU 2/3 process inventories are checked to be empty before every run; no other
test jobs share those devices. Step time is the maximum wall time across ranks,
including forward, backward, FSDP communication, and AdamW update. Features are
preloaded on CPU. This does not include online producer/transport or offline
disk-reader performance.

Peak allocated memory is reset after warmup and reduced with MAX across ranks
over measured training updates, before held-out evaluation. The earlier
[shared-process experiment](../../../benchmarks/results/length-aware-h200/pretrained-dflash2-shared-process-superseded.json)
is retained only as superseded evidence: allocated memory increased across
model resets, so it is not used for the final throughput or memory comparison.

## Recorded results (2026-10-02)

All nine isolated runs completed, with finite training losses, gradient norms,
and held-out metrics. Production sources were from
`e8e51f2cf26e56174c375dae8e64af904dd09e6d`, using PyTorch 2.13.0+cu130 and CUDA
13.0. The [summary and all step measurements](../../../benchmarks/results/length-aware-h200/pretrained-dflash2.json)
and [unmodified per-process reports](../../../benchmarks/results/length-aware-h200/pretrained-dflash2-isolated-raw.json)
retain source hashes, model configuration, sample hashes, counts, and timing.

| Schedule | Median update time, repeats 1 / 2 / 3 (ms) | Paired median-time speedup | Supervised positions/s, all measured updates | Maximum allocated memory (GB) |
| --- | --- | --- | ---: | ---: |
| Baseline | 799.45 / 800.55 / 801.21 | 1.000 / 1.000 / 1.000 | 31,468 | 43.837 |
| Online scheduling | 658.35 / 655.89 / 657.94 | 1.214 / 1.221 / 1.218 | 36,817 | 42.066 |
| Offline bucketing | 635.74 / 636.65 / 636.75 | 1.258 / 1.257 / 1.258 | 38,996 | 43.789 |

Actual supervised-position throughput increased **17.0% online and 23.9%
offline** over the baseline in this bounded workload. This calculation sums
actual counts and durations across all 15 measured updates per variant;
occasional slower updates, up to 1.075 s, are retained. Median-time ratios are
reported separately and must not replace that all-sample throughput result.
Each update supervised 26,131–26,137 positions, so the gain did not come from
a material reduction in supervised work. No confidence interval or guarantee
for another dataset, context length, or deployment is claimed.

Context padding decreased from 23.08% to zero. Metadata also predicts 3,781
valid anchor blocks per window in every variant, while allocated blocks fall
from 5,038 to 4,328 online and 4,498 offline. These are model-shape counts, not
a profiler attribution of the speedup. Memory values are the maximum over
both ranks after warmup and are identical across the three repeats. Offline
peak memory is nearly unchanged.

All variants initially had combined held-out objective 4.454173 and unary
hard-label accuracy 34.3750%. After eight updates:

| Schedule | Combined held-out objective | Unary hard-label accuracy | Correct labels / 448 |
| --- | ---: | ---: | ---: |
| Baseline | 4.286357 | 35.0446% | 157 |
| Online scheduling | 4.280660 | 34.8214% | 156 |
| Offline bucketing | 4.267283 | 35.0446% | 157 |

The result is identical across repeats for each schedule. Online accuracy is
**0.2232 percentage points lower** than baseline, corresponding to one label.
This observation is retained; this experiment does **not** pass an accuracy
acceptance gate or establish quality equivalence.

The final summary explicitly records metadata-only postprocessing: the original
`hard_label_accuracy` field is labeled `unary_hard_label_accuracy`, and a legacy
hardcoded multi-variant order is replaced with the actual isolated-process
wrapper order. Raw reports remain unchanged. The utility received those same
report-label corrections after execution; both utility hashes are recorded.
No model, data, timing, or metric value changed during this postprocessing.

## Held-out diagnostic boundary

Evaluation uses eight fixed full-block anchors per held-out conversation,
giving **448 label positions**. Every variant starts with exactly the same
held-out training objective and unary hard-label accuracy. The objective combines
the decayed DFlash token loss with the selector loss at coefficient 1; the
accuracy counts unary-logit argmax predictions before candidate-selector
decisions. Neither is a serving metric. This is deliberately a smaller fixed
evaluation budget than the 512 native random training anchors. One different prediction
changes reported accuracy by 0.2232 percentage points. Eight training updates
on 16 conversations and this small evaluation set cannot establish an accuracy
acceptance gate, convergence equivalence, or serving acceptance length.

The separate synthetic BF16 update-parity experiment uses the D-PACE objective.
Its failed parameter-update gate must remain reported; these real-checkpoint
DFlash-loss measurements do not supersede that result or establish a universal
BF16 equivalence guarantee.

## Reproduce

Use the exact cached revisions above; no download is performed by the utility.
The paths below are placeholders for local snapshots and the dataset JSON:

```bash
set -e

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 python -m benchmarks.pretrained_length_smoke capture \
  --work-dir /tmp/pretrained-length-validation \
  --target-path /models/qwen-target-snapshot \
  --draft-path /models/dflash2-snapshot \
  --dataset-path /datasets/ShareGPT_V3_unfiltered_cleaned_split.json

for repeat in 0 1 2; do
  if [ "$repeat" = 1 ]; then
    schedules="offline online baseline"
  else
    schedules="baseline online offline"
  fi
  for schedule in $schedules; do
    CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=2 \
    torchrun --standalone --nproc-per-node=2 -m benchmarks.pretrained_length_smoke train \
      --work-dir /tmp/pretrained-length-validation \
      --warmup 3 --steps 8 --repeats 1 --repeat-start "$repeat" \
      --schedules "$schedule" --report-name "$repeat-$schedule.json"
  done
done
```

The capture manifest supplies the target and draft paths to training. Optional
explicit training paths must match that manifest. A tar checkout can pass
`--source-revision` to record its source commit; the report additionally hashes
the utility and relevant production files. Feature/token tensors remain in the
local work directory; reports contain aggregate metrics and source hashes.
