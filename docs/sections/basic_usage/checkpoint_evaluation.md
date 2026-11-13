# Evaluate a checkpoint on cached teacher features

`specforge evaluate-checkpoint` compares DFlash and DFlash2 checkpoints on the
same examples and sampled anchor positions. It loads the draft weights and the
target embeddings/head, then replays cached target features in one process.
It does not create an optimizer or start a target server.

These are **teacher-forced diagnostics**: they measure prediction of the saved
continuation. Use `specforge benchmark` against a running SGLang server to measure
generation throughput and native speculative acceptance.

## Run

Use an existing SpecForge YAML/JSON recipe to select the target, draft architecture,
attention backend, objective, anchor count, and block size:

```bash
SPECFORGE_FLEX_ATTENTION_BACKEND=FLASH specforge evaluate-checkpoint \
  --config evaluation.yaml \
  --checkpoint /checkpoints/draft-step1000 \
  --feature-index /features/development/index.json \
  --output /results/development-step1000.json
```

The checkpoint can be a SpecForge checkpoint directory/`file://` URI or a
Hugging Face draft model. Training resume state is ignored: this command loads
only the selected draft weights. Existing output files are refused. `--limit N`
evaluates the first N manifest entries for a smoke check. Run directly in one
process, outside `torchrun`; both CPU fixtures and CUDA execution are supported.
The model must fit on one device.

Configuration overrides use the same `section.field=value` syntax as training.
For comparable CE diagnostics across training objectives, use the same evaluation
recipe for every checkpoint, for example:

```bash
specforge evaluate-checkpoint \
  --config run.yaml --checkpoint /checkpoints/draft-step1000 \
  --feature-index /features/development/index.json --output /results/ce1000.json \
  training.loss_type=dflash training.lk_loss_type=null \
  training.dflash2_selector_loss_alpha=1.0
```

The selector coefficient is the configured constant; training warmup/ramp
schedules are not advanced. Keep backend, objective, block size, anchor count,
loss decay, target revision, and feature cache fixed across comparisons. An
architecture supported only by an unmerged model PR still requires that PR.

## Feature index

Create an index for a precomputed feature cache:

```json
{
  "rows": [
    {"id": "example-0", "path": "000.pt", "seed": 42},
    {"id": "example-1", "path": "001.pt", "seed": 43}
  ]
}
```

Each `.pt` file is a tensor dictionary with `input_ids` and `loss_mask` shaped
`[1, sequence_length]`, and `hidden_states` shaped `[1, sequence_length, width]`.
It can also contain `target_last_hidden_states` with the same batch/sequence
dimensions for teacher-distribution diagnostics. Preserve the capture layer
ordering, token alignment, target revision and supervision mask used by training.
No tokenization, truncation or recapture happens during evaluation.

Paths are absolute or relative to the index. IDs must be unique. Optional `sha256`
entries reject changed feature files. Each report records actual feature-file
digests and the index digest. Without `seed`, the first eight hex digits of
`group_hash` supply the seed; without either, SHA256 of the ID supplies it.
This makes anchor sampling independent of example order. RNG and model training
mode are restored by the reusable `evaluate_cached_features` function.

Use a held-out, deduplicated cohort. If an existing cohort overlaps training,
report overlap strata explicitly; freezing the cache does not make it held out.

## Read and log results

The JSON report includes the resolved recipe, checkpoint path, draft config,
per-example scalar metrics, additive ratio numerators/denominators, and counts.
`mean.loss` is the mean of per-example composite losses. Ratios such as `ce_loss`
are computed as `sum(numerator) / sum(denominator)`, not an average of ratios.
`ratio_totals` and `sum_totals` preserve the populations behind each diagnostic.
An absent/zero exposure has a zero raw ratio and must not be interpreted as a
measured acceptance rate.

An external checkpoint hook can log the completed report at the saved optimizer
step using its existing tracker:

```python
logger({f"eval/{name}": value for name, value in report["mean"].items()}, step)
```

Keep tracker lifecycle and distributed synchronization in the owning training
controller. The standalone command performs no tracker writes or optimization.
