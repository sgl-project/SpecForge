# SpecForge TPU plugin

The TPU image installs this plugin alongside SpecForge. To install it from a
SpecForge checkout in an existing `torch_tpu` environment, run:

```bash
pip install -e . -e ./plugins/specforge-tpu
```

Set `SPECFORGE_DEVICE=tpu` to activate the plugin. The TPU image sets this
variable for you.

By default, each batch is padded to its longest sequence. To reduce the number
of shapes compiled during training, optionally set sequence length buckets:

```bash
export SPECFORGE_TPU_SEQ_BUCKETS=512,1024,2048
```

A batch with a longest sequence of 700 tokens then pads to 1024. A sequence
longer than the largest bucket raises an error.

See the [single-host DSpark recipe](../../examples/configs/offline/colocated/gpt-oss-120b-dspark-tpu.yaml).
