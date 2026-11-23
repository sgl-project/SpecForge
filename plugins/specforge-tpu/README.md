# SpecForge TPU plugin

The TPU image installs this plugin alongside SpecForge. To install it from a
SpecForge checkout in an existing `torch_tpu` environment, run:

```bash
pip install -e . -e ./plugins/specforge-tpu
```

The SpecForge CLI discovers the `tpu` entry point at startup. Set
`SPECFORGE_DEVICE=tpu` to select it explicitly. When unset, the plugin selects
TPU if `torch_tpu` is installed.

See the [single-host DSpark recipe](../../examples/configs/offline/colocated/gpt-oss-120b-dspark-tpu.yaml).
