# Legacy draft checkpoint manifests

Inputs and expected results for
`tests/test_scripts/test_convert_legacy_checkpoint.py`. Each checkpoint file
holds the legacy `config.json`, the settings the old recipe supplied, the target
configuration, the legacy state-dict layout and the expected conversion. Only
non-float buffers store values; the test draws the floating-point tensors.
`variants.json` holds configuration changes that keep a checkpoint's tensor
layout (RoPE kinds, attention windows, capture layers), each with the canonical
fields that differ from its base.

`generate.py` wrote these on the last tree whose package still read historical
checkpoints (transfer stack 07, commit `036ae06d`). Layouts are the state dicts
of that tree's legacy model classes (MTP: its `tests/fixtures/legacy_mtp`).
`expected` is what that tree's reader returned for the same checkpoint: the
decoded `config`, the legacy-to-canonical tensor `names`, and the saved target
tables (`bound_copies`) it compared with the target and left out. That tree also
loaded every checkpoint and exported those that need no recipe settings. To
regenerate, copy `generate.py` to this path in that checkout and run it there
with an output directory argument.
