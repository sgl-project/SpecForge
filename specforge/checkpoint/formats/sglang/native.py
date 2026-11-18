"""Compatibility entry point for historical SGLang native exports."""

from specforge.checkpoint.formats.native.writer import export_native as _export_native

from . import artifact_metadata


def export_native(config, state, target, output, *, copy_target=False):
    return _export_native(
        config,
        state,
        target,
        output,
        copy_target=copy_target,
        metadata=artifact_metadata(),
    )
