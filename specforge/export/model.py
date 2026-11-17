"""Public export modes, independent of draft architecture and serving engines."""

from .canonical import read_export


def check_export_options(
    *, export_mode=None, converter=None, target_model=None, copy_target=False
):
    """Validate mode/flag combinations without reading checkpoint files."""
    if export_mode not in (None, "standalone", "native"):
        raise ValueError(f"unknown export mode: {export_mode}")
    if converter not in (None, "sglang"):
        raise ValueError(f"unknown export converter: {converter}")
    if export_mode == "native" and converter is not None:
        raise ValueError("native export keeps the target format; omit --converter")
    if export_mode == "native" and target_model is None:
        raise ValueError("native export requires --target-model")
    if copy_target and export_mode != "native":
        raise ValueError("--copy-target requires native export")


def export_draft(
    checkpoint_path,
    output_dir,
    *,
    export_mode=None,
    converter=None,
    target_model=None,
    copy_target=False,
):
    """Publish a standalone draft or merge it into a target checkpoint copy.

    Omitting the mode means standalone; native merge-back must be explicit.
    """
    check_export_options(
        export_mode=export_mode,
        converter=converter,
        target_model=target_model,
        copy_target=copy_target,
    )
    checkpoint, state, model_class = read_export(checkpoint_path, target=target_model)
    mode = export_mode or "standalone"
    if mode == "native":
        from specforge.checkpoint.formats.native.writer import export_native

        export_native(
            checkpoint.config,
            state,
            target_model,
            output_dir,
            copy_target=copy_target,
        )
    else:
        if converter is None:
            from .to_hf import write_hf

            write_hf(checkpoint, state, model_class, output_dir)
        else:
            from .to_sglang import write_sglang

            write_sglang(checkpoint, state, output_dir)
    return output_dir
