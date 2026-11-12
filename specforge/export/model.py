"""Public export modes, independent of draft architecture and serving engines."""

from .canonical import read_export


def export_draft(
    checkpoint_path,
    draft_config_path,
    output_dir,
    *,
    export_mode=None,
    converter=None,
    target_model=None,
    copy_target=False,
    vocab_mapping_path=None,
):
    """Publish a standalone draft or merge it into a target checkpoint copy.

    Omitting the mode means standalone. Only the historical Qwen3.5 SGLang
    command retains implicit native export during migration, with a warning.
    """
    if export_mode not in (None, "standalone", "native"):
        raise ValueError(f"unknown export mode: {export_mode}")
    if converter not in (None, "sglang"):
        raise ValueError(f"unknown export converter: {converter}")
    if export_mode == "native" and converter is not None:
        raise ValueError("native export keeps the target format; omit --converter")
    if export_mode == "native" and target_model is None:
        raise ValueError("native export requires --target-model")
    if copy_target and (
        export_mode == "standalone" or (export_mode is None and converter is None)
    ):
        raise ValueError("--copy-target requires native export")
    checkpoint, state, model_class = read_export(
        checkpoint_path,
        draft_config=draft_config_path,
        target=target_model,
        vocab_mapping=vocab_mapping_path,
    )
    mode = export_mode or "standalone"
    legacy_native = False
    if export_mode is None and converter == "sglang":
        from .to_sglang import legacy_native_mode

        legacy_native = legacy_native_mode(checkpoint.config, target_model)
        if legacy_native:
            mode = "native"
    if mode == "native":
        from specforge.checkpoint.formats.native.writer import export_native

        metadata = None
        if legacy_native:
            from specforge.checkpoint.formats.sglang import artifact_metadata

            metadata = artifact_metadata()
        export_native(
            checkpoint.config,
            state,
            target_model,
            output_dir,
            copy_target=copy_target,
            metadata=metadata,
        )
    else:
        if copy_target:
            raise ValueError("--copy-target requires native export")
        if converter is None:
            from .to_hf import write_hf

            write_hf(checkpoint, state, model_class, output_dir)
        else:
            from .to_sglang import write_sglang

            write_sglang(checkpoint, state, output_dir)
    return output_dir
