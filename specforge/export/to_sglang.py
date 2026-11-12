"""Export canonical or imported legacy weights to a pinned serving format."""

import argparse

from specforge.checkpoint.draft_dir import write_artifact
from specforge.checkpoint.formats.sglang import artifact_metadata, convert_config
from specforge.checkpoint.formats.sglang.converters import serving_weights
from specforge.checkpoint.native import is_native_architecture

from .canonical import read_export


def export_to_sglang(
    checkpoint_path,
    draft_config_path,
    output_dir,
    *,
    target_model=None,
    copy_target=False,
    vocab_mapping_path=None,
    weight_map=None,
):
    if weight_map is not None:
        raise ValueError(
            "arbitrary weight_map overrides are not supported by the serving format"
        )
    checkpoint, state, _ = read_export(
        checkpoint_path,
        draft_config=draft_config_path,
        target=target_model,
        vocab_mapping=vocab_mapping_path,
    )
    if is_native_architecture(checkpoint.config.architecture):
        from specforge.checkpoint.formats.sglang.native import export_native

        export_native(
            checkpoint.config,
            state,
            target_model,
            output_dir,
            copy_target=copy_target,
        )
    else:
        fields = convert_config(checkpoint.config, hf_metadata=checkpoint.hf_metadata)
        weights = serving_weights(checkpoint.config, state)
        if copy_target:
            raise ValueError("copy_target is only supported for native export")
        dtypes = {
            str(t.dtype).removeprefix("torch.")
            for t in weights.values()
            if t.is_floating_point()
        }
        if len(dtypes) != 1:
            raise ValueError("SGLang export requires one floating-point dtype")
        fields["torch_dtype"] = dtypes.pop()
        write_artifact(output_dir, fields, weights, metadata=artifact_metadata())
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--draft-config")
    parser.add_argument("--target-model")
    parser.add_argument("--copy-target", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    export_to_sglang(
        args.checkpoint,
        args.draft_config,
        args.output_dir,
        target_model=args.target_model,
        copy_target=args.copy_target,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
