"""Export canonical weights to a pinned serving format."""

import argparse

from specforge.checkpoint.draft_dir import write_artifact
from specforge.checkpoint.formats.sglang import artifact_metadata, convert_config
from specforge.checkpoint.formats.sglang.converters import serving_weights

from .model import export_draft


def export_to_sglang(
    checkpoint_path,
    output_dir,
    *,
    target_model=None,
    copy_target=False,
    vocab_mapping_path=None,
    weight_map=None,
    export_mode=None,
):
    if weight_map is not None:
        raise ValueError(
            "arbitrary weight_map overrides are not supported by the serving format"
        )
    return export_draft(
        checkpoint_path,
        output_dir,
        converter="sglang",
        export_mode=export_mode,
        target_model=target_model,
        copy_target=copy_target,
        vocab_mapping_path=vocab_mapping_path,
    )


def write_sglang(checkpoint, state, output_dir):
    fields = convert_config(checkpoint.config, hf_metadata=checkpoint.hf_metadata)
    weights = serving_weights(checkpoint.config, state)
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
    parser.add_argument("--target-model")
    parser.add_argument("--copy-target", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    export_to_sglang(
        args.checkpoint,
        args.output_dir,
        target_model=args.target_model,
        copy_target=args.copy_target,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
