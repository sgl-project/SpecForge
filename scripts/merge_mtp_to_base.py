#!/usr/bin/env python3
"""Compatibility CLI for checked native MTP export.

New callers can use ``specforge export --export-mode native --target-model TARGET``.
"""

import argparse

from specforge.export.model import export_draft


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--mtp-checkpoint-path", required=True)
    parser.add_argument(
        "--draft-config",
        help="Supplementary JSON for config-less historical checkpoints",
    )
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--key-format",
        default="sglang",
        choices=["sglang", "hf"],
        help=(
            "Both historical spellings select native merge-back. "
            "Use specforge export --export-mode standalone for Transformers artifacts."
        ),
    )
    parser.add_argument(
        "--copy-target", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args(argv)
    export_draft(
        args.mtp_checkpoint_path,
        args.draft_config,
        args.output_path,
        target_model=args.base_model_path,
        copy_target=args.copy_target,
        export_mode="native",
    )


if __name__ == "__main__":
    main()
