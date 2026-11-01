#!/usr/bin/env python3
# coding=utf-8
"""Merge a trained MTP draft checkpoint back into the base target model.

Thin CLI wrapper around ``specforge.export.mtp.merge_mtp_into_base``; the merge
logic and the per-family key mapping live in the package.

Example:
    python scripts/merge_mtp_to_base.py \
        --base-model-path PATH/TO/Qwen3.5-4B \
        --mtp-checkpoint-path PATH/TO/outputs/qwen3.5-4b-mtp/RUN-latest \
        --draft-config configs/qwen3.5-4b-mtp.json \
        --output-path PATH/TO/Qwen3.5-4B-MTP \
        --key-format sglang
"""

import argparse

from specforge.export.mtp import merge_mtp_into_base


def main():
    parser = argparse.ArgumentParser(
        description="Merge trained MTP weights back into the base Qwen3.5 model."
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        required=True,
        help="Path to the original Qwen3.5 base model checkpoint.",
    )
    parser.add_argument(
        "--mtp-checkpoint-path",
        type=str,
        required=True,
        help=(
            "SpecForge runtime checkpoint/output path, or an already-exported "
            "HF MTP draft directory."
        ),
    )
    parser.add_argument(
        "--draft-config",
        type=str,
        default=None,
        help=(
            "Draft config JSON (required for a SpecForge runtime checkpoint; "
            "the exported HF directory already contains config.json)."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Directory to write the merged checkpoint.",
    )
    parser.add_argument(
        "--key-format",
        type=str,
        default="sglang",
        choices=["sglang", "hf"],
        help=(
            "MTP key layout. Both 'sglang' and 'hf' produce the flat native "
            "layout (mtp.layers.0.* / mtp.norm.weight) that SGLang's flat "
            "Qwen3_5ForCausalLMMTP and HF/vLLM MTP modules expect; the argument "
            "is kept for backward compatibility."
        ),
    )
    args = parser.parse_args()

    merge_mtp_into_base(
        args.base_model_path,
        args.mtp_checkpoint_path,
        args.output_path,
        args.key_format,
        draft_config_path=args.draft_config,
    )


if __name__ == "__main__":
    main()
