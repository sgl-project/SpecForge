# coding=utf-8
"""Convert DeepSeek-V4's DSpark draft between the official ``mtp.*`` checkpoint and SpecForge.

Import the official draft (bf16, routed experts fused) into a SpecForge draft directory::

    python scripts/convert_deepseek_v4_dspark.py \
        --source deepseek-ai/DeepSeek-V4-Flash-DSpark --output outputs/deepseek-v4-flash-dspark-draft

Export a trained SpecForge draft back to the official layout for SGLang serving::

    python scripts/convert_deepseek_v4_dspark.py --to-official \
        --source outputs/my-draft --output outputs/my-draft-official \
        --official-config /path/to/DeepSeek-V4-Flash-DSpark/config.json
"""

import argparse

from specforge.export.deepseek_v4_dspark import (
    convert_official_to_specforge,
    convert_specforge_to_official,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--source",
        default="deepseek-ai/DeepSeek-V4-Flash-DSpark",
        help="Official checkpoint (local directory or Hub repo), or the SpecForge draft with --to-official.",
    )
    parser.add_argument("--output", required=True, help="Directory to write.")
    parser.add_argument(
        "--draft-config",
        default="configs/deepseek-v4-flash-moe-dspark.json",
        help="SpecForge draft config written next to the imported weights.",
    )
    parser.add_argument(
        "--to-official",
        action="store_true",
        help="Export a SpecForge draft to the official mtp.* layout instead of importing.",
    )
    parser.add_argument(
        "--official-config",
        help="Official config.json to derive the exported draft config from (--to-official).",
    )
    parser.add_argument(
        "--cache-dir", help="Hub download cache for the official shards."
    )
    parser.add_argument("--max-shard-size", default="5GB")
    args = parser.parse_args()

    if args.to_official:
        if not args.official_config:
            parser.error("--to-official requires --official-config")
        convert_specforge_to_official(
            args.source,
            args.output,
            args.official_config,
            max_shard_size=args.max_shard_size,
        )
    else:
        convert_official_to_specforge(
            args.source,
            args.output,
            args.draft_config,
            cache_dir=args.cache_dir,
            max_shard_size=args.max_shard_size,
        )


if __name__ == "__main__":
    main()
