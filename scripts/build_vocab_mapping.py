"""Build a draft vocabulary mapping from prepared offline features.

Training can derive this map on its own, but only for a colocated offline run,
and only by reading every feature file first -- which for gzipped features means
decompressing the whole dataset before the first step. This script does that
pass once, caches the token counts, and then answers any number of
``draft_vocab_size`` questions from the cache in milliseconds.

That separation is the point: counting is the expensive half and depends only on
the dataset, while choosing the top-K is cheap and is the half you actually want
to iterate on. Changing K therefore never requires regenerating hidden states,
and never requires a second pass over them.

The map is emitted at the *draft config's* ``vocab_size``, which is the length
the model's ``t2d`` buffer is registered with. Sizing it from the target config
instead would produce a file that silently fails to load whenever the target
declares ``padded_vocab_size``.

Two sources, same numbers. ``--hidden-states-path`` reads the prepared
features, which is exact but serial -- for a large gzipped dataset it is not
merely slow, it is impractical, since every file is decompressed in full to
recover two small tensors. ``--data-path`` re-tokenizes the source JSONL with
the same stack the capture used, in parallel, without touching the features at
all; pass it the same tokenizer, template, max length, and filters.

Survey several sizes before committing to one (writes nothing):

    python scripts/build_vocab_mapping.py \
        --data-path ./cache/dataset/train.jsonl \
        --tokenizer-path Qwen/Qwen3-8B --chat-template qwen --max-length 4096 \
        --draft-model-config configs/qwen3.6-27b-dspark.json \
        --draft-vocab-size 16000,32000,48000,64000

Then write the chosen one, reusing the cached counts:

    python scripts/build_vocab_mapping.py \
        --data-path ./cache/dataset/train.jsonl \
        --tokenizer-path Qwen/Qwen3-8B --chat-template qwen --max-length 4096 \
        --draft-model-config configs/qwen3.6-27b-dspark-draftvocab64k.json \
        --output-path ./cache/vocab_mapping/qwen3.6-27b-k64000.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Optional

import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Derive t2d/d2t from prepared offline features, without "
            "regenerating them and without a second pass per vocabulary size."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--hidden-states-path",
        type=Path,
        help=(
            "Directory of prepared offline features (.ckpt / .ckpt.gz). Exact, "
            "but reads every file serially -- impractical for a large gzipped "
            "dataset; prefer --data-path there."
        ),
    )
    source.add_argument(
        "--data-path",
        type=Path,
        help=(
            "Raw conversation JSONL. Re-tokenizes with the same stack "
            "prepare_hidden_states.py uses, in parallel and without touching "
            "the features at all. Pass the same tokenizer/template/max-length "
            "and --minimum-valid-tokens the capture ran with, or the counts "
            "will describe a different corpus than training sees."
        ),
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Target model/tokenizer path. Required with --data-path.",
    )
    parser.add_argument(
        "--chat-template",
        default=None,
        help="Chat template used at capture time. Required with --data-path.",
    )
    parser.add_argument(
        "--is-preformatted",
        action="store_true",
        help="Source rows already have the chat template applied.",
    )
    parser.add_argument(
        "--minimum-valid-tokens",
        type=int,
        default=None,
        help=(
            "Mirror prepare_hidden_states.py's filter so dropped samples do "
            "not contribute frequencies."
        ),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Mirror prepare_hidden_states.py's --num-samples.",
    )
    parser.add_argument(
        "--build-dataset-num-proc",
        type=int,
        default=8,
        help="Tokenization worker processes for --data-path.",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        type=Path,
        default=Path("./cache"),
        help=(
            "Root for the conversation and tokenized-dataset caches "
            "(<root>/hf_dataset and <root>/processed_dataset). Point it at a "
            "partition with room; matching data.cache_dir lets a later "
            "training run reuse the tokenization."
        ),
    )
    parser.add_argument(
        "--draft-model-config",
        type=Path,
        required=True,
        help=(
            "Draft config JSON. Supplies vocab_size (the map's length, matching "
            "the model's t2d buffer) and the default draft_vocab_size."
        ),
    )
    parser.add_argument(
        "--draft-vocab-size",
        default=None,
        help=(
            "One size, or a comma-separated list to survey. Defaults to the "
            "draft config's draft_vocab_size. A list writes nothing."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=("Where to write the {t2d, d2t} file. Omit to only report coverage."),
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Truncate each sample's ids/mask, matching data.max_length.",
    )
    parser.add_argument(
        "--counts-cache",
        type=Path,
        default=None,
        help=(
            "Token-count cache. Defaults to <hidden-states-path>/.token_counts.pt "
            "for --hidden-states-path, and "
            "<dataset-cache-dir>/vocab_mapping/.token_counts.pt for --data-path. "
            "Reused only when the corpus fingerprint is unchanged."
        ),
    )
    parser.add_argument(
        "--recount",
        action="store_true",
        help="Ignore any cached counts and rescan the features.",
    )
    return parser


def _feature_identity(hidden_states_path: str, max_length: Optional[int]) -> str:
    """Fingerprint the feature set so a stale cache is never silently reused."""
    from specforge.runtime.data_plane.offline_reader import list_feature_files

    entries = []
    for path in list_feature_files(hidden_states_path):
        stat = os.stat(path)
        entries.append((os.path.abspath(path), stat.st_size, stat.st_mtime_ns))
    payload = json.dumps(
        {"kind": "offline-features-v1", "files": entries, "max_length": max_length},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _dataset_identity(args, vocab_size: int) -> str:
    """Fingerprint the tokenization inputs, so a cache answers for its own corpus."""
    stat = os.stat(args.data_path)
    payload = json.dumps(
        {
            "kind": "conversations-jsonl-v1",
            "path": os.path.abspath(args.data_path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "tokenizer": args.tokenizer_path,
            "chat_template": args.chat_template,
            "max_length": args.max_length,
            "is_preformatted": bool(args.is_preformatted),
            "minimum_valid_tokens": args.minimum_valid_tokens,
            "num_samples": args.num_samples,
            "vocab_size": vocab_size,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def count_dataset_tokens(args, *, vocab_size: int) -> Counter:
    """Count loss-bearing tokens by re-tokenizing the source conversations.

    Runs the same tokenizer, chat template, truncation, and trainable-token
    filter that ``prepare_hidden_states.py`` applied, so the frequencies match
    the captured features without reading them. Any of those knobs differing
    from the capture yields a mapping for a different corpus, which is why they
    are all explicit rather than defaulted.
    """
    from datasets import Dataset

    from specforge.data.preprocessing import build_eagle3_dataset
    from specforge.utils import load_tokenizer, safe_conversations_generator

    tokenizer = load_tokenizer(args.tokenizer_path)
    dataset = Dataset.from_generator(
        generator=safe_conversations_generator,
        gen_kwargs={"file_path": str(args.data_path)},
        # Pinned under --dataset-cache-dir like prepare_hidden_states.py does.
        # Left to its default this lands in ~/.cache/huggingface, which is
        # rarely the partition with room for a 600k-conversation corpus.
        cache_dir=str(args.dataset_cache_dir / "hf_dataset"),
        num_proc=min(args.build_dataset_num_proc, 32),
    )
    if args.num_samples is not None:
        dataset = dataset.select(range(args.num_samples))
    processed = build_eagle3_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        cache_dir=str(args.dataset_cache_dir / "processed_dataset"),
        cache_key=_dataset_identity(args, vocab_size),
        is_preformatted=args.is_preformatted,
        num_proc=args.build_dataset_num_proc,
        minimum_valid_tokens=args.minimum_valid_tokens,
    )
    print(f"Tokenized {len(processed)} samples")
    return tally_loss_tokens(processed, vocab_size=vocab_size)


def tally_loss_tokens(dataset, *, vocab_size: int, batch_size: int = 512) -> Counter:
    """Sum loss-bearing token frequencies over a tokenized dataset.

    Streams batches rather than touching ``dataset["input_ids"]``: column access
    materializes every sequence as Python ints at once, which for a corpus this
    feature targets is tens of gigabytes and looks like a hang. Frequencies then
    accumulate into one dense bincount per batch instead of per token, keeping
    the work in tensors rather than in a billion-iteration Python loop.
    """
    from tqdm import tqdm

    totals = torch.zeros(vocab_size, dtype=torch.int64)
    batches = (len(dataset) + batch_size - 1) // batch_size
    for batch in tqdm(
        dataset.iter(batch_size=batch_size),
        total=batches,
        desc="Counting tokens for vocab mapping",
    ):
        selected = []
        for input_ids, loss_mask in zip(batch["input_ids"], batch["loss_mask"]):
            ids = torch.as_tensor(input_ids).reshape(-1)
            mask = torch.as_tensor(loss_mask).reshape(-1)
            kept = ids[mask.to(dtype=torch.bool)]
            if kept.numel():
                selected.append(kept)
        if not selected:
            continue
        flat = torch.cat(selected).long()
        if int(flat.min()) < 0 or int(flat.max()) >= vocab_size:
            raise ValueError(
                f"token id {int(flat.max())} is outside the draft config's "
                f"vocab_size {vocab_size}"
            )
        totals += torch.bincount(flat, minlength=vocab_size)

    present = torch.nonzero(totals, as_tuple=False).flatten()
    return Counter({int(token): int(totals[token]) for token in present.tolist()})


def load_or_count_tokens(args, *, vocab_size: int, counts_cache: Path) -> Counter:
    """Return loss-bearing token frequencies, counting at most once per corpus."""
    from_features = args.hidden_states_path is not None
    identity = (
        _feature_identity(str(args.hidden_states_path), args.max_length)
        if from_features
        else _dataset_identity(args, vocab_size)
    )
    if not args.recount and counts_cache.exists():
        cached = torch.load(counts_cache, map_location="cpu", weights_only=False)
        if cached.get("identity") == identity:
            print(f"Reusing token counts from {counts_cache}")
            return Counter(cached["counts"])
        print(f"{counts_cache} describes a different corpus; recounting.")

    if from_features:
        from specforge.data.vocab_mapping import count_effective_feature_tokens

        print(f"Counting loss-bearing tokens under {args.hidden_states_path} ...")
        counts = count_effective_feature_tokens(
            str(args.hidden_states_path),
            max_length=args.max_length,
            target_vocab_size=vocab_size,
        )
    else:
        print(f"Tokenizing {args.data_path} to count loss-bearing tokens ...")
        counts = count_dataset_tokens(args, vocab_size=vocab_size)

    counts_cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = counts_cache.with_suffix(f".{os.getpid()}.tmp")
    torch.save({"identity": identity, "counts": dict(counts)}, temporary)
    os.replace(temporary, counts_cache)
    print(f"Cached token counts at {counts_cache}")
    return counts


def coverage_ratio(counts: Counter, draft_vocab_size: int) -> float:
    """Share of loss-bearing token occurrences the top-K tokens account for.

    This is the ceiling on acceptance: a target token outside the draft
    vocabulary can never be proposed, so that position is always rejected.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    kept = sum(frequency for _, frequency in counts.most_common(draft_vocab_size))
    return kept / total


def write_mapping(
    counts: Counter,
    *,
    draft_vocab_size: int,
    vocab_size: int,
    output_path: Path,
) -> None:
    from specforge.core.compact_teacher import validate_vocab_mapping_consistency
    from specforge.data.preprocessing import process_token_dict_to_mappings

    d2t, t2d = process_token_dict_to_mappings(
        Counter(counts), draft_vocab_size, vocab_size
    )
    # The same invariant the model checks on install; catching it here keeps a
    # broken file from ever reaching a training run.
    validate_vocab_mapping_consistency(t2d, d2t)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f".{os.getpid()}.tmp")
    torch.save({"d2t": d2t, "t2d": t2d}, temporary)
    os.replace(temporary, output_path)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    with open(args.draft_model_config, encoding="utf-8") as handle:
        draft_config = json.load(handle)
    vocab_size = int(draft_config["vocab_size"])

    if args.draft_vocab_size is not None:
        sizes = [int(item) for item in str(args.draft_vocab_size).split(",") if item]
    else:
        configured = draft_config.get("draft_vocab_size")
        if configured is None:
            raise ValueError(
                f"{args.draft_model_config} has no draft_vocab_size; pass "
                "--draft-vocab-size explicitly"
            )
        sizes = [int(configured)]
    for size in sizes:
        if not 0 < size <= vocab_size:
            raise ValueError(
                f"draft_vocab_size must be in (0, {vocab_size}], got {size}"
            )
    if len(sizes) > 1 and args.output_path is not None:
        raise ValueError(
            "--output-path writes a single mapping; pass one --draft-vocab-size"
        )

    if args.data_path is not None:
        missing = [
            name
            for name, value in (
                ("--tokenizer-path", args.tokenizer_path),
                ("--chat-template", args.chat_template),
                ("--max-length", args.max_length),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                f"--data-path re-tokenizes the corpus and must reproduce the "
                f"capture exactly; missing {missing}"
            )

    if args.counts_cache is not None:
        counts_cache = args.counts_cache
    elif args.hidden_states_path is not None:
        counts_cache = args.hidden_states_path / ".token_counts.pt"
    else:
        counts_cache = args.dataset_cache_dir / "vocab_mapping" / ".token_counts.pt"
    counts = load_or_count_tokens(
        args, vocab_size=vocab_size, counts_cache=counts_cache
    )
    distinct = len(counts)
    print(f"Distinct loss-bearing tokens: {distinct} of {vocab_size}")

    for size in sizes:
        ratio = coverage_ratio(counts, size)
        note = ""
        if size > distinct:
            note = f"  (only {distinct} tokens ever appear; the rest are padding)"
        print(f"  top {size:>7} token frequency ratio: {ratio:7.2%}{note}")

    if args.output_path is None:
        print(
            "\nNo --output-path given, so nothing was written. The ratio above "
            "is the acceptance ceiling for that size."
        )
        return 0

    write_mapping(
        counts,
        draft_vocab_size=sizes[0],
        vocab_size=vocab_size,
        output_path=args.output_path,
    )
    print(f"\nWrote mapping to {args.output_path}")
    print("Point model.vocab_mapping_path at it to skip the training-time scan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
