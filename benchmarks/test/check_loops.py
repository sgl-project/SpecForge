#!/usr/bin/env python3
"""Scan a bench answers JSONL for likely greedy-decode repetition loops.

Heuristic: any 16-word phrase that repeats >= 5 times in a single generation
is flagged. Catches both the "We can comply" / "The user wants X" cycles and
the long ellipsis runs without false-positiving on normal repeated tokens
(stop words, code keywords, etc.) since 5-grams have to match exactly.

Usage:
    python benchmarks/check_loops.py results/<run>/results_<ts>_answers.jsonl
"""
import argparse
import json
from collections import Counter


def has_loop(text: str, ngram, min_repeats) -> tuple[bool, str | None]:
    if not text:
        return False, None
    words = text.split()
    if len(words) < ngram * min_repeats:
        return False, None
    counts = Counter(
        " ".join(words[i : i + ngram]) for i in range(len(words) - ngram + 1)
    )
    top, n = counts.most_common(1)[0]
    return (n >= min_repeats), (top if n >= min_repeats else None)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("path", help="answers JSONL produced by bench_eagle3.py")
    p.add_argument("--ngram", type=int, default=16)
    p.add_argument("--min-repeats", type=int, default=5)
    p.add_argument("--show", type=int, default=10000, help="max examples to print")
    args = p.parse_args()

    flagged_by_bench: dict[str, int] = {}
    examples: list[dict] = []
    total_by_bench: dict[str, int] = {}

    with open(args.path) as f:
        for line in f:
            rec = json.loads(line)
            bench = rec.get("benchmark", "?")
            total_by_bench[bench] = total_by_bench.get(bench, 0) + 1
            for key, value in (rec.get("outputs") or {}).items():
                if not isinstance(value, str):
                    continue
                looped, ngram = has_loop(value, args.ngram, args.min_repeats)
                if looped:
                    flagged_by_bench[bench] = flagged_by_bench.get(bench, 0) + 1
                    if len(examples) < args.show:
                        examples.append(
                            {
                                "benchmark": bench,
                                "index": rec.get("index"),
                                "key": key,
                                "ngram": ngram,
                            }
                        )
                    break  # one flag per record is enough

    print(f"file: {args.path}")
    print(f"ngram={args.ngram}  min_repeats={args.min_repeats}")
    print()
    width = max((len(b) for b in total_by_bench), default=10)
    print(f"{'benchmark':<{width}}  flagged / total")
    print("-" * (width + 20))
    for bench, total in sorted(total_by_bench.items()):
        n = flagged_by_bench.get(bench, 0)
        print(f"{bench:<{width}}  {n:>5} / {total}")
    if examples:
        print(f"\nfirst {len(examples)} flagged records:")
        for e in examples:
            print(f"  [{e['benchmark']}#{e['index']}.{e['key']}] repeats: {e['ngram']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
