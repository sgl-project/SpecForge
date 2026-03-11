#!/usr/bin/env python3
"""Mix multiple datasets according to specified ratios."""
import json
import random
from pathlib import Path
from typing import List, Tuple

def load_jsonl(path: str) -> List[dict]:
    """Load JSONL file."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def mix_datasets(
    datasets: List[Tuple[str, float]],  # [(path, ratio)]
    output_path: str,
    total_samples: int = None,
    seed: int = 42
):
    """Mix datasets according to ratios."""
    random.seed(seed)

    # Load all datasets
    all_data = []
    for path, ratio in datasets:
        data = load_jsonl(path)
        print(f"Loaded {len(data)} samples from {path}")

        if total_samples:
            # Sample according to ratio
            n_samples = int(total_samples * ratio)
            sampled = random.sample(data, min(n_samples, len(data)))
        else:
            # Use all data weighted by ratio
            sampled = random.sample(data, int(len(data) * ratio / sum(r for _, r in datasets)))

        all_data.extend(sampled)
        print(f"Added {len(sampled)} samples ({ratio*100:.1f}%)")

    # Shuffle combined dataset
    random.shuffle(all_data)

    # Normalize IDs to strings (fix type mismatch between datasets)
    for i, item in enumerate(all_data):
        item['id'] = str(item.get('id', i))

    # Save mixed dataset
    with open(output_path, 'w') as f:
        for item in all_data:
            f.write(json.dumps(item) + '\n')

    print(f"\nSaved {len(all_data)} mixed samples to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--total-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Mix for Experiment J: 45% ShareGPT, 35% UltraChat, 20% PerfectBlend
    base_dir = Path(__file__).parent.parent / "cache" / "dataset"
    datasets = [
        (str(base_dir / "sharegpt_train.jsonl"), 0.45),
        (str(base_dir / "ultrachat_train.jsonl"), 0.35),
        (str(base_dir / "perfectblend_train.jsonl"), 0.20),
    ]

    mix_datasets(
        datasets=datasets,
        output_path=args.output,
        total_samples=args.total_samples,
        seed=args.seed
    )
