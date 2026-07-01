"""
Entry point to generate the synthetic Bangla FIR dataset.

Usage:
    python scripts/generate_data.py --n_samples 3000 --output_dir data/synthetic
"""

import argparse
import json
import sys
from pathlib import Path

# Allow imports from sibling package
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.synthetic.fir_templates import generate_dataset, build_instruction_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=3000)
    parser.add_argument("--output_dir", default="data/synthetic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n_samples} synthetic FIR samples (seed={args.seed})...")
    samples = generate_dataset(n_samples=args.n_samples, seed=args.seed)
    pairs = build_instruction_pairs(samples)

    n = len(pairs)
    train_end = int(n * args.train_ratio)
    val_end = int(n * (args.train_ratio + args.val_ratio))

    splits = {
        "train": pairs[:train_end],
        "validation": pairs[train_end:val_end],
        "test": pairs[val_end:],
    }

    for split_name, data in splits.items():
        out_path = out_dir / f"{split_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  {split_name:12s}: {len(data):5d} samples → {out_path}")

    # Write dataset stats
    stats = {
        "total": n,
        "splits": {k: len(v) for k, v in splits.items()},
        "offense_distribution": {},
    }
    from collections import Counter
    counter = Counter(p["offense_type"] for p in pairs)
    stats["offense_distribution"] = dict(counter.most_common())

    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\nDataset stats:\n{json.dumps(stats, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
