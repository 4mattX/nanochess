"""
Inspect chess training data quality.

Shows statistics about your prepared parquet data:
- Number of positions
- Eval coverage
- Move distribution
- Sample positions

Usage:
    python dev/inspect_data.py
    python dev/inspect_data.py --data-dir ~/.cache/nanochess/pra_data
"""

import argparse
import os
import glob
from collections import Counter
from pathlib import Path


def inspect_data(data_dir: str):
    """Inspect parquet data and print statistics."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("Error: pyarrow required")
        print("Install with: pip install pyarrow")
        return

    # Find all parquet files
    pattern = os.path.join(data_dir, "shard_*.parquet")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No parquet files found in {data_dir}")
        print("Run data fetching first:")
        print("  bash dev/quick_fetch.sh")
        return

    print("=" * 60)
    print("NANOCHESS DATA INSPECTION")
    print("=" * 60)
    print(f"\nData directory: {data_dir}")
    print(f"Parquet files: {len(files)}")
    print()

    # Load all data
    total_positions = 0
    all_moves = []
    all_evals = []
    sample_positions = []

    for i, file in enumerate(files):
        table = pq.read_table(file)
        df = table.to_pandas()

        positions = len(df)
        total_positions += positions

        # Collect samples from first file
        if i == 0 and len(df) > 0:
            sample_positions = df.head(5).to_dict('records')

        # Collect move statistics
        all_moves.extend(df['best_move'].tolist())

        # Collect eval statistics
        if 'eval' in df.columns:
            evals = df['eval'].tolist()
            all_evals.extend([e for e in evals if e != 0])

        # Print file info
        is_val = i == len(files) - 1
        label = "VAL" if is_val else "TRAIN"
        print(f"  {label:5s} {Path(file).name:20s} {positions:8,} positions")

    print()
    print("-" * 60)
    print(f"TOTAL: {total_positions:,} positions")
    print("-" * 60)
    print()

    # Eval coverage
    if all_evals:
        eval_coverage = len(all_evals) / total_positions * 100
        print(f"Evaluation coverage: {eval_coverage:.1f}% ({len(all_evals):,} positions)")

        # Eval statistics
        avg_eval = sum(all_evals) / len(all_evals)
        min_eval = min(all_evals)
        max_eval = max(all_evals)

        print(f"  Average eval: {avg_eval/100:+.2f} pawns")
        print(f"  Range: {min_eval/100:+.2f} to {max_eval/100:+.2f} pawns")
        print()
    else:
        print("No evaluations found in data")
        print()

    # Move statistics
    print("Move distribution (top 10):")
    move_counts = Counter(all_moves)
    for move, count in move_counts.most_common(10):
        pct = count / len(all_moves) * 100
        print(f"  {move:6s} {count:8,} ({pct:5.2f}%)")
    print()

    # Unique moves
    unique_moves = len(move_counts)
    print(f"Unique moves: {unique_moves:,}")
    print()

    # Sample positions
    print("Sample positions:")
    print("-" * 60)
    for i, pos in enumerate(sample_positions[:3], 1):
        print(f"\n{i}. Position:")
        print(f"   FEN: {pos['fen']}")
        print(f"   Move: {pos['best_move']}")
        if 'eval' in pos and pos['eval'] != 0:
            print(f"   Eval: {pos['eval']/100:+.2f} pawns")
    print()

    # Data quality checks
    print("=" * 60)
    print("QUALITY CHECKS")
    print("=" * 60)
    print()

    checks = []

    # Check 1: Sufficient data
    if total_positions < 10000:
        checks.append(("⚠️  WARN", "Less than 10K positions - consider fetching more data"))
    elif total_positions < 100000:
        checks.append(("ℹ️  INFO", f"{total_positions:,} positions - sufficient for initial training"))
    else:
        checks.append(("✓ GOOD", f"{total_positions:,} positions - good dataset size"))

    # Check 2: Eval coverage
    if all_evals:
        if eval_coverage < 50:
            checks.append(("⚠️  WARN", f"Low eval coverage ({eval_coverage:.1f}%) - value head may underperform"))
        else:
            checks.append(("✓ GOOD", f"Eval coverage {eval_coverage:.1f}% - value head will train well"))
    else:
        checks.append(("⚠️  WARN", "No evaluations - only policy head will train"))

    # Check 3: Move diversity
    avg_frequency = len(all_moves) / unique_moves if unique_moves > 0 else 0
    if unique_moves < 1000:
        checks.append(("⚠️  WARN", f"Low move diversity ({unique_moves} unique) - may overfit"))
    elif unique_moves > 2000:
        checks.append(("✓ GOOD", f"High move diversity ({unique_moves:,} unique moves)"))
    else:
        checks.append(("ℹ️  INFO", f"Moderate move diversity ({unique_moves:,} unique moves)"))

    # Check 4: Train/val split
    if len(files) < 2:
        checks.append(("⚠️  WARN", "No validation split detected"))
    else:
        checks.append(("✓ GOOD", f"{len(files)-1} training shards, 1 validation shard"))

    for status, message in checks:
        print(f"{status:8s} {message}")

    print()
    print("=" * 60)
    print()

    if total_positions < 10000:
        print("Recommendation: Fetch more data")
        print("  bash dev/quick_fetch.sh")
        print()
    elif total_positions < 100000:
        print("Recommendation: Ready for quick training experiments")
        print("  python -m nanochess")
        print()
    else:
        print("Recommendation: Ready for full training run")
        print("  python -m nanochess")
        print()


def main():
    parser = argparse.ArgumentParser(description="Inspect chess training data")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.expanduser("~/.cache/nanochat/pra_data"),
        help="Data directory to inspect",
    )
    args = parser.parse_args()

    inspect_data(args.data_dir)


if __name__ == "__main__":
    main()
