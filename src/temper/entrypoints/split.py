"""Command-line entry point for splitting configured TemPER domains."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.temper.grouping import partition_domain_groups
from src.temper.splitting.split import split_grouped_domain
from src.temper.schemas.split import SplitConfig


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the split entry point."""
    parser = argparse.ArgumentParser(
        description="Split every configured grouping of a TemPER data domain."
    )
    parser.add_argument("domain", help="Domain directory name under --root-path.")
    parser.add_argument(
        "--root-path", type=Path, required=True, help="Directory containing the domain."
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="JSON file receiving split groups."
    )
    parser.add_argument("--method", choices=("random", "quests"), default="quests")
    parser.add_argument("--split-repeats", type=int, default=1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument(
        "--seed", type=int, default=0, help="Base seed used to derive deterministic repeat seeds."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run splitting and save every produced ``SplitGroup`` as a JSON list."""
    args = build_parser().parse_args(argv)
    if args.split_repeats <= 0:
        raise ValueError("--split-repeats must be positive.")
    if not 0.0 < args.test_ratio < 1.0:
        raise ValueError("--test-ratio must be in (0, 1).")

    repeat_offsets = range(args.split_repeats)
    config = SplitConfig(
        root_path=args.root_path,
        split_repeats=args.split_repeats,
        trainval_test_split_seeds=[args.seed + offset for offset in repeat_offsets],
        train_val_split_seeds=[args.seed + args.split_repeats + offset for offset in range(args.split_repeats)],
        test_ratio=args.test_ratio,
        train_val_split_method=args.method,
    )
    split_groups = [
        split_group
        for grouped_domain in partition_domain_groups(args.domain, root_path=args.root_path)
        for split_group in split_grouped_domain(grouped_domain, config)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "[\n" + ",\n".join(group.model_dump_json(indent=2) for group in split_groups) + "\n]\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
