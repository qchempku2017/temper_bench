"""Command-line entry point for splitting configured TemPER domains."""
from __future__ import annotations

import argparse
from pathlib import Path

from monty.serialization import dumpfn, loadfn

from temper.grouping import partition_domain_into_groups
from temper.splitting.split import split_grouped_domain
from temper.splitting.io import write_all_sets_in_split_group_to_extxyz
from temper.schemas.split import SplitConfig

from temper.utils.defaults import (
    DEFAULT_SPLIT_CONFIG_FILE,
    DEFAULT_METADATA_FILE,
    DEFAULT_GROUPED_DOMAIN_FILE,
    DEFAULT_SPLIT_GROUPS_FILE,
    DEFAULT_TRAINING_UNITS_FILE
)


def add_split_parser(subparser: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Build the argument parser for the split entry point."""
    parser_split = subparser.add_parser(
        "split",
        help="Group and split domains using a JSON or YAML configuration file."
    )

    parser_split.add_argument(
        "-c",
        "--config-file",
        type=Path,
        default=Path(DEFAULT_SPLIT_CONFIG_FILE),
        help=(
            "JSON or YAML SplitConfig file. Defaults to the "
            "DEFAULT_SPLIT_CONFIG_FILE environment variable or split_config.json."
        ),
    )

    return parser_split


def _load_split_config(config_file: Path) -> SplitConfig:
    """Load and validate a JSON or YAML split configuration."""
    if config_file.suffix.lower() not in {".json", ".yaml", ".yml"}:
        raise ValueError(
            f"Split configuration must be JSON or YAML, got {config_file}."
        )
    loaded = loadfn(config_file)
    if isinstance(loaded, SplitConfig):
        return loaded
    return SplitConfig.model_validate(loaded)


def _reproduce_config_path(config_file: Path) -> Path:
    """Return the non-overwriting JSON reproduction path for ``config_file``."""
    return config_file.with_name(f"{config_file.stem}_reproduce.json")


def split_cli(config_file: Path | str = DEFAULT_SPLIT_CONFIG_FILE) -> int:
    """Load configuration, persist exact seeds, then group, split, and export."""
    config_file = Path(config_file)
    split_config = _load_split_config(config_file)
    dumpfn(split_config, _reproduce_config_path(config_file), indent=2)

    root_path = split_config.root_path
    output_path = split_config.output_path

    valid_domains = (
        split_config.domains
        if split_config.domains is not None
        else [
            metadata.parent.name
            for metadata in Path(root_path).resolve().rglob(DEFAULT_METADATA_FILE)
        ]
    )

    for domain in valid_domains:
        domain_output_path = output_path / domain
        domain_output_path.mkdir(parents=True, exist_ok=True)
        # Partition domain into groups.
        grouped_domains = partition_domain_into_groups(
            domain, root_path=root_path, metadata_file_name=DEFAULT_METADATA_FILE
        )
        dumpfn(
            grouped_domains,
            domain_output_path / DEFAULT_GROUPED_DOMAIN_FILE,
            indent=2,
        )
        # Split groups into train, validation and test sets.
        split_groups = []
        for grouped_domain in grouped_domains:
            split_groups.extend(split_grouped_domain(grouped_domain, split_config))
        dumpfn(
            split_groups,
            domain_output_path / DEFAULT_SPLIT_GROUPS_FILE,
            indent=2,
        )
        # Write train, val and test sets to files.
        training_units = []
        for split_group in split_groups:
            training_units_local, _ = write_all_sets_in_split_group_to_extxyz(
                split_group, root_path, output_path,
                write_validation=split_config.write_validation,
                write_extra_tests=split_config.write_extra_tests,
                all_split_groups=split_groups,
            )
            training_units.extend(training_units_local)
        dumpfn(
            training_units,
            domain_output_path / DEFAULT_TRAINING_UNITS_FILE,
            indent=2,
        )
    return 0
