"""Command-line entry point for splitting configured TemPER domains."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time

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
from temper.logging import format_elapsed, progress_task


logger = logging.getLogger(__name__)


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
    command_started_at = time.monotonic()
    config_file = Path(config_file)
    split_config = _load_split_config(config_file)
    reproduce_path = _reproduce_config_path(config_file)
    dumpfn(split_config, reproduce_path, indent=2)

    root_path = split_config.root_path
    output_path = split_config.output_path

    valid_domains = list(
        split_config.domains
        if split_config.domains is not None
        else (
            metadata.parent.name
            for metadata in Path(root_path).resolve().rglob(DEFAULT_METADATA_FILE)
        )
    )
    logger.info(
        "Starting split command from %s: %d domain(s), %d repeat(s), "
        "method=%s, requested device=%s.",
        config_file.resolve(),
        len(valid_domains),
        split_config.split_repeats,
        split_config.train_val_split_method,
        split_config.quests_adapter_config.device,
    )
    logger.info(
        "Input root: %s; output root: %s; reproduction config: %s.",
        root_path,
        output_path,
        reproduce_path.resolve(),
    )
    logger.debug(
        "Resolved split seeds: train/test=%s; train/validation=%s.",
        split_config.trainval_test_split_seeds,
        split_config.train_val_split_seeds,
    )

    for domain_index, domain in enumerate(valid_domains, start=1):
        domain_started_at = time.monotonic()
        domain_context = f"[domain {domain_index}/{len(valid_domains)}: {domain}]"
        logger.info("%s Starting domain.", domain_context)
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
        logger.debug(
            "%s Persisted %d grouped-domain record(s) to %s.",
            domain_context,
            len(grouped_domains),
            domain_output_path / DEFAULT_GROUPED_DOMAIN_FILE,
        )
        # Split groups into train, validation and test sets.
        split_groups = []
        for strategy_index, grouped_domain in enumerate(grouped_domains, start=1):
            logger.debug(
                "%s Starting grouping strategy %d/%d: %r.",
                domain_context,
                strategy_index,
                len(grouped_domains),
                getattr(grouped_domain, "grouping_strategy", grouped_domain),
            )
            split_groups.extend(split_grouped_domain(grouped_domain, split_config))
        dumpfn(
            split_groups,
            domain_output_path / DEFAULT_SPLIT_GROUPS_FILE,
            indent=2,
        )
        logger.debug(
            "%s Persisted %d split-group record(s) to %s.",
            domain_context,
            len(split_groups),
            domain_output_path / DEFAULT_SPLIT_GROUPS_FILE,
        )
        # Write train, val and test sets to files.
        training_units = []
        logger.info(
            "%s Exporting split datasets for %d split group(s).",
            domain_context,
            len(split_groups),
        )
        with progress_task(
            logger,
            f"Exporting split datasets for domain {domain!r}",
            total=len(split_groups),
            unit="split groups",
        ) as progress:
            for split_group in split_groups:
                group_label = getattr(split_group, "group_name", None)
                if group_label is None:
                    group_label = str(split_group)
                progress.update(
                    detail=(
                        f"group {group_label!r}, "
                        f"repeat {getattr(split_group, 'repeat_id', '?')}"
                    )
                )
                training_units_local, _ = write_all_sets_in_split_group_to_extxyz(
                    split_group, root_path, output_path,
                    write_validation=split_config.write_validation,
                    write_extra_tests=split_config.write_extra_tests,
                    all_split_groups=split_groups,
                )
                training_units.extend(training_units_local)
                progress.advance()
        dumpfn(
            training_units,
            domain_output_path / DEFAULT_TRAINING_UNITS_FILE,
            indent=2,
        )
        logger.info(
            "%s Completed domain with %d split group(s) and %d training "
            "unit(s) in %s. Artifacts: %s.",
            domain_context,
            len(split_groups),
            len(training_units),
            format_elapsed(time.monotonic() - domain_started_at),
            domain_output_path,
        )
    logger.info(
        "Split command completed for %d domain(s) in %s. Output root: %s.",
        len(valid_domains),
        format_elapsed(time.monotonic() - command_started_at),
        output_path,
    )
    return 0
