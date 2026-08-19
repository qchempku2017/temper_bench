"""Command-line entry point for splitting configured TemPER domains."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Literal

import multiprocessing as mp
from monty.serialization import dumpfn

from src.temper.grouping import partition_domain_into_groups
from src.temper.splitting.split import split_grouped_domain
from src.temper.splitting.io import write_all_sets_in_split_group_to_extxyz
from src.temper.schemas.split import SplitConfig
from src.temper.schemas.quests_adapter import QuestsAdapterConfig

from src.temper.utils.defaults import (
    DEFAULT_DATA_DIR,
    DEFAULT_SPLIT_RESULTS_DIR,
    DEFAULT_SPLIT_REPEATS,
    DEFAULT_TRAIN_RATIOS,
    DEFAULT_TEST_RATIO,
    DEFAULT_MAX_N_TRAIN,
    DEFAULT_METADATA_FILE,
    DEFAULT_GROUPED_DOMAIN_FILE,
    DEFAULT_SPLIT_GROUPS_FILE,
    DEFAULT_TRAINING_UNITS_FILE
)


def add_split_parser(subparser: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Build the argument parser for the split entry point."""
    parser_split = subparser.add_parser(
        "split",
        help="Load data domains, group and split groups into train/validation and test sets."
    )

    parser_split.add_argument(
        "-r",
        "--root-path",
        type=Path,
        default=Path(DEFAULT_DATA_DIR),
        help="Directory containing the domain. If not specified, will use environment variable"
             " DEFAULT_DATA_DIR or `./data`."
    )
    parser_split.add_argument(
        "-d",
        "--domains",
        nargs="+",
        type=str,
        default=None,
        help="Name of domain directories to split under --root-path. If not specified, will use"
             " all domain subfolders that contain a valid metadata.json file."
    )
    parser_split.add_argument(
        "-o",
        "--output-path",
        type=Path,
        default=Path(DEFAULT_SPLIT_RESULTS_DIR),
        help="Directory to save the split results. If not specified, will use environment variable"
             " DEFAULT_SPLIT_RESULTS_DIR or `./split_results`."
    )
    parser_split.add_argument(
        "-m",
        "--method",
        choices=("random", "quests"),
        default="quests",
        help="Method to split train and validation sets. Default to `quests`,"
             " i.e., split by maximization of information entropy."
    )
    parser_split.add_argument(
        "-s",
        "--split-repeats",
        type=int,
        default=DEFAULT_SPLIT_REPEATS,
        help="Number of times to repeat the train/validation/test split on one group."
             " Default to environment variable DEFAULT_SPLIT_REPEATS or 3."
    )
    parser_split.add_argument(
        "-test",
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help="Ratio of test set size to the total number of samples in each group."
             " Default to environment variable DEFAULT_TEST_RATIO or 0.2."
    )
    parser_split.add_argument(
        "-train",
        "--train-ratios",
        type=float,
        nargs="+",
        default=DEFAULT_TRAIN_RATIOS,
        help="Ratio of train set size to the number of samples in the train/validation set of"
             " each group."
             " Default to environment variable DEFAULT_TRAIN_RATIOS"
             " or [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]."
    )
    parser_split.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed used to derive deterministic repeat seeds."
    )
    parser_split.add_argument(
        "--max-train-size",
        type=int,
        default=DEFAULT_MAX_N_TRAIN,
        help="Maximum number of structures in the largest training set."
    )

    # Add quests adapter config.
    parser_split.add_argument(
        "--descriptor-k",
        type=int,
        default=32,
        help="Number of neighbors to consider when computing the QUESTS descriptor."
    )
    parser_split.add_argument(
        "--descriptor-cutoff",
        type=float,
        default=5.0,
        help="Cutoff distance for the QUESTS descriptor."
    )
    parser_split.add_argument(
        "--descriptor-dtype",
        type=str,
        default="float32",
        choices=("float32", "float64"),
        help="Data type of the QUESTS descriptor."
    )
    parser_split.add_argument(
        "--compute-descriptor-chunk-size",
        type=int,
        default=200,
        help="Chunk size for computing the QUESTS descriptor every chunk_size samples."
             " Used to avoid memory overflow when total number of structures is too large."
    )
    parser_split.add_argument(
        "--entropy-bandwidth",
        type=float,
        default=0.015,
        help="Bandwidth for the QUESTS entropy computation."
    )
    parser_split.add_argument(
        "--entropy-batch-size",
        type=int,
        default=20000,
        help="Batch size for the QUESTS entropy computation."
    )
    parser_split.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "gpu",)
    )
    parser_split.add_argument(
        "--gpu-device",
        type=str,
        default=None,
        help="GPU device to use for QUESTS computation."
    )
    parser_split.add_argument(
        "--numba-threads",
        type=int,
        default=min(8, max(1, mp.cpu_count() // 2)),
        help="Number of threads to use for QUESTS computation."
    )

    # Dataset writing options.
    parser_split.add_argument(
        "--write-validation",
        action="store_true",
        help="Write validation set to disk. Default to False."
    )
    parser_split.add_argument(
        "--no-write-extra-tests",
        action="store_true",
        help="Do not write extra test sets to disk. Default to False, will write."
    )

    return parser_split


def split_cli(
        root_path: Path,
        output_path: Path,
        domains: List[str] | None,
        split_repeats: int,
        seed: int,
        test_ratio: float,
        train_ratios: List[float],
        max_train_size: int,
        method: Literal["random", "quests"],
        descriptor_k: int,
        descriptor_cutoff: float,
        descriptor_dtype: Literal["float32", "float64"],
        compute_descriptor_chunk_size: int,
        entropy_bandwidth: float,
        entropy_batch_size: int,
        device: Literal["auto", "cpu", "gpu"],
        gpu_device: str | None,
        numba_threads: int,
        write_validation: bool,
        write_extra_tests: bool,
) -> int:
    """Group, split, and export the selected data domains."""

    split_config = SplitConfig(
        root_path=root_path,
        split_repeats=split_repeats,
        seed=seed,
        test_ratio=test_ratio,
        requested_train_ratios=train_ratios,
        train_val_split_method=method,
        max_train_size=max_train_size,
        quests_adapter_config=QuestsAdapterConfig(
            descriptor_k=descriptor_k,
            descriptor_cutoff=descriptor_cutoff,
            descriptor_dtype=descriptor_dtype,
            compute_descriptor_chunk_size=compute_descriptor_chunk_size,
            entropy_bandwidth=entropy_bandwidth,
            entropy_batch_size=entropy_batch_size,
            device=device,
            gpu_device=gpu_device,
            numba_threads=numba_threads,
        )
    )

    valid_domains = domains or [
        metadata.parent.name
        for metadata in Path(root_path).resolve().rglob(DEFAULT_METADATA_FILE)
    ]

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
                write_validation=write_validation,
                write_extra_tests=write_extra_tests,
                all_split_groups=split_groups,
            )
            training_units.extend(training_units_local)
        dumpfn(
            training_units,
            domain_output_path / DEFAULT_TRAINING_UNITS_FILE,
            indent=2,
        )
    return 0
