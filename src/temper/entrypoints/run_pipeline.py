"""Command-line entry point for preparing and executing MLFF bundles."""
from __future__ import annotations

import argparse
from pathlib import Path


def add_run_pipeline_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the run_pipeline command and its bundle execution options.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Top-level parser's subcommand collection. Registration adds the required
        bundle path and optional --prepare-only flag without loading a bundle
        or executing models. Relative bundle paths are resolved from the
        working directory when the command runs.
    """
    pipeline = subparsers.add_parser("run_pipeline", help="Fine-tune and evaluate a packaged MLFF bundle.")
    pipeline.add_argument("bundle", type=Path)
    pipeline.add_argument("--prepare-only", action="store_true", help="Prepare directories and restore missing configs without executing models.")


def run_pipeline_cli(bundle_path: str | Path, *, prepare_only: bool = False) -> None:
    """Prepare a packaged bundle or run its training and evaluation stages.

    ``bundle_path`` points to bundle.json; relative paths are resolved from the
    current working directory. Paths recorded inside the bundle are resolved
    relative to that file's parent directory.

    With ``prepare_only=True``, create output directories and missing configs
    while preserving existing files, without running models or requiring CUDA.
    Otherwise, prepare the bundle, fine-tune if requested, and evaluate its test
    datasets, writing results under the bundle's configured directories.

    Return None on success; errors propagate to the CLI exit-status handler.

    Parameters
    ----------
    bundle_path : str or Path
        Path to the packaged bundle.json file, resolved as described above.
    prepare_only : bool, optional
        If True, prepare files without running training or evaluation.
        Defaults to False, which executes the complete pipeline.
    """
    from temper.mlff.pipeline import runner

    if prepare_only:
        runner.prepare_bundle(bundle_path)
    else:
        runner.run_pipeline(bundle_path)
