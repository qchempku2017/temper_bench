"""Command-line entry point for preparing and executing MLFF bundles."""
from __future__ import annotations

from pathlib import Path

from temper.mlff.pipeline import runner


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
    """
    if prepare_only:
        runner.prepare_bundle(bundle_path)
    else:
        runner.run_pipeline(bundle_path)
