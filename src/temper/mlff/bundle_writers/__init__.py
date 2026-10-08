"""Dispatch the six supported MLFF families to fixed bundle writers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from temper.mlff.bundle_writers.deepmd import (
    DPA4BundleWriter,
    DPA4CBundleWriter,
)
from temper.mlff.bundle_writers.mace import MACEBundleWriter
from temper.mlff.bundle_writers.mattersim import MatterSimBundleWriter
from temper.mlff.bundle_writers.nep89 import NEP89BundleWriter
from temper.mlff.bundle_writers.sevennet import SevenNetBundleWriter

if TYPE_CHECKING:
    from temper.schemas.mlff_train_bundle import MLFFTrainBundle


from temper.mlff.bundle_writers.base import BaseMLFFBundleWriter


def mlff_bundle_writer_factory(mlff_type: str) -> type[BaseMLFFBundleWriter]:
    """Return the writer class registered for a supported MLFF family."""
    try:
        return BaseMLFFBundleWriter.registry[mlff_type]
    except KeyError as error:
        raise ValueError(f"Unsupported MLFF type {mlff_type!r}.") from error


def write_submit_folder(
    bundle: MLFFTrainBundle,
    target_dir: str | Path | None = None,
) -> Path:
    """Write and return a complete submit folder for one dataset/recipe pair.

    bundle supplies the source data and model artifacts. target_dir must not
    exist; omitting it creates a caller-owned temporary directory. Files are
    copied and the folder is removed if writing fails.
    """
    writer = mlff_bundle_writer_factory(bundle.mlff_spec.mlff_type)
    return writer(bundle).write_submit_folder(target_dir)
