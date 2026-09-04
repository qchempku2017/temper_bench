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


_WRITERS = {
    "dpa4": DPA4BundleWriter,
    "dpa4c": DPA4CBundleWriter,
    "mattersim": MatterSimBundleWriter,
    "mace": MACEBundleWriter,
    "sevennet": SevenNetBundleWriter,
    "nep89": NEP89BundleWriter,
}


def _write_submit_folder(
    bundle: MLFFTrainBundle,
    target_dir: str | Path | None,
) -> Path:
    """Select the concrete writer and create one local submit folder."""
    try:
        writer = _WRITERS[bundle.mlff_spec.mlff_type]
    except KeyError as error:
        raise ValueError(
            f"Unsupported MLFF type {bundle.mlff_spec.mlff_type!r}; expected one "
            f"of {sorted(_WRITERS)!r}."
        ) from error
    return writer(bundle).write_submit_folder(target_dir)
