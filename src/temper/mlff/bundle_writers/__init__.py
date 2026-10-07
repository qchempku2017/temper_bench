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


# Comment: These subclass may instead take a @register decorator to manage. You may refer to dpdata's registry mechanism.
#  Subclasses are registered into the base class's registry dict, and can be retrieved by name.
#  You can leave a handy subclass factory function here for convenience of retrieving the correct subclass by name.
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
    # Comment: Make this method public as it will be used elsewhere. Meanwhile,
    #  improve the docstring to explain what it does and what the parameters are.
    try:
        writer = _WRITERS[bundle.mlff_spec.mlff_type]
    except KeyError as error:
        raise ValueError(
            f"Unsupported MLFF type {bundle.mlff_spec.mlff_type!r}; expected one "
            f"of {sorted(_WRITERS)!r}."
        ) from error
    return writer(bundle).write_submit_folder(target_dir)
