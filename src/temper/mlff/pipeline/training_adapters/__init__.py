"""Dispatch supported MLFF families to installed runtime adapters."""

from __future__ import annotations

from temper.mlff.pipeline.training_adapters.deepmd import (
    DPA4Adapter,
    DPA4CAdapter,
)
from temper.mlff.pipeline.training_adapters.mace import MACEAdapter
from temper.mlff.pipeline.training_adapters.mattersim import MatterSimAdapter
from temper.mlff.pipeline.training_adapters.nep89 import NEP89Adapter
from temper.mlff.pipeline.training_adapters.sevennet import SevenNetAdapter

from temper.mlff.pipeline.training_adapters.base import BaseMLFFAdapter


def mlff_adapter_factory(mlff_type: str) -> type[BaseMLFFAdapter]:
    """Return the adapter class registered for a supported MLFF family."""
    try:
        return BaseMLFFAdapter.registry[mlff_type]
    except KeyError as error:
        raise ValueError(f"Unsupported MLFF type {mlff_type!r}.") from error
