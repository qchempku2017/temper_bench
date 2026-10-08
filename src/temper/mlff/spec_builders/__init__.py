"""Concrete builders for TEMPER's six supported MLFF families."""

from temper.mlff.spec_builders.deepmd import DPA4CSpecBuilder, DPA4SpecBuilder
from temper.mlff.spec_builders.mace import MACESpecBuilder
from temper.mlff.spec_builders.mattersim import MatterSimSpecBuilder
from temper.mlff.spec_builders.nep89 import NEP89SpecBuilder
from temper.mlff.spec_builders.sevennet import SevenNetSpecBuilder


from temper.mlff.spec_builders.base import BaseSpecBuilder


def mlff_spec_builder_factory(mlff_type: str) -> type[BaseSpecBuilder]:
    """Return the registered builder class for an MLFF family key."""
    return BaseSpecBuilder.registry[mlff_type]

__all__ = [
    "mlff_spec_builder_factory",
    "DPA4CSpecBuilder",
    "DPA4SpecBuilder",
    "MACESpecBuilder",
    "MatterSimSpecBuilder",
    "NEP89SpecBuilder",
    "SevenNetSpecBuilder",
]
