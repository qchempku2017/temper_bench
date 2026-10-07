"""Concrete builders for TEMPER's six supported MLFF families."""

from temper.mlff.spec_builders.deepmd import DPA4CSpecBuilder, DPA4SpecBuilder
from temper.mlff.spec_builders.mace import MACESpecBuilder
from temper.mlff.spec_builders.mattersim import MatterSimSpecBuilder
from temper.mlff.spec_builders.nep89 import NEP89SpecBuilder
from temper.mlff.spec_builders.sevennet import SevenNetSpecBuilder


# Comment: as I implied in the bundle_writers/__init__.py, you may consider using a registry mechanism to manage the spec builders, and leave a subclass factory here.

__all__ = [
    "DPA4CSpecBuilder",
    "DPA4SpecBuilder",
    "MACESpecBuilder",
    "MatterSimSpecBuilder",
    "NEP89SpecBuilder",
    "SevenNetSpecBuilder",
]
