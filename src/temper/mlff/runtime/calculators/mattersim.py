"""Construct the MatterSim ASE Calculator for a written TEMPER bundle."""

from __future__ import annotations


def build_calculator(config):
    """Load a local MatterSim model using the package's native device choice."""
    from mattersim.forcefield import MatterSimCalculator

    return MatterSimCalculator(
        load_path=config["model"],
        **dict(config["calculator"].get("parameters", {})),
    )
