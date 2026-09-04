"""Construct the SevenNet ASE Calculator for a written TEMPER bundle."""

from __future__ import annotations


def build_calculator(config):
    """Load a local SevenNet model using SevenNet's native auto device."""
    from sevenn.calculator import SevenNetCalculator

    return SevenNetCalculator(
        model=config["model"],
        **dict(config["calculator"].get("parameters", {})),
    )
