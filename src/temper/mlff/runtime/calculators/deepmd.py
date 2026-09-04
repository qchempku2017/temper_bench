"""Construct the DeepMD ASE Calculator for a written TEMPER bundle."""

from __future__ import annotations


def build_calculator(config):
    """Load a local DeepMD model and let DeepMD choose its runtime device."""
    from deepmd.calculator import DP

    return DP(
        model=config["model"],
        **dict(config["calculator"].get("parameters", {})),
    )
