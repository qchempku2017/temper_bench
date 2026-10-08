"""Construct the MatterSim ASE Calculator for a written TEMPER bundle."""

from __future__ import annotations


def build_calculator(config):
    """Load a local MatterSim model on CUDA."""
    from check_cuda import torch_device
    from mattersim.forcefield import MatterSimCalculator

    return MatterSimCalculator(
        load_path=config["model"],
        device=torch_device(),
        **dict(config["calculator"].get("parameters", {})),
    )
