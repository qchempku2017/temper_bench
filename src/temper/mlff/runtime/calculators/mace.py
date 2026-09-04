"""Construct the MACE ASE Calculator for a written TEMPER bundle."""

from __future__ import annotations


def build_calculator(config):
    """Load a local MACE model after resolving CUDA, MPS, or CPU remotely."""
    from device import torch_device
    from mace.calculators import MACECalculator

    parameters = dict(config["calculator"].get("parameters", {}))
    parameters["device"] = torch_device(include_mps=True)
    return MACECalculator(model_paths=[config["model"]], **parameters)
