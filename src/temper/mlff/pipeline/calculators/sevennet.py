"""Construct the SevenNet ASE Calculator for a written TEMPER bundle."""

from __future__ import annotations


def build_calculator(config):
    """Load a local SevenNet model on CUDA."""
    from temper.mlff.pipeline.cuda_utils import torch_device
    from sevenn.calculator import SevenNetCalculator

    return SevenNetCalculator(
        model=config["model"],
        device=torch_device(),
        **dict(config["calculator"].get("parameters", {})),
    )
