"""Construct the DeepMD ASE Calculator for a written TEMPER bundle."""

from __future__ import annotations

import os


def build_calculator(config):
    """Load a local DeepMD model and require CUDA for its runtime device."""
    from check_cuda import torch_device
    os.environ["DEVICE"] = torch_device()
    from deepmd.calculator import DP

    return DP(
        model=config["model"],
        **dict(config["calculator"].get("parameters", {})),
    )
