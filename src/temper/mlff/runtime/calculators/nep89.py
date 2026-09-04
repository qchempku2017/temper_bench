"""Construct a calorine NEP Calculator for a written TEMPER bundle."""

from __future__ import annotations

import shutil


def build_calculator(config):
    """Prefer GPUNEP when CUDA and GPUMD exist, otherwise use CPUNEP."""
    from device import cuda_available

    parameters = dict(config["calculator"].get("parameters", {}))
    model = config["model"]
    if cuda_available() and shutil.which("gpumd") is not None:
        from calorine.calculators import GPUNEP

        return GPUNEP(model, **parameters)

    from calorine.calculators import CPUNEP

    return CPUNEP(model, **parameters)
