"""Construct the GPU calorine NEP Calculator."""
from __future__ import annotations

import shutil


def build_calculator(config):
    """Return GPUNEP, requiring both CUDA and the gpumd executable."""
    from temper.mlff.pipeline.cuda_utils import require_cuda
    from calorine.calculators import GPUNEP

    require_cuda()
    if shutil.which("gpumd") is None:
        raise RuntimeError("NEP evaluation requires the gpumd executable.")
    return GPUNEP(config["model"], **dict(config["calculator"].get("parameters", {})))
