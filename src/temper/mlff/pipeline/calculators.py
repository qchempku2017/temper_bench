"""Construct ASE calculators for supported MLFF families.

Implement new calculators by adding a new family identifier and import block.
"""

import os
import shutil


def build_calculator(config):
    """Construct the selected ASE calculator without importing other backends.

    config supplies the family identifier, an absolute model path, and the
    calculator parameters. CUDA is required; NEP also requires gpumd on PATH.
    """
    from temper.mlff.pipeline.cuda_utils import torch_device, require_cuda

    family = config["calculator"]["identifier"]
    parameters = dict(config["calculator"].get("parameters", {}))
    model = config["model"]
    if family in {"dpa4", "dpa4c"}:
        os.environ["DEVICE"] = torch_device()
        from deepmd.calculator import DP
        return DP(model=model, **parameters)
    if family == "mace":
        from mace.calculators import MACECalculator
        parameters["device"] = torch_device()
        return MACECalculator(model_paths=[model], **parameters)
    if family == "mattersim":
        from mattersim.forcefield import MatterSimCalculator
        parameters["device"] = torch_device()
        return MatterSimCalculator(load_path=model, **parameters)
    if family == "sevennet":
        from sevenn.calculator import SevenNetCalculator
        parameters["device"] = torch_device()
        return SevenNetCalculator(model=model, **parameters)
    if family == "nep89":
        require_cuda()
        if shutil.which("gpumd") is None:
            raise RuntimeError("NEP evaluation requires the gpumd executable.")
        from calorine.calculators import GPUNEP
        return GPUNEP(model, **parameters)
    raise ValueError(f"Unsupported MLFF type {family!r}.")
