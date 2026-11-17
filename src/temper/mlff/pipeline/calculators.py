"""Construct ASE calculators for supported MLFF families.

Implement new calculators by adding a new family identifier and import
block.
"""

from __future__ import annotations

import os
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator


def build_calculator(
    identifier: str, model: str | Path, parameters: dict[str, Any],
) -> Calculator:
    """Construct the selected ASE calculator without importing other
    backends.

    Parameters
    ----------
    identifier : str
        Model family: dpa4, dpa4c, mace, mattersim, sevennet or nep89.
    model : str or pathlib.Path
        Checkpoint or exported-model path. Relative paths use the current
        working directory; the bundle runner supplies an absolute path.
    parameters : dict[str, Any]
        Backend-specific calculator keyword arguments from MLFFSpec.
        Values may include nested backend settings. A deep copy is passed
        to the backend so constructing a calculator cannot modify the
        stored specification. Device selection follows the rules below.

    Returns
    -------
    ase.calculators.calculator.Calculator
        Initialized ASE calculator for energies in eV, forces in eV/angstrom
        and stresses in eV/angstrom cubed. Construction may load model weights
        and allocate GPU memory; run this inside the scheduled allocation.

    Raises
    ------
    RuntimeError
        If CUDA is unavailable or NEP cannot find ``gpumd`` on PATH.
    ValueError
        If the family identifier is unsupported.
    ImportError
        If the selected backend is not installed.

    Notes
    -----
    GPU selection follows the scheduler's visible devices. Explicit NEP
    calculator overrides are retained; by default calorine leaves
    CUDA_VISIBLE_DEVICES unchanged. DeepMD also receives the selected device
    through its DEVICE environment variable.
    """
    from temper.utils.cuda import torch_device, require_cuda

    family = identifier
    parameters = deepcopy(parameters)
    model = str(model)
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

        parameters.setdefault("gpu_identifier_index", None)
        return GPUNEP(model, **parameters)
    raise ValueError(f"Unsupported MLFF type {family!r}.")
