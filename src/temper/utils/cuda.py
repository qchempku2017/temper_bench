"""Require a visible CUDA GPU on the execution host."""

from __future__ import annotations

import os
import shutil
import subprocess


def cuda_available() -> bool:
    """Check whether the execution host reports available CUDA devices.

    Returns
    -------
    bool
        False when CUDA_VISIBLE_DEVICES is empty or '-1'. Otherwise, use
        torch.cuda.is_available() when PyTorch is installed. If PyTorch is
        unavailable, return True only when nvidia-smi -L succeeds and reports
        at least one device; return False when that executable is absent.

    Raises
    ------
    OSError
        If the nvidia-smi fallback is found but cannot be executed.

    Notes
    -----
    PyTorch is imported only when this function is called. The nvidia-smi
    fallback checks the host driver, not compatibility with a particular
    Python model package.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES", "visible").strip() in {"", "-1"}:
        return False
    try:
        import torch
    except ImportError:
        executable = shutil.which("nvidia-smi")
        if executable is None:
            return False
        result = subprocess.run(
            [executable, "-L"],
            capture_output=True,
            check=False,
            text=True,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    return bool(torch.cuda.is_available())


def require_cuda() -> None:
    """Require CUDA availability before starting a GPU calculation.

    Uses cuda_available, including its nvidia-smi fallback when PyTorch is
    absent. Returns None when the check succeeds.

    Raises
    ------
    RuntimeError
        If CUDA is disabled or no available CUDA device is reported.
    OSError
        If the nvidia-smi fallback cannot be executed.
    """
    if not cuda_available():
        raise RuntimeError(
            "MLFF execution requires a visible NVIDIA CUDA GPU."
        )


def torch_device() -> str:
    """Return the CUDA device name after checking the PyTorch runtime.

    Returns
    -------
    str
        'cuda', referring to PyTorch's default visible CUDA device. No CPU or
        MPS fallback is selected, and CUDA_VISIBLE_DEVICES is not modified.

    Raises
    ------
    ImportError
        If PyTorch is not installed. It is imported only when called.
    RuntimeError
        If CUDA_VISIBLE_DEVICES is empty or '-1', or PyTorch reports that
        CUDA is unavailable.
    """
    import torch

    if (
        os.environ.get("CUDA_VISIBLE_DEVICES", "visible").strip() in {"", "-1"}
        or not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "MLFF execution requires a visible NVIDIA CUDA GPU."
        )
    return "cuda"
