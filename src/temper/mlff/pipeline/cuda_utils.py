"""Require a visible CUDA GPU on the execution host."""
from __future__ import annotations

import os
import shutil
import subprocess


def cuda_available() -> bool:
    """Return whether CUDA is visible, respecting scheduler device restrictions."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", "visible").strip() in {"", "-1"}:
        return False
    try:
        import torch
    except ImportError:
        executable = shutil.which("nvidia-smi")
        if executable is None:
            return False
        result = subprocess.run(
            [executable, "-L"], capture_output=True, check=False, text=True,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    return bool(torch.cuda.is_available())


def require_cuda() -> None:
    """Raise RuntimeError if the runner cannot use a CUDA GPU."""
    if not cuda_available():
        raise RuntimeError("MLFF execution requires a visible NVIDIA CUDA GPU.")


def torch_device() -> str:
    """Return cuda after checking the installed PyTorch runtime.

    Raises RuntimeError if the runner cannot use a CUDA GPU.
    """
    import torch

    if (
        os.environ.get("CUDA_VISIBLE_DEVICES", "visible").strip() in {"", "-1"}
        or not torch.cuda.is_available()
    ):
        raise RuntimeError("MLFF execution requires a visible NVIDIA CUDA GPU.")
    return "cuda"
