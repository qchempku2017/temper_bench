#!/usr/bin/env python3
"""Require a visible CUDA GPU on the execution host."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import warnings



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


def main() -> None:
    """Check the generated runner's GPU requirement before executing an MLFF."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--warn-mattersim", action="store_true")
    arguments = parser.parse_args()
    require_cuda()
    if arguments.warn_mattersim:
        warnings.warn(
            "MatterSim 1.2.5 has unresolved CUDA fine-tuning and batch-index "
            "issues (microsoft/mattersim#163). See docs/mlff-bundles.md.",
            RuntimeWarning,
        )


if __name__ == "__main__":
    main()
