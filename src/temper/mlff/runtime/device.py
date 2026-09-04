#!/usr/bin/env python3
"""Resolve hardware only after a submit bundle reaches its execution host."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys


def _cuda_hidden() -> bool:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    return value is not None and value.strip() in {"", "-1"}


def torch_device(*, include_mps: bool = False) -> str:
    """Choose CUDA, optionally MPS, or CPU through the installed PyTorch."""
    import torch

    if not _cuda_hidden() and torch.cuda.is_available():
        return "cuda"
    if (
        include_mps
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"


def cuda_available() -> bool:
    """Report whether the remote process can see an NVIDIA CUDA device."""
    if _cuda_hidden():
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


def main() -> None:
    """Resolve a requested runtime device for use by the generated run script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("torch", "require-cuda"))
    parser.add_argument("--mps", action="store_true")
    parser.add_argument("--warn-mattersim", action="store_true")
    arguments = parser.parse_args()

    if arguments.mode == "require-cuda":
        if not cuda_available():
            raise RuntimeError(
                "NEP-89 fine-tuning requires a visible NVIDIA CUDA GPU on the "
                "remote runner."
            )
        return

    device = torch_device(include_mps=arguments.mps)
    if device == "cuda" and arguments.warn_mattersim:
        print(
            "Warning: MatterSim 1.2.5 has known CUDA fine-tuning issues; "
            "batch_size remains fixed at 1.",
            file=sys.stderr,
        )
    print(device)


if __name__ == "__main__":
    main()
