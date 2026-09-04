#!/usr/bin/env python3
"""Convert one TEMPER extxyz dataset to DeepMD NumPy systems."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    """Parse paths and convert one extxyz file to DeepMD NumPy systems."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    from dpdata import MultiSystems

    source = Path(arguments.input)
    destination = Path(arguments.output)
    destination.mkdir(parents=True, exist_ok=False)
    systems = MultiSystems.from_file(str(source), fmt="extxyz")
    systems.to("deepmd/npy", str(destination))


if __name__ == "__main__":
    main()
