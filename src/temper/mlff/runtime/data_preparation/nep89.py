#!/usr/bin/env python3
"""Normalize labeled TEMPER extxyz files for TorchNEP training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import iread, write


def convert(
    source: Path,
    destination: Path,
    *,
    include_stress: bool,
) -> None:
    """Convert one labeled extxyz dataset to TorchNEP's training representation.

    Parameters
    ----------
    source : pathlib.Path
        Input extxyz file containing energy and force labels.
    destination : pathlib.Path
        TorchNEP-compatible extxyz file to write.
    include_stress : bool
        Whether to convert ASE stress labels to GPUMD virials.

    Raises
    ------
    ValueError
        If a structure is nonperiodic or the input contains no frames.
    """
    frames = []
    for frame_index, atoms in enumerate(iread(source, index=":")):
        if not np.all(atoms.pbc):
            raise ValueError(
                f"TorchNEP requires full periodicity; frame {frame_index} in "
                f"{source} is not periodic in every direction."
            )
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(apply_constraint=False), dtype=float)
        converted = atoms.copy()
        converted.calc = SinglePointCalculator(
            converted,
            energy=energy,
            forces=forces,
        )
        if include_stress:
            stress = np.asarray(
                atoms.get_stress(
                    voigt=False,
                    apply_constraint=False,
                    include_ideal_gas=False,
                ),
                dtype=float,
            )
            converted.info["virial"] = -stress * converted.get_volume()
        frames.append(converted)
    if not frames:
        raise ValueError(f"Cannot prepare empty TorchNEP dataset: {source}.")
    write(destination, frames, format="extxyz")


def main() -> None:
    """Parse submit-runner arguments and prepare train and validation data."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--stress", action="store_true")
    arguments = parser.parse_args()
    output = Path(arguments.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    convert(
        Path(arguments.train),
        output / "train.xyz",
        include_stress=arguments.stress,
    )
    convert(
        Path(arguments.validation),
        output / "test.xyz",
        include_stress=arguments.stress,
    )


if __name__ == "__main__":
    main()
