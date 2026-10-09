"""Convert one TEMPER extxyz dataset to DeepMD NumPy systems."""

from __future__ import annotations

from pathlib import Path


def prepare_dataset(source: str | Path, destination: str | Path) -> None:
    """Convert labeled extxyz structures into DeepMD NumPy system directories.

    Parameters
    ----------
    source : str or Path
        Input extxyz dataset, read through dpdata's ASE structure reader.
    destination : str or Path
        New directory in which to write the converted systems.

    Returns
    -------
    None
        Writes files for the native DeepMD training command.
    """
    from dpdata import MultiSystems

    source = Path(source)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    # Dpdata does not read extxyz. Must specify ase/structure.
    # File first converted into ase Atoms, then into MultiSystems.
    systems = MultiSystems.from_file(str(source), fmt="ase/structure")
    systems.to("deepmd/npy", str(destination))
