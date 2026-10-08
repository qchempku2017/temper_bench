"""Inspect extxyz headers and species without constructing ASE structures."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ase.data import chemical_symbols
from ase.io.extxyz import REV_PROPERTY_NAME_MAP, key_val_str_to_dict
from numpy import atleast_1d


@dataclass
class ExtxyzMetadata:
    """Metadata for one extxyz frame.

    Attributes
    ----------
    number_of_atoms : int
        Number of atom rows in the frame.
    info : dict[str, Any]
        Parsed header values, excluding the Properties column declaration.
    properties : set[str]
        Per-atom property names, using ASE names for standard columns.
        Frame properties such as energy, stress and virial remain in info.
        Values and physical units are not validated.
    symbols : tuple[str, ...]
        Chemical symbols in atom order, or an empty tuple when not requested.
    """

    number_of_atoms: int
    info: dict[str, Any]
    properties: set[str]
    symbols: tuple[str, ...]

    @property
    def fully_periodic(self) -> bool:
        """Return periodicity using the extxyz lattice-dependent default."""
        return bool(all(atleast_1d(self.info.get("pbc", "Lattice" in self.info))))


def iter_extxyz_metadata(
    path: str | Path, *, read_symbols: bool = False,
) -> Iterator[ExtxyzMetadata]:
    """Stream frame metadata while skipping numerical atom columns.

    Parameters
    ----------
    path : str or Path
        Source extxyz file.
    read_symbols : bool, optional
        Read only the species or atomic-number column from each atom row.
        Otherwise atom rows are skipped without splitting them.

    Yields
    ------
    ExtxyzMetadata
        One frame's header metadata and optionally its species.

    Raises
    ------
    ValueError
        If an atom block is truncated or requested species are absent.
    """
    with Path(path).open(encoding="utf-8") as stream:
        while line := stream.readline():
            if not line.strip():
                continue
            count = int(line)
            info = key_val_str_to_dict(stream.readline())
            declaration = info.pop("Properties", "species:S:1:pos:R:3").split(":")
            columns = {}
            offset = 0
            for name, _, width in zip(
                declaration[::3], declaration[1::3], declaration[2::3], strict=True,
            ):
                columns[name] = offset
                offset += int(width)
            symbols = []
            for _ in range(count):
                row = stream.readline()
                if not row:
                    raise ValueError(f"Truncated atom block in {path}.")
                if read_symbols:
                    values = row.split()
                    if "species" in columns:
                        symbols.append(values[columns["species"]])
                    elif "Z" in columns:
                        symbols.append(chemical_symbols[int(values[columns["Z"]])])
                    else:
                        raise ValueError(f"No species or Z column in {path}.")
            properties = {REV_PROPERTY_NAME_MAP.get(name, name) for name in columns}
            yield ExtxyzMetadata(count, info, properties, tuple(symbols))


def check_extxyz_properties(frame: ExtxyzMetadata, *, source: str) -> bool:
    """Require energy and forces and report the presence of stress.

    Parameters
    ----------
    frame : ExtxyzMetadata
        Metadata returned by iter_extxyz_metadata.
    source : str
        Filename and frame index to include in missing-label errors.

    Returns
    -------
    bool
        Whether stress is declared. Virial remains available in info,
        but is not treated as stress because ASE does not convert it on read.

    Raises
    ------
    ValueError
        If the header lacks energy or the Properties declaration lacks forces.
    """
    for label, names in (("energy", frame.info), ("forces", frame.properties)):
        if label not in names:
            raise ValueError(f"{source} is missing {label}.")
    return "stress" in frame.info
