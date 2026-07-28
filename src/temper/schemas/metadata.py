from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, ClassVar
from collections import OrderedDict
import json

from ase import Atoms
from ase.io import read

from pydantic import BaseModel, Field, model_validator

from src.temper.env import DEFAULT_METADATA_FILE

def check_atoms_has_stress(frames: Atoms | list[Atoms]) -> bool:
    """Check if the frames have energy, forces, and stress correctly loaded.

      Implemented as checking whether the frames have SinglePointCalculators correctly loaded
    and callable.
    Args:
        frames (list[Atoms]): List of ASE Atoms objects.
    Returns:
        bool: Whether the frames have stress.
    Raises:
        ValueError: If the frames do not have energy or forces.
    """
    if isinstance(frames, Atoms):
        frames = [frames]

    has_stress = True

    for i, atoms in enumerate(frames):
        try:
            atoms.get_potential_energy()
        except Exception as exc:
            raise ValueError(
                f"Frame {i} in extxyz is missing energy. "
                "Please fix the extxyz file."
            ) from exc

        try:
            atoms.get_forces()
        except Exception as exc:
            raise ValueError(
                f"Frame {i} in extxyz is missing forces. "
                "Please fix the extxyz file."
            ) from exc

        try:
            atoms.get_stress()
        except Exception:
            has_stress = False

    if not has_stress:
        warnings.warn(
            "Stress information is missing in one or more frames. "
            "The dataset may not be suitable for stress-dependent benchmarks.",
            UserWarning,
        )

    return has_stress


class MetadataEntry(BaseModel):
    """Metadata entry for a dataset in extxyz file format.

    Attributes:
        name (str): Name of the dataset. Default name would be the name of the extxyz file without extension.
        description (str): Description of the dataset.
        source (str): Source of the dataset (organization, author, or where to download).
        datapath (str): Path to the dataset.
        first_principle_software (str): First principle software used to generate the dataset.
        first_principles_settings (str): First principle settings used to generate the dataset.
        theory_level (str): Theory level used to generate the dataset.
        system_type (list[str]): Type of the system.
        structure_generation_method (list[str]): Method used to generate the structure.
        has_stress (bool): Whether the dataset contains stress.
        has_other_properties (list[str]): Other properties contained in the dataset.
        num_systems (int): Number of systems in the dataset.
        num_frames_per_system (list[int]): Number of frames per system.
        num_atoms_per_system (list[int]): Number of atoms per system.
        formulas (list[str]): Formulas of the systems.
        additional_info (str): Additional information about the dataset.
    """
    # Class variables.
    required_fields: ClassVar[list[str]] = [
        "name",
        "source",
        "datapath",
        "system_type",
    ]
    auto_detected_fields: ClassVar[list[str]] = [
        "num_systems",
        "num_frames_per_system",
        "num_atoms_per_system",
        "formulas",
        "has_stress",
    ]
    optional_fields: ClassVar[list[str]] = [
        "description",
        "has_other_properties",
        "additional_info",
        "theory_level",
        "first_principle_software",
        "first_principles_settings",
        "structure_generation_method",
    ]


    name: str
    description: str = ""
    source: str
    datapath: str

    first_principle_software: str = ""
    first_principles_settings: str = ""
    theory_level: str = ""

    system_type: list[str]
    structure_generation_method: list[str] = Field(default_factory=list)

    has_stress: bool = False
    has_other_properties: list[str] = Field(default_factory=list)

    num_systems: int = 0
    num_frames_per_system: list[int] = Field(default_factory=list)
    num_atoms_per_system: list[int] = Field(default_factory=list)
    formulas: list[str] = Field(default_factory=list)

    additional_info: str = ""

    @model_validator(mode="after")
    def check_required_and_optional_information(self):
        """Throw error if important reproducibility information is missing."""

        missing_required = [
            field
            for field in self.required_fields
            if not getattr(self, field)
        ]

        if missing_required:
            raise ValueError(
                f"Missing required fields to MetaDataEntry: {missing_required}"
            )

        missing_optional = [
            field
            for field in self.optional_fields
            if not getattr(self, field)
        ]

        if missing_optional:
            warnings.warn(
                f"Missing optional fields to MetaDataEntry: {missing_optional}"
            )

        return self

    @classmethod
    def from_extxyz(
        cls,
        extxyz_path: str | Path,
        **kwargs: Any,
    ) -> "MetadataEntry":
        """Constructs metadata automatically from an extxyz file.

        Args:
            extxyz_path:
                Path to extxyz file.

            kwargs:
                User-provided metadata fields, e.g.
                name, source, theory_level, etc.
        """

        extxyz_path = Path(extxyz_path)

        if not extxyz_path.exists():
            raise FileNotFoundError(extxyz_path)

        frames = read(extxyz_path, index=":")

        if len(frames) == 0:
            raise ValueError(
                f"No structures found in {extxyz_path}"
            )

        # Organize into dpdata-like systems.
        #
        # ASE extxyz does not explicitly store MultiSystems information,
        # therefore here we infer systems by atom numbers + composition.
        systems = OrderedDict()

        for atoms in frames:
            formula = atoms.get_chemical_formula(mode="hill")
            key = (
                len(atoms),
                formula,
            )
            if key not in systems:
                systems[key] = []

            systems[key].append(atoms)

        num_systems = len(systems)

        num_frames_per_system = [
            len(v)
            for v in systems.values()
        ]

        num_atoms_per_system = [
            len(v[0])
            for v in systems.values()
        ]

        formulas = [
            k[1]
            for k in systems.keys()
        ]

        # Detect properties
        has_stress = check_atoms_has_stress(frames)

        datapath = f"{extxyz_path.parent.name}/{extxyz_path.name}"
        metadata = dict(
            name=extxyz_path.stem,
            datapath=datapath,
            has_stress=has_stress,
            num_systems=num_systems,
            num_frames_per_system=num_frames_per_system,
            num_atoms_per_system=num_atoms_per_system,
            formulas=formulas,
        )

        # User supplied metadata overrides automatic detection.
        auto_detected_fields = cls.auto_detected_fields
        if any(k in kwargs for k in auto_detected_fields):
            warnings.warn(
                f"User supplied metadata overrides automatic detection of {auto_detected_fields}."
                f" Not recommended!",
                UserWarning,
            )
        metadata.update(kwargs)

        return cls(**metadata)

    def as_dict(self) -> dict[str, Any]:
        """
        Convert to a plain dictionary compatible with monty.serialization.dumpfn.
        """
        return self.model_dump()

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "MetadataEntry":
        """
        Restore from dictionary generated by as_dict().
        """
        return cls.model_validate(data)


def load_metadata_entries_from_datadir(
    datadir: str | Path,
    metadata_file_name: str = DEFAULT_METADATA_FILE
) -> list[MetadataEntry]:
    """
    Load all metadata entries from a data-hosting directory.
    Args:
        datadir (str | Path): path to data-hosting directory. Should contain
            a metadata file named `info_file_name`, as well as structure data
            files in extxyz format.
        metadata_file_name (str): name of metadata file. Defaults to "metadata.json"
            under datadir.
            This file should contain a list of dicts, each corresponding to a
            structure data file under the data-hosting directory. And each dict must
            contain required keys by MetadataEntry, including `name`, `source`
            and `system_type`. `datapath` not required as it will be automatically
            detected when intializing metadata entry from data file.
            Other optinal fields can be found in MetadataEntry.optinal_fields.
            Should not contain any field in MetadataEntry.auto_detected_fields as
            they are supposed to be automatically detected.
    Returns:
        MetadataEntry: metadata entry.
    """
    datadir = Path(datadir)
    info_path = datadir / metadata_file_name

    with open(info_path, "r") as f:
        info = json.load(f)["info"]

    datafiles = list(datadir.glob("*.extxyz"))
    if len(datafiles) != len(info):
        raise ValueError(
            f"Number of data files ({len(datafiles)})"
            f" does not match row of entries in {info_path} ({len(info)})."
        )

    # Keep only required and optional fields in info.
    fields = set(MetadataEntry.required_fields + MetadataEntry.optional_fields) - {"datapath"}
    info = [
        {k: v for k, v in entry.items() if k in fields}
        for entry in info
    ]

    return [
        MetadataEntry.from_extxyz(datafile, **entry)
        for datafile, entry in zip(datafiles, info)
    ]
