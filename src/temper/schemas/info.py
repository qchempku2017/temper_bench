from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, ClassVar, List, Dict
from collections import OrderedDict

from ase.io import read

from pydantic import Field, model_validator
from monty.serialization import loadfn

from src.temper.schemas.base import JsonIOModel
from src.temper.utils.defaults import DEFAULT_METADATA_FILE
from src.temper.schemas.utils import check_atoms_has_stress, check_atoms_have_other_properties


class InfoEntry(JsonIOModel):
    """Info entry for a data contained in a single extxyz file.

    Corresponds to the `info` section in the `metadata.json` for each data domain.

    This API is not recommended for public exposure, and shall instead be called by
    GroupedDomain.

    Attributes:
        name (str): Name of the dataset. Default name would be the name of the extxyz file without extension.
        description (str): Description of the dataset.
        source (str): Source of the dataset (organization, author, or where to download).
        domain (str): Domain of the dataset. Must be the same as the name of the folder containing the dataset.
        filename (str): File name of the dataset.
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
    required_fields: ClassVar[List[str]] = [
        "name",
        "source",
        "domain",  # Required, but often inferred from the folder name rather than provided.
        "filename",
        "system_type",
    ]
    auto_detected_fields: ClassVar[List[str]] = [
        "num_systems",
        "num_frames_per_system",
        "num_atoms_per_system",
        "formulas",
        "has_stress",
        "has_other_properties",
    ]
    optional_fields: ClassVar[List[str]] = [
        "description",
        "additional_info",
        "theory_level",
        "first_principle_software",
        "first_principles_settings",
        "structure_generation_method",
    ]


    name: str
    description: str = ""
    source: str
    domain: str
    filename: str

    first_principle_software: str = ""
    first_principles_settings: str = ""
    theory_level: str = ""

    system_type: List[str]
    structure_generation_method: List[str] = Field(default_factory=list)

    has_stress: bool = False
    has_other_properties: List[str] = Field(default_factory=list)

    num_systems: int = 0
    num_frames_per_system: List[int] = Field(default_factory=list)
    num_atoms_per_system: List[int] = Field(default_factory=list)
    formulas: List[str] = Field(default_factory=list)

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

    # TODO: Efficiency issue in current design: extxyz files loaded twice, once here, once
    #  in ReferenceResolver. May improve in the future. (not urgent)
    @classmethod
    def from_extxyz(
        cls,
        extxyz_path: str | Path,
        **kwargs: Any,
    ) -> "InfoEntry":
        """Constructs metadata automatically from an extxyz file.

        Parameters
        ----------
        extxyz_path : str | Path
            Path to the extxyz file.
        **kwargs : Any
            Additional keyword arguments to pass to the constructor.
            Including necessary and optional fields beyond automatically detected ones.
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
        has_other_properties = check_atoms_have_other_properties(frames)

        metadata = dict(
            name=extxyz_path.stem,  # Required, but can be inferred from file name.
            domain=extxyz_path.parent.name,  # Required, but can be inferred from file path.
            filename=extxyz_path.name,  # Required, but can be inferred from file path.
            has_stress=has_stress,
            has_other_properties=has_other_properties,
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


def load_info_entries_from_datadir(
    datadir: str | Path,
    metadata_file_name: str = DEFAULT_METADATA_FILE
) -> List[InfoEntry]:
    """
    Load all metadata entries from a data-hosting directory.

    Parameters
    --------
    datadir : str | Path
        The path to data-hosting directory. Should contain a metadata file named
        `info_file_name`, as well as structure data files in extxyz format.
    metadata_file_name : str, optional
        Name of metadata file. Defaults to "metadata.json" under datadir.
        Must contain `info` as a top-level key.
        For file structure, see README.md for details.

    Returns
    -------
    List[InfoEntry]
        List of InfoEntry objects.

    Raises
    ------
    ValueError
        If number of data files does not match number of entries in metadata file.
    """
    datadir = Path(datadir)
    info_path = datadir / metadata_file_name

    info = loadfn(info_path)["info"]

    datafiles = list(datadir.glob("*.extxyz"))
    if len(datafiles) != len(info):
        raise ValueError(
            f"Number of data files ({len(datafiles)})"
            f" does not match row of entries in {info_path} ({len(info)})."
        )
    # Sort datafiles to match the order of entries in info.
    filenames_in_info = [entry["filename"] for entry in info]
    datafiles = sorted(datafiles, key=lambda x: filenames_in_info.index(x.name))

    # Keep only required and optional fields in info.
    fields = set(InfoEntry.required_fields + InfoEntry.optional_fields) - {"filename"}
    info = [
        {k: v for k, v in entry.items() if k in fields}
        for entry in info
    ]

    return [
        InfoEntry.from_extxyz(datafile, **entry)
        for datafile, entry in zip(datafiles, info)
    ]
