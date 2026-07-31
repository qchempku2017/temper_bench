from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

from ase import Atoms
from ase.io import read
from monty.serialization import loadfn
from pydantic import BaseModel, field_validator

from src.temper.utils.env import DEFAULT_DATA_DIR, DEFAULT_METADATA_FILE
from src.temper.utils.grouping import GROUPING_STRATEGIES


class GroupEntry(BaseModel):
    """Definition of a single grouping strategy in groups.json.

    Each GroupEntry corresponds to one dictionary entry:
    {
        "grouping_strategy": "...",
        "groups": [...],
        "add_extra_cross_tests": true/false
    }

    Attributes:
        grouping_strategy (str): The name of the grouping strategy.
        groups (Dict[str, List[str]]): The groups of structure data files.
        add_extra_cross_tests (bool): Whether to add extra cross tests.
          If true, beyond testing within each group, data from other groups will also be used for testing,
          and testing results will be reported separately for each group.
    """

    grouping_strategy: str

    groups: Dict[str, List[str]]

    add_extra_cross_tests: bool = False

    @field_validator("groups")
    @classmethod
    def validate_groups(
        cls,
        value: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:

        if not isinstance(value, Dict):
            raise TypeError(
                "groups must be a dict mapping from string to list of lists."
            )

        for name, group in value.items():
            if not isinstance(name, str):
                raise TypeError(
                    f"Group name: {name} must be a string."
                )

            if not isinstance(group, List):
                raise TypeError(
                    f"groups[{name}] must be a list."
                )

            for filename in group:
                if not isinstance(filename, str):
                    raise TypeError(
                        f"groups[{name}] contains non-string value: "
                        f"{filename}"
                    )
                filepath = Path(filename)
                if filepath.suffix != ".extxyz":
                    raise ValueError(
                        f"groups[{name}] contains invalid file extension: "
                        f"{filepath.suffix}."
                        f" Only .extxyz files are supported."
                    )

        return value

    def as_dict(self) -> Dict[str, Any]:
        """
        Convert to a plain dictionary.

        Compatible with monty.serialization.dumpfn.
        """
        return self.model_dump()

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
    ) -> "GroupEntry":
        """
        Construct from dictionary.
        """
        return cls.model_validate(data)

    @classmethod
    def from_datadir_with_strategy(
            cls,
            datadir: str | Path,
            grouping_strategy: str,
            add_extra_cross_tests: bool = False,
            **grouping_strategy_kwargs
    ) -> "GroupEntry":
        """Create GroupEntry from a directory of structure data files.

        Parameters
        -----------
        datadir : str | Path
            The directory containing the structure data files.
        grouping_strategy : str
            The name of the grouping strategy.
        add_extra_cross_tests : bool, optional
            Whether to add extra cross tests. Defaults to False.
        **grouping_strategy_kwargs
            Additional keyword arguments for the grouping strategy function.
            See grouping.py for details.

        Returns
        --------
        GroupEntry

        Raises
        --------
        ValueError
           If the grouping strategy is not recognized.
        """
        if grouping_strategy not in GROUPING_STRATEGIES:
            raise ValueError(
                f"Unknown grouping strategy: {grouping_strategy}. "
                f"Available strategies: {GROUPING_STRATEGIES}"
            )
        files = list(Path(datadir).glob("*.extxyz"))
        groups = GROUPING_STRATEGIES[grouping_strategy](
            files, **grouping_strategy_kwargs
        )

        return cls(
            grouping_strategy=grouping_strategy,
            groups=groups,
            add_extra_cross_tests=add_extra_cross_tests,
        )

    def load_atoms_in_group(self) -> Dict[str, List[Atoms]]:
        """Load the atoms in each group.

        Returns
        --------
        Dict[str, List[Atoms]]
            A dictionary mapping group names to lists of ASE Atoms objects.
        """
        atoms_groups = {}
        for group_name, group in self.groups.items():
            atoms_groups[group_name] = []
            for i, filename in enumerate(group):
                atoms = read(filename, index=":")
                atoms_groups[group_name].extend(atoms)
        return atoms_groups


def load_grouped_data_from_domain(
        domain_name: str,
        data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
        metadata_file: str = DEFAULT_METADATA_FILE,
) -> Dict[str, Dict[str, List[Atoms]]]:
    """Load data from domain_name folder in data_dir by groups specified in groups.json.

    Parameters
    ----------
    domain_name: str
        Name of the domain to load data from.
    data_dir: str | Path
        The path to the directory containing the data.
    metadata_file: str
        Name of the metadata file. Default is DEFAULT_METADATA_FILE.

    Returns
    -------
    Dict[str, Dict[str, List[Atoms]]]
        Dictionary mapping grouping strategies to grouped atoms, each group
         stored as a dictionary mapping group names to lists of atoms.
    """
    domain_path = Path(data_dir) / domain_name

    metadata = loadfn(domain_path / metadata_file)
    grouping_strategies_and_kwargs = metadata["groupings"]

    # Build group entries.
    group_entries: Dict[str, GroupEntry] = {
        kwargs["grouping_strategy"]: GroupEntry.from_datadir_with_strategy(
            domain_path,
            **kwargs
        )
        for kwargs in grouping_strategies_and_kwargs
    }

    # Load atoms.
    grouped_atoms: Dict[str, dict[str, list[Atoms]]] = {
        key : val.load_atoms_in_group(domain_path)
        for key, val in group_entries
    }

    return grouped_atoms
