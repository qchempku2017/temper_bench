"""Utilities to create fine-tuning experiment semantic from data in standard extxyz format."""
from typing import List, Dict, Union
from pathlib import Path
import itertools

from ase import Atoms
from ase.io import read
from monty.serialization import loadfn

from src.temper.env import DEFAULT_DATA_DIR


def load_grouped_data_from_domain(
        domain_name: str,
        data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
) -> List[Dict[str, str | List[Atoms] | None | Dict[str, List[Atoms]]]]:
    """Load data from domain_name folder in data_dir by groups specified in groups.json.

    Args:
        domain_name (str): domain name, name of the folder.
        data_dir (Union[str, Path], optional, default=DEFAULT_DATA_DIR):
            Root directory of the data where domain_name folder is located.

    Returns:
        List[Dict[str, str | List[Atoms] | None | Dict[str, List[Atoms]]]]:
        List of grouped data entries. Each entry is a dictionary with the following keys:
            - strategy (str): grouping strategy.
            - group_name (str): name of the group.
            - atoms_in_group (List[Atoms]): list of atoms in the group.
            - cross_test_groups (None | Dict[str, List[Atoms]]): dictionary of other groups of atoms
                to cross test with.
    """
    root_path = Path(data_dir)
    domain_path = Path(data_dir) / domain_name

    # TODO: managing this in a separate groups.json file is bad. Should go together into info.json.
    group_entries = loadfn(domain_path / "groups.json")

    # Collect all file names and load them.
    all_files = []
    for entry in group_entries:
        all_files.extend(itertools.chain(*entry["groups"].values()))
    all_files = sorted(set(all_files))
    loaded_atoms = {
        file_name: read(root_path / file_name, index=":")
        for file_name in all_files
    }

    def _get_atoms_from_file_names(file_names):
        return [loaded_atoms[file_name] for file_name in file_names]

    # Apply groups.
    grouped_data_entries = []
    for entry in group_entries:
        should_cross_test = entry["add_extra_cross_tests"]
        for group_name, group_files in entry["groups"].items():
            grouped_atoms_entry = {
                "strategy": entry["grouping_strategy"],
                "group_name": group_name,
                "atoms_in_group": _get_atoms_from_file_names(group_files),
            }
            if should_cross_test:
                other_group_names = sorted(set(entry["groups"].keys()) - {group_name})
                grouped_atoms_entry["cross_test_groups"] = {
                    other_group_name: _get_atoms_from_file_names(
                        entry["groups"][other_group_name]
                    )
                    for other_group_name in other_group_names
                }
            else:
                grouped_atoms_entry["cross_test_groups"] = None
            grouped_data_entries.append(grouped_atoms_entry)
    return grouped_data_entries
