"""Loads a domain metadata and builds grouped-domain objects using its configured grouping strategies."""


from __future__ import annotations

from pathlib import Path
from typing import Union, List

from monty.serialization import loadfn

from src.temper.schemas.group import GroupedDomain
from src.temper.utils.defaults import DEFAULT_DATA_DIR, DEFAULT_METADATA_FILE


def partition_domain_into_groups(
        domain_name: str,
        root_path: Union[str, Path] = DEFAULT_DATA_DIR,
        metadata_file_name: str = DEFAULT_METADATA_FILE,
) -> List[GroupedDomain]:
    """Load data from a domain folder and partition data into groups.

    Will use grouping strategies specified in groups.json.

    Parameters
    ----------
    domain_name: str
        Name of the domain to load data from.
    root_path: str | Path
        The path to the directory containing the data. Defaults to ``DEFAULT_DATA_DIR``.
        See ``src.temper.utils.defaults`` for more information.
    metadata_file_name: str
        Name of the metadata file. Default is DEFAULT_METADATA_FILE.

    Returns
    -------
    List[GroupedDomain]:
        A list of grouped domains, each represents a collection of groups partitioned
        using one of the grouping strategies defined in the metadata file.
    """
    domain_path = (Path(root_path) / domain_name).resolve()

    metadata = loadfn(domain_path / metadata_file_name)
    grouping_strategies_and_kwargs = metadata["groupings"]

    # Build group entries.
    grouped_domains: List[GroupedDomain] = []
    for kwargs in grouping_strategies_and_kwargs:
        if len(grouped_domains) == 0:
            preload_info_entries = None
        else:
            preload_info_entries = grouped_domains[-1].info_entries
        grouped_domain = GroupedDomain.from_datadir_with_strategy(
            domain_path,
            info_entries=preload_info_entries,
            metadata_file_name=metadata_file_name,
            **kwargs
        )
        grouped_domains.append(grouped_domain)

    return grouped_domains
