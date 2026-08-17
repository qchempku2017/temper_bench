"""Utility functions for loading data from files inside a domain and grouping them."""


from __future__ import annotations

from pathlib import Path
from typing import Union, List

from monty.serialization import loadfn

from temper.schemas import GroupedDomain
from temper.utils.defaults import DEFAULT_DATA_DIR, DEFAULT_METADATA_FILE


def load_grouped_domains_from_domain_name(
        domain_name: str,
        data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
        metadata_file_name: str = DEFAULT_METADATA_FILE,
) -> List[GroupedDomain]:
    """Load data from domain_name folder in data_dir by groups specified in groups.json.

    Parameters
    ----------
    domain_name: str
        Name of the domain to load data from.
    data_dir: str | Path
        The path to the directory containing the data.
    metadata_file_name: str
        Name of the metadata file. Default is DEFAULT_METADATA_FILE.

    Returns
    -------
    List[GroupedDomain]:
        All group names with the corresponding GroupEntry objects constructed
        from the domain folder.
    """
    domain_path = Path(data_dir) / domain_name

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
