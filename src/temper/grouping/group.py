"""Loads a domain metadata and builds grouped-domain objects using its configured grouping strategies."""


from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Union, List

from monty.serialization import loadfn

from temper.schemas.group import GroupedDomain
from temper.utils.defaults import DEFAULT_DATA_DIR, DEFAULT_METADATA_FILE
from temper.logging import format_elapsed, progress_task


logger = logging.getLogger(__name__)


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
    started_at = time.monotonic()

    metadata = loadfn(domain_path / metadata_file_name)
    grouping_strategies_and_kwargs = metadata["groupings"]
    logger.info(
        "Grouping domain %r with %d configured strategy/strategies.",
        domain_name,
        len(grouping_strategies_and_kwargs),
    )

    # Build group entries.
    grouped_domains: List[GroupedDomain] = []
    with progress_task(
        logger,
        f"Grouping domain {domain_name!r}",
        total=len(grouping_strategies_and_kwargs),
        unit="strategies",
    ) as progress:
        for strategy_index, kwargs in enumerate(
            grouping_strategies_and_kwargs,
            start=1,
        ):
            strategy_name = str(kwargs.get("grouping_strategy", "unknown"))
            progress.update(
                detail=(
                    f"strategy {strategy_index}/{len(grouping_strategies_and_kwargs)} "
                    f"{strategy_name!r}"
                )
            )
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
            progress.advance()
            logger.debug(
                "Grouping strategy %r produced %d group(s).",
                strategy_name,
                len(grouped_domain.groups),
            )

    group_count = sum(len(grouped.groups) for grouped in grouped_domains)
    info_entries = grouped_domains[0].info_entries if grouped_domains else []
    frame_count = sum(
        sum(entry.num_frames_per_system)
        for entry in info_entries
    )
    files_without_stress = sum(not entry.has_stress for entry in info_entries)
    logger.info(
        "Grouped domain %r into %d group(s) across %d strategy/strategies "
        "from %d file(s) and %d frame(s) in %s.",
        domain_name,
        group_count,
        len(grouped_domains),
        len(info_entries),
        frame_count,
        format_elapsed(time.monotonic() - started_at),
    )
    if files_without_stress:
        logger.warning(
            "Domain %r has %d/%d source file(s) without stress labels.",
            domain_name,
            files_without_stress,
            len(info_entries),
        )

    return grouped_domains
