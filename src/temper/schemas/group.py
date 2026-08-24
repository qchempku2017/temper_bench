"""Defines the schema for a data domain partitioned into file groups. It also builds the schema from domain metadata and grouping strategies."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from pydantic import field_validator, model_validator

from temper.utils.defaults import DEFAULT_METADATA_FILE
from temper.schemas.frame_refrence import FrameReference
from temper.schemas.base import MSONableModel
from temper.schemas.info import InfoEntry, load_info_entries_from_datadir


class GroupedDomain(MSONableModel):
    """Storage of a data domain that has been grouped into multiple groups by a strategy.

    Each GroupEntry corresponds to one dictionary entry:
    {
        "grouping_strategy": "...",
        "groups": [...],
        "add_extra_cross_tests": true/false
    }

    This is the recommended top-level API to access domain data.

    Attributes:
        domain (str): The name of the data domain that the grouping strategy is applied to.
        info_entries (List[InfoEntry]): The information entries of all files under the data domain.
        grouping_strategy (str): The name of the grouping strategy. See available strategies in
          src.temper.utils.grouping.GROUPING_STRATEGIES.
        groups (Dict[str, List[str]]): The groups of structure data files. Each group is a list of file names,
          corresponding to all structure data that will be merged to create a dataset for train-val-test.
        add_extra_cross_tests (bool): Whether to add extra cross tests.
          If true, beyond testing within each group, data from other groups will also be used for testing,
          and testing results will be reported separately for each group.
        specify_cross_tests (Dict[str, List[str]] | None):
          If not None, will add cross tests for the specified groups instead of all groups on all other groups.
          The keys are the names of the groups that the model trained with whom should be tested, and the values
          are the names of the groups to be used as additional testing data.
          Only effective when add_extra_cross_tests is False, otherwise will automatically use all other groups
          to test the models trained with the current group.
          Default to None.
    """
    domain: str

    info_entries: List[InfoEntry]

    grouping_strategy: str

    groups: Dict[str, List[str]]

    add_extra_cross_tests: bool = False

    specify_cross_tests: Dict[str, List[str]] | None = None

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

    @model_validator(mode="after")
    def validate_group_membership(self) -> "GroupedDomain":
        """Ensure groups partition filenames are in info entries and are disjoint."""
        info_filenames = {entry.filename for entry in self.info_entries}
        filenames_to_groups: Dict[str, str] = {}
        for group_name, filenames in self.groups.items():
            for filename in filenames:
                if filename not in info_filenames:
                    raise ValueError(
                        f"groups[{group_name}] contains filename {filename!r} "
                        "that is not present in info_entries."
                    )
                previous_group = filenames_to_groups.get(filename)
                if previous_group is not None:
                    raise ValueError(
                        f"Filename {filename!r} occurs more than once: "
                        f"in groups {previous_group!r} and {group_name!r}."
                    )
                filenames_to_groups[filename] = group_name
        return self

    @classmethod
    def from_datadir_with_strategy(
            cls,
            datadir: str | Path,
            grouping_strategy: str,
            add_extra_cross_tests: bool = False,
            specify_cross_tests: Dict[str, List[str]] | None = None,
            metadata_file_name: str = DEFAULT_METADATA_FILE,
            info_entries: List[InfoEntry] | None = None,
            **grouping_strategy_kwargs
    ) -> "GroupedDomain":
        """Create GroupEntry from a directory of structure data files.

        Parameters
        -----------
        datadir : str | Path
            The directory containing the structure data files in the domain.
        grouping_strategy : str
            The name of the grouping strategy.
        add_extra_cross_tests : bool, optional
            Whether to add extra cross tests. Defaults to False.
        specify_cross_tests: Dict[str, List[str]], optional
            Specify the cross tests to be added. Only effective when
            `add_extra_cross_tests` is False. Defaults to None.
        metadata_file_name: str, optional
            The name of the metadata file. Defaults to `DEFAULT_METADATA_FILE`.
            See src.temper.utils.defaults.
        info_entries: str, optional
            Pre-loaded info entries. If not provided, the info entries will be
            loaded from the metadata file. Defaults to None.
            Recommended to use if you have one available because it is faster.
        **grouping_strategy_kwargs
            Additional keyword arguments for the grouping strategy function.
            See strategies.py for details.

        Returns
        --------
        GroupedDomain

        Raises
        --------
        ValueError
           If the grouping strategy is not recognized.
        """
        # Move local to avoid circular imports.
        from temper.grouping.strategies import GROUPING_STRATEGIES

        if grouping_strategy not in GROUPING_STRATEGIES:
            raise ValueError(
                f"Unknown grouping strategy: {grouping_strategy}. "
                f"Available strategies: {GROUPING_STRATEGIES}"
            )
        datadir = Path(datadir)
        domain = datadir.name
        if info_entries is None:
            info_entries = load_info_entries_from_datadir(
                datadir,
                metadata_file_name=metadata_file_name,
            )
        files = [entry.filename for entry in info_entries]
        groups = GROUPING_STRATEGIES[grouping_strategy](
            files, **grouping_strategy_kwargs
        )

        return cls(
            domain=domain,
            info_entries=info_entries,
            grouping_strategy=grouping_strategy,
            groups=groups,
            add_extra_cross_tests=add_extra_cross_tests,
            specify_cross_tests=specify_cross_tests,
        )

    def load_frame_references_in_groups(self) -> Dict[str, List[FrameReference]]:
        """Load the frames in the form of FrameReference objects in each group.

        Returns
        --------
        Dict[str, List[FrameReference]]
            Mapping group names to lists of FrameReference objects.
        """
        frame_references = {}
        all_file_names = [entry.filename for entry in self.info_entries]
        for group_name, group in self.groups.items():
            frame_references[group_name] = []
            for filename in group:
                # Get the info entry for the file from self.info_entries.
                info_entry = self.info_entries[all_file_names.index(filename)]
                n_frames_in_file = sum(info_entry.num_frames_per_system)
                frame_references[group_name].extend([
                    FrameReference(
                        domain=self.domain,
                        filename=filename,
                        frame_index=ii,
                    ) for ii in range(n_frames_in_file)
                ])
        return frame_references


