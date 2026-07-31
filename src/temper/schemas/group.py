from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.temper.grouping_strategies import GROUPING_STRATEGIES

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
        groups (dict[str, list[str]]): The groups of structure data files.
        add_extra_cross_tests (bool): Whether to add extra cross tests.
          If true, beyond testing within each group, data from other groups will also be used for testing,
          and testing results will be reported separately for each group.
    """

    grouping_strategy: str

    groups: dict[str, list[str]]

    add_extra_cross_tests: bool = False

    @field_validator("groups")
    @classmethod
    def validate_groups(
        cls,
        value: dict[str, list[str]],
    ) -> dict[str, list[str]]:

        if not isinstance(value, dict):
            raise TypeError(
                "groups must be a dict mapping from string to list of lists."
            )

        for name, group in value.items():
            if not isinstance(name, str):
                raise TypeError(
                    f"Group name: {name} must be a string."
                )

            if not isinstance(group, list):
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

    def as_dict(self) -> dict[str, Any]:
        """
        Convert to a plain dictionary.

        Compatible with monty.serialization.dumpfn.
        """
        return self.model_dump()

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
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
    ):
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
