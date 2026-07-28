"""Grouping strategies used for splitting datas in a datadir into groups.

Structure data from files belonging to the same group will be merged together
into a dataset that will further be split into train, validation and test sets.
"""


from typing import Callable
import re


def group_by_every_file(files: list[str]) -> list[list[str]]:
    """Each file is a group of its own."""
    return [[file] for file in files]

def group_by_regex(files: list[str], regex: str) -> list[list[str]]:
    """Group files by regex."""
    # TODO: not what I wanted. Should reimplement.
    groups: dict[str, list[str]] = {}
    for file in files:
        match = re.search(regex, file)
        if match:
            group_name = match.group(1)
            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append(file)
    return list(groups.values())


# All the grouping strategy functions should only take in a list of files (and necessary kwargs)
# and return a list of lists of files.
GROUPING_STRATEGIES: dict[str, Callable] = {
    "by_every_file": group_by_every_file,
    "by_regex": group_by_regex
}

