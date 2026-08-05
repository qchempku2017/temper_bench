"""Grouping strategies used for splitting datas in a datadir into groups.

Structure data from files belonging to the same group will be merged together
into a dataset that will further be split into train, validation and test sets.

Allowed grouping strategies, including:
- group_by_every_file: Each file is a group of its own.
- group_by_regex: Group files by regex matching.
- group_by_property: Group files by properties extracted from the file name.
- group_as_specified: Group files as specified by the user.

When using `group_by_regex` or `group_by_property`, the file names must follow certain naming conventions,
 this module determines which files belong to the same group by matching file names with regexes.

Therefore, for default grouping strategies to work, the file names must follow certain naming conventions:
1. The file must be extxyz format with the exact extension ".extxyz".
2. The file name must be parts separated by "_", and only with underscore.
3. The file name must contain the property to be grouped by, followed by the value of the property. The property
   and the values must be separated by "_". Only properties "u_specification" and "mag_specification" can be
   specified by "u" and "no_u", "mag" and "no_mag" values only without the property name.
4. Property names and values must follow conventions specified in DEFAULT_REGEX_AND_GROUP_NAMES. Supported default
   properties include:
    temperature, pressure, composition_1d_numerical, composition_string, u_specification, mag_specification.
5. If the required file name format does not match the default naming conventions,
    you need to use function `group_by_regex` with custom regex and group_name_format`.
"""
from typing import Callable, Dict, Pattern, Optional
import re
from pathlib import Path
from collections import defaultdict


# property: (attempt_regex, group_name_format)
DEFAULT_REGEX_AND_GROUP_NAMES = {
    "temperature": (
        r".*_(?:t|T|Temp|temp|temperature|Temperature)_(\d*\.?\d+)[kK]*_.*",
        "temperature_{0}"
    ),
    "pressure": (
        r".*_(?:p|P|pressure|Pressure)_(\d*\.?\d+)[gG][pP][aA]*_.*",
        "pressure_{0}"
    ),
    # One-dimensional composition in binary systems. More component compositions in numerical format
    # can not be treated by default. Use a custom regex and group_name_format, or use "composition_string"
    # instead.
    "composition_1d_numerical": (
        r".*_(?:x|c|C|comp|Comp|composition|Composition)_(\d*\.?\d+)_.*",
        "composition_{0}"
    ),
    "composition_string": (
        r".*_(?:c|C|comp|Comp|composition|Composition)_([a-zA-Z]+)_.*",
        "composition_{0}"
    ),
    "u_specification": (
        r".*_(u|no_u)_.*",
        "{0}"
    ),
    "mag_specification": (
        r".*_(mag|no_mag)_.*",
        "{0}"
    ),
}


def group_by_every_file(files: list[str]) -> Dict[str, list[str]]:
    """Each file is a group of its own."""
    result = {}
    for file in files:
        key = str(Path(file).stem)
        if key in result:
            raise ValueError(
                f"Duplicate file stem: {key} from {result[key][0]} and {file}."
            )
        result[key] = [file]
    return result


def group_by_regex(
    files: list[str],
    regex: str | Pattern[str],
    group_name: Optional[str] = None,
    strict: bool = True,
) -> dict[str, list[str]]:
    """
    Group filenames according to regex-matched groups and generate
    human-readable group names.

    The file extension is ignored before applying the regex. The captured
    regex groups are used to construct a string identifier for each group.

    Parameters
    ----------
    files : list[str]
        List of filenames.

        Example:
            [
                "sto_x_0.000_t_300_no_u_no_mag.extxyz",
                "lco_x_0.000_t_300_no_u_no_mag.extxyz",
            ]

    regex : str | Pattern[str]
        Regular expression with capturing groups.

        Example:
            r".*_t_(\\d+)_(.*)"

        Captures:
            ("300", "no_u_no_mag")

    group_name : str, default=None
        Format string used to generate group names.

        Available formats:

        1. None:
           Join all captured groups with "_".

           Example:
               ("300", "no_u_no_mag")
               ->
               "300_no_u_no_mag"

        2. Positional formatting:

               "t_{0}_{1}"

           Example:
               ("300", "no_u_no_mag")
               ->
               "t_300_no_u_no_mag"

    strict : bool, default=True
        If True, raise an error if any file does not match the regex.

    Returns
    -------
    dict[str, list[str]]
        Mapping from group names to filenames.

        Example:

        {
            "t_300_no_u_no_mag": [
                "sto_x_0.000_t_300_no_u_no_mag.extxyz",
                "lco_x_0.000_t_300_no_u_no_mag.extxyz",
            ],

            "t_600_no_u_no_mag": [
                "sto_x_0.000_t_600_no_u_no_mag.extxyz",
            ]
        }

    Raises
    ------
    ValueError
        If strict=True and some files do not match the regex.

    Examples
    --------
    >>> files = [
    ...     "sto_x_0.000_t_300_no_u_no_mag.extxyz",
    ...     "sto_x_0.000_t_600_no_u_no_mag.extxyz",
    ...     "sto_x_0.000_t_300_u_mag.extxyz",
    ...     "lco_x_0.000_t_300_no_u_no_mag.extxyz",
    ... ]

    >>> groups = group_by_regex(
    ...     files,
    ...     r".*_t_(\\d+)_(.*)",
    ...     group_name="t_{0}_{1}",
    ... )

    >>> groups["t_300_no_u_no_mag"]
    [
        "sto_x_0.000_t_300_no_u_no_mag.extxyz",
        "lco_x_0.000_t_300_no_u_no_mag.extxyz",
    ]
    """

    pattern = re.compile(regex) if isinstance(regex, str) else regex

    groups: defaultdict[str, list[str]] = defaultdict(list)
    unmatched = []

    for filename in files:
        stem = Path(filename).stem

        match = pattern.search(stem)

        if match is None:
            unmatched.append(filename)
            continue

        captured = match.groups()

        if group_name is None:
            name = "_".join(captured)
        else:
            name = group_name.format(*captured)

        groups[name].append(filename)

    if strict and unmatched:
        raise ValueError(
            "The following files do not match the regex:\n"
            + "\n".join(unmatched)
        )

    return dict(groups)


def group_by_property(
        files: list[str],
        grouping_property: str,
        strict: bool = True,
) -> dict[str, list[str]]:
    """Group files by a single property using regex match of filenames.

    Parameters
    ----------
    files : list[str]
        List of filenames.
    grouping_property : str
        Property to group by. Corresponding regexes and group name formats can be found in
        `DEFAULT_REGEX_AND_GROUP_NAMES`.
    strict : bool, default=True
        If True, raise an error if any file does not match the regex.

    Returns
    -------
    dict[str, list[str]]
        Mapping from group names to filenames.

    Raises
    ------
    ValueError
        If strict=True and some files do not match the regex.
    """
    if not grouping_property in DEFAULT_REGEX_AND_GROUP_NAMES:
        raise ValueError(
            f"Property {grouping_property} not found in DEFAULT_REGEX_AND_GROUP_NAMES."
        )
    attempt_regex, group_name = DEFAULT_REGEX_AND_GROUP_NAMES[grouping_property]
    try:
        groups = group_by_regex(
            files,
            attempt_regex,
            group_name=group_name,
            strict=strict,
        )
    except ValueError as exc:
        raise ValueError(
            f"Could not group files by property `{grouping_property}` using default regex: "
            f"{attempt_regex}. Original error:\n{exc}\n\n"
            "Please use `group_by_regex` and customize a valid regex in the `attempt_regex` argument."
        ) from exc
    else:
        return groups


def group_as_specified(
        files: list[str],  # pylint: disable=unused-argument
        groups: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Group files as specified in a dictionary.

    Parameters
    ----------
    files : list[str]
        List of filenames. Only placeholder for consistency with other grouping strategies.
    groups : dict[str, list[str]]
        Mapping from group names to filenames.

    Returns
    -------
    dict[str, list[str]]
        Mapping from group names to filenames.
    """
    return groups



# All the grouping strategy functions should only take in a list of files (and necessary kwargs)
# and return a dict mapping group names to lists of files.
GROUPING_STRATEGIES: dict[str, Callable] = {
    "by_every_file": group_by_every_file,
    "by_regex": group_by_regex,
    "by_property": group_by_property,
    "as_specified": group_as_specified,
}

