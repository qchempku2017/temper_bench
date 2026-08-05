"""Grouping strategies used for splitting datas in a datadir into groups.

Structure data from files belonging to the same group will be merged together
into a dataset that will further be split into train, validation and test sets.

Allowed grouping strategies, including:
- group_by_every_file: Each file is a group of its own.
- group_as_specified: Group files as specified by the user.
- group_all: Group all files into one group named "all".
- group_by_regex: Group files by regex matching.
- group_by_property: Group files by properties extracted from the file name.
- group_by_neb_generalization: Group files by the location of the NEB image on reaction coordinates
   as indicated by increasing indices. End and midpoints are grouped together, while other points
   are divided into another group.

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
from typing import Callable, Dict, Pattern, Optional, List
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


def group_by_every_file(files: List[str]) -> Dict[str, List[str]]:
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


def group_as_specified(
        files: List[str],  # pylint: disable=unused-argument
        groups: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """Group files as specified in a dictionary.

    Parameters
    ----------
    files : List[str]
        List of filenames.
    groups : Dict[str, List[str]]
        Mapping from group names to filenames. The filenames must be a subset of `files`.

    Returns
    -------
    Dict[str, List[str]]
        Mapping from group names to filenames.

    Raises
    -------
    ValueError
        If a file in `groups` is not in `files`.
    """
    # Safety: check whether all file names appearing in groups are also in files.
    # Good practice to ensure consistency.
    files_set = set(files)
    for group_name, group_files in groups.items():
        for file in group_files:
            if file not in files_set:
                raise ValueError(
                    f"File {file} in group {group_name} not found in files."
                )
    return groups


def group_all(
        files: List[str],
) -> Dict[str, List[str]]:
    """Group all files into the same group.

    Parameters
    ----------
    files : List[str]
        List of filenames.

    Returns
    -------
    Dict[str, List[str]]
        Mapping from group names to filenames.
    """
    return {"all": files}


def group_by_regex(
    files: List[str],
    regex: str | Pattern[str],
    group_name: Optional[str] = None,
    strict: bool = True,
) -> Dict[str, List[str]]:
    """
    Group filenames according to regex-matched groups and generate
    human-readable group names.

    The file extension is ignored before applying the regex. The captured
    regex groups are used to construct a string identifier for each group.

    Parameters
    ----------
    files : List[str]
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
    Dict[str, List[str]]
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

    groups: defaultdict[str, List[str]] = defaultdict(list)
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
        files: List[str],
        grouping_property: str,
        strict: bool = True,
) -> Dict[str, List[str]]:
    """Group files by a single property using regex match of filenames.

    Parameters
    ----------
    files : List[str]
        List of filenames.
    grouping_property : str
        Property to group by. Corresponding regexes and group name formats can be found in
        `DEFAULT_REGEX_AND_GROUP_NAMES`.
    strict : bool, default=True
        If True, raise an error if any file does not match the regex.

    Returns
    -------
    Dict[str, List[str]]
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


def group_by_neb_generalization(
        files: List[str],
        strict: bool = True,
) -> Dict[str, List[str]]:
    """Group NEB files according to their location along reaction coordinates.

    Files are classified into two groups:

    - ``endpoints_and_midpoint``:
        Initial and final structures of each NEB path, along with the image
         located at the center of each NEB path, judged by index of the image.

    - ``intermediate``:
        Other images between endpoints and midpoint.

    The NEB image index is extracted from filename patterns:

        <reaction>_fp_00.extxyz
        <reaction>_loc_03.extxyz
        <reaction>_location_05.extxyz
        <reaction>_neb_06.extxyz

    The location prefix matching is case-insensitive. Supported prefixes:

        fp, loc, location, neb

    For each reaction path, the minimum and maximum image indices are
    treated as endpoints. The midpoint is determined from the average of
    the two endpoint indices.

    This will test generalization to the entire NEB path when training only
    on a portion of critical frames along the path.

    Parameters
    ----------
    files : List[str]
        List of filenames.

        Example:

            [
                "110_N2_N-N_fp_00.extxyz",
                "110_N2_N-N_fp_01.extxyz",
                "110_N2_N-N_fp_03.extxyz",
                "110_N2_N-N_fp_05.extxyz",
            ]

    strict : bool, default=True
        If True, raise an error if any file does not contain a valid
        NEB location index.

    Returns
    -------
    Dict[str, List[str]]
        Mapping from location categories to filenames.

        Example:

            {
                "endpoints_and_midpoint": [
                    "110_N2_N-N_fp_00.extxyz",
                    "110_N2_N-N_fp_05.extxyz",
                    "110_N2_N-N_fp_03.extxyz",
                ],

                "intermediate": [
                    "110_N2_N-N_fp_01.extxyz",
                    "110_N2_N-N_fp_02.extxyz",
                    "110_N2_N-N_fp_04.extxyz",
                ],
            }

    Raises
    ------
    ValueError
        If strict=True and some files do not contain valid NEB locations.

    Notes
    -----
    The midpoint is determined separately for each reaction path.

    Therefore, different NEB lengths are supported:

        fp_00 ... fp_05
            midpoint -> fp_02/fp_03 region, the one with smaller index taken, i.e., fp_02.

        fp_00 ... fp_06
            midpoint -> fp_03

        fp_00 ... fp_07
            midpoint -> fp_03/fp_04 region, the one with smaller index taken, i.e., fp_03.
    """

    location_pattern = re.compile(
        r"^(.*?)_(?:fp|loc|location|neb)_(\d+)(?:\D.*)?$",  # Non-greedy match.
        flags=re.IGNORECASE,
    )

    # First collect files belonging to the same reaction.
    reactions: defaultdict[str, List[tuple[int, str]]] = defaultdict(list)

    unmatched = []

    for filename in files:
        match = location_pattern.match(filename)

        if match is None:
            unmatched.append(filename)
            continue

        reaction = match.group(1)
        index = int(match.group(2))

        reactions[reaction].append((index, filename))

    if strict and unmatched:
        raise ValueError(
            "The following files do not contain valid NEB locations:\n"
            + "\n".join(unmatched)
        )

    groups: defaultdict[str, List[str]] = defaultdict(list)

    for frames in reactions.values():

        indices = sorted(idx for idx, _ in frames)

        min_idx = min(indices)
        max_idx = max(indices)

        midpoint = (min_idx + max_idx) / 2

        # Select closest image to midpoint.
        midpoint_idx = min(
            indices,
            key=lambda idx: abs(idx - midpoint)
        )

        for idx, filename in frames:

            if (
                    idx == min_idx
                    or idx == max_idx
                    or idx == midpoint_idx
            ):
                groups["endpoints_and_midpoint"].append(filename)

            else:
                groups["intermediate"].append(filename)

    return dict(groups)



# All the grouping strategy functions should only take in a list of files (and necessary kwargs)
# and return a dict mapping group names to lists of files.
GROUPING_STRATEGIES: Dict[str, Callable] = {
    "by_every_file": group_by_every_file,
    "as_specified": group_as_specified,
    "all": group_all,
    "by_regex": group_by_regex,
    "by_property": group_by_property,
    "by_neb_generalization": group_by_neb_generalization,
}

