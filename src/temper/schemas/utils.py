"""Helpers for validating ASE structure properties and safe extxyz source paths used by schemas."""

from __future__ import annotations

import warnings
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import List

from ase import Atoms

from temper.logging import DataQualityWarning


def validate_submit_relative_path(value: str, *, label: str = "path") -> str:
    """Validate a portable path that is safe to place below a submit directory.

    Submit bundles are created locally and may later be copied to a different
    operating system. Their stored paths therefore use POSIX ``/`` separators
    and must be relative to the bundle root. This helper rejects absolute paths,
    Windows drive paths, traversal, redundant segments, and other forms whose
    meaning could change after transport.

    Parameters
    ----------
    value : str
        Path to validate, written with POSIX separators. It may contain
        subdirectories, for example ``"datasets/train.extxyz"``.
    label : str, optional
        Human-readable name used in validation errors. It does not affect the
        returned path.

    Returns
    -------
    str
        ``value`` unchanged after validation.

    Raises
    ------
    TypeError
        If ``value`` or ``label`` is not a string.
    ValueError
        If ``value`` is empty, absolute, drive-qualified, uses backslashes, or
        contains empty, current-directory, parent-directory, or unnormalized
        path segments.
    """
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string, got {type(value).__name__}.")
    if not isinstance(label, str):
        raise TypeError(f"label must be a string, got {type(label).__name__}.")
    if not value:
        raise ValueError(f"{label} must be a non-empty string.")
    if "\\" in value:
        raise ValueError(f"{label} must use '/' separators, got {value!r}.")

    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ValueError(f"{label} must be relative, got {value!r}.")
    if not posix.parts or any(
        part in {"", ".", ".."} for part in posix.parts
    ):
        raise ValueError(
            f"{label} must not contain empty, '.' or '..' segments: {value!r}."
        )
    if str(posix) != value:
        raise ValueError(f"{label} must be normalized, got {value!r}.")
    return value


def check_atoms_has_stress(frames: Atoms | List[Atoms]) -> bool:
    """Check required labels and report whether every frame includes stress.

    Energy, forces, and stress are queried through ASE rather than by inspecting
    Calculator internals, so any Calculator implementing the standard methods
    is accepted. Missing energy or forces is invalid; missing stress is reported
    as a data-quality warning and a false return value.

    Parameters
    ----------
    frames : Atoms | list[Atoms]
        One ASE structure or a list of structures whose labels should be
        checked.

    Returns
    -------
    bool
        True when every supplied frame provides stress, otherwise False.

    Raises
    ------
    ValueError
        If any frame does not provide energy or forces.

    Warns
    -----
    DataQualityWarning
        If one or more frames does not provide stress.
    """
    if isinstance(frames, Atoms):
        frames = [frames]

    has_stress = True

    for i, atoms in enumerate(frames):
        try:
            atoms.get_potential_energy()
        except Exception as exc:
            raise ValueError(
                f"Frame {i} in extxyz is missing energy. "
                "Please fix the extxyz file."
            ) from exc

        try:
            atoms.get_forces()
        except Exception as exc:
            raise ValueError(
                f"Frame {i} in extxyz is missing forces. "
                "Please fix the extxyz file."
            ) from exc

        try:
            atoms.get_stress()
        except Exception:  # noqa: PERF203
            has_stress = False

    if not has_stress:
        warnings.warn(
            "Stress information is missing in one or more frames. "
            "The dataset may not be suitable for stress-dependent benchmarks.",
            DataQualityWarning,
            stacklevel=2,
        )

    return has_stress


def check_atoms_have_other_properties(
        frames: Atoms | list[Atoms],
) -> list[str]:
    """Check if the frames have other properties.

    Checks within atoms.info and atoms.calc.results for properties other than
    energy, forces, stress, and virial.

    A property must be present in **all** frames to be reported. Properties
    that appear in only a subset of frames are excluded.

    Parameters
    ----------
    frames : Atoms | list[Atoms]
        List of ASE Atoms objects.

    Returns
    -------
    list[str]
        List of the name of other properties common to all frames.
    """
    if isinstance(frames, Atoms):
        frames = [frames]

    if len(frames) == 0:
        return []

    def _get_other_properties(a: Atoms) -> set[str]:
        """Collect non-standard property keys from a single Atoms object."""
        props: set[str] = set()
        for key in a.info.keys():
            if key not in ["energy", "forces", "stress", "virial"]:
                props.add(key)
        if a.calc is not None:
            for key in a.calc.results.keys():
                if key not in ["energy", "forces", "stress", "virial"]:
                    props.add(key)
        return props

    # Start with properties from the first frame ...
    other_properties = _get_other_properties(frames[0])

    # ... and keep only those present in every subsequent frame.
    for atoms in frames[1:]:
        other_properties &= _get_other_properties(atoms)

    return sorted(list(other_properties))


def validate_relative_extxyz_path(path: str) -> str:
    """Validate that a path is a relative extxyz path without directory traversal.

    Used to validate persisted source filenames. The path must be relative
    (not absolute, no drive/root), must not contain ``..`` traversal segments,
    and must end with the ``.extxyz`` suffix. Subdirectories (e.g.
    ``subdir/file.extxyz``) are allowed as long as they are relative.

    Parameters
    ----------
    path : str
        The source filename to validate.

    Returns
    -------
    str
        The validated path, unchanged.

    Raises
    ------
    TypeError
        If ``path`` is not a string.
    ValueError
        If ``path`` is absolute, contains traversal, or does not end with
        ``.extxyz``.
    """
    if not isinstance(path, str):
        raise TypeError(
            f"path must be a str, got {type(path).__name__}."
        )

    filepath = Path(path)

    # Reject absolute paths. On Windows a leading-root path such as
    # "/a.extxyz" is drive-relative (is_absolute() is False) but is still
    # rooted, so a non-empty ``root`` is rejected as well.
    if filepath.is_absolute() or filepath.root:
        raise ValueError(
            f"Source filename must be relative, got absolute path: {path}."
        )

    if ".." in filepath.parts:
        raise ValueError(
            f"Source filename must not contain directory traversal, got: {path}."
        )

    if filepath.suffix != ".extxyz":
        raise ValueError(
            f"Source filename must have the '.extxyz' extension, got: {path}."
        )

    return path
