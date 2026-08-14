from __future__ import annotations

import warnings
from pathlib import Path
from typing import List

from ase import Atoms


def check_atoms_has_stress(frames: Atoms | List[Atoms]) -> bool:
    """Check if the frames have energy, forces, and stress correctly loaded.

      Implemented as checking whether the frames have SinglePointCalculators correctly loaded
    and callable.
    Parameters:
    -----------
    frames : Atoms | list[Atoms]
        List of ASE Atoms objects.

    Returns:
    --------
    bool
        True if the frames have stress information.

    Raises:
    -------
    ValueError
        If the frames do not have energy or forces information.
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
            UserWarning,
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

    def _get_other_properties(atoms: Atoms) -> set[str]:
        """Collect non-standard property keys from a single Atoms object."""
        props: set[str] = set()
        for key in atoms.info.keys():
            if key not in ["energy", "forces", "stress", "virial"]:
                props.add(key)
        if atoms.calc is not None:
            for key in atoms.calc.results.keys():
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
