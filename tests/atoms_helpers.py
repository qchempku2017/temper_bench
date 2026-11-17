"""Inspect ASE labels and shared properties for test assertions."""

from __future__ import annotations

import warnings

from ase import Atoms

from temper.logging import DataQualityWarning


def check_atoms_has_stress(frames: Atoms | list[Atoms]) -> bool:
    """Check required labels and report whether every frame includes
    stress.

    Energy, forces, and stress are queried through ASE rather than by
    inspecting Calculator internals, so any Calculator implementing the
    standard methods is accepted. Missing energy or forces is invalid;
    missing stress is reported
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
        An empty list returns True because no frame lacks stress.

    Raises
    ------
    ValueError
        If any frame does not provide energy or forces.

    Warns
    -----
    DataQualityWarning
        If one or more frames does not provide stress.

    Notes
    -----
    Calling ASE property getters may evaluate the attached calculator. This
    function is not restricted to labels already stored in an extxyz file.
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
    """Return additional property names present in every supplied structure.

    Checks within atoms.info and atoms.calc.results for properties other
    than energy, forces, stress, and virial.

    A property must be present in **all** frames to be reported.
    Properties that appear in only a subset of frames are excluded.

    Parameters
    ----------
    frames : Atoms | list[Atoms]
        One ASE structure or a list of structures. Property names are read
        from info and, when a calculator is attached, its existing results.
        No calculator evaluations are requested.

    Returns
    -------
    list[str]
        Sorted unique property names common to every frame, excluding energy,
        forces, stress and virial. An empty input list returns an empty list.
    """
    if isinstance(frames, Atoms):
        frames = [frames]

    if len(frames) == 0:
        return []

    def _get_other_properties(a: Atoms) -> set[str]:
        """Collect non-standard property keys from a single Atoms
        object."""
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
