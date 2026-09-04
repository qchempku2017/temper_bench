#!/usr/bin/env python3
"""Run ordered MLFF predictions through one locally selected ASE Calculator."""

from __future__ import annotations

import argparse
import json
import os
import time
from importlib import metadata
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
from ase.io import iread

# The filename is part of the materialized runtime contract, but also matches
# pytest's default ``*_test.py`` discovery pattern in the TEMPER source tree.
__test__ = False


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _installed_versions(requirements: list[dict[str, str]]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for requirement in requirements:
        name = requirement["name"]
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    try:
        versions["ase"] = metadata.version("ase")
        versions["numpy"] = metadata.version("numpy")
    except metadata.PackageNotFoundError:
        pass
    return versions


def _bundle_path(bundle_root: Path, value: str, *, label: str) -> Path:
    """Resolve a portable lexical path below the submit-directory root."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Invalid {label}: {value!r}.")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} for part in posix.parts)
        or str(posix) != value
    ):
        raise ValueError(f"{label.capitalize()} escapes bundle root: {value!r}.")
    return bundle_root.joinpath(*posix.parts)


def _evaluate_dataset(
    *,
    bundle_root: Path,
    dataset: dict[str, Any],
    calculator: Any,
    common_metadata: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    properties = list(dataset["properties"])
    if "energy" not in properties or "forces" not in properties:
        raise ValueError(
            f"Dataset {dataset['id']} requires energy and forces."
        )
    source = _bundle_path(
        bundle_root,
        dataset["path"],
        label="test dataset",
    )

    energies: list[float] = []
    forces: list[np.ndarray] = []
    stresses: list[np.ndarray] = []
    offsets = [0]
    frame_indices: list[int] = []
    for frame_index, atoms in enumerate(iread(source, index=":")):
        atoms.calc = calculator
        try:
            energies.append(float(atoms.get_potential_energy()))
            frame_forces = np.asarray(
                atoms.get_forces(apply_constraint=False), dtype=np.float64
            )
            if frame_forces.shape != (len(atoms), 3):
                raise ValueError(
                    f"forces shape {frame_forces.shape} does not match ({len(atoms)}, 3)"
                )
            forces.append(frame_forces)
            if "stress" in properties:
                stress = np.asarray(
                    atoms.get_stress(
                        voigt=False,
                        apply_constraint=False,
                        include_ideal_gas=False,
                    ),
                    dtype=np.float64,
                )
                if stress.shape != (3, 3):
                    raise ValueError(f"stress shape {stress.shape} does not match (3, 3)")
                stresses.append(stress)
        except Exception as error:
            raise RuntimeError(
                f"Evaluation failed for {dataset['id']} frame {frame_index}: {error}"
            ) from error
        offsets.append(offsets[-1] + len(atoms))
        frame_indices.append(frame_index)

    concatenated_forces = (
        np.concatenate(forces, axis=0)
        if forces
        else np.empty((0, 3), dtype=np.float64)
    )
    arrays = {
        "energies": np.asarray(energies, dtype=np.float64),
        "forces": concatenated_forces,
        "atom_offsets": np.asarray(offsets, dtype=np.int64),
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
    }
    if "stress" in properties:
        arrays["stresses"] = np.asarray(stresses, dtype=np.float64).reshape((-1, 3, 3))
    if any(array.dtype == object for array in arrays.values()):
        raise RuntimeError("Prediction output must not contain object arrays.")

    prediction_path = _bundle_path(
        bundle_root,
        dataset["output"],
        label="prediction output",
    )
    metadata_path = _bundle_path(
        bundle_root,
        dataset["metadata_output"],
        label="prediction metadata output",
    )
    _write_npz(prediction_path, arrays)
    wall_time = time.perf_counter() - started
    result_metadata = {
        **common_metadata,
        "dataset_id": dataset["id"],
        "source_domain": dataset["source_domain"],
        "source_filename": dataset["source_filename"],
        "submit_path": dataset["path"],
        "number_of_frames": len(energies),
        "total_atoms": offsets[-1],
        "requested_properties": properties,
        "units": {
            "energy": "eV",
            "forces": "eV/Angstrom",
            "stress": "eV/Angstrom^3" if "stress" in properties else None,
        },
        "wall_time_seconds": wall_time,
    }
    _write_json(metadata_path, result_metadata)
    return {
        "dataset_id": dataset["id"],
        "prediction_file": dataset["output"],
        "metadata_file": dataset["metadata_output"],
        "number_of_frames": len(energies),
        "total_atoms": offsets[-1],
        "wall_time_seconds": wall_time,
    }


def run(config_path: str | Path) -> None:
    """Evaluate every configured dataset and write standardized predictions.

    Parameters
    ----------
    config_path : str or pathlib.Path
        Version-2 test configuration inside a written submit directory.

    Raises
    ------
    ValueError
        If the configuration or any submit-relative path is invalid.
    RuntimeError
        If a Calculator fails on a requested dataset property.
    """
    # Import from the colocated, selected adapter only when the standalone
    # runtime actually executes. This keeps the source-tree module importable
    # without a sibling ``calculator.py`` and preserves the remote-free
    # materialized contract.
    from calculator import build_calculator

    config_file = Path(config_path).expanduser().resolve()
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if config.get("schema_version") != 2:
        raise ValueError("Unsupported test_config schema_version.")
    bundle_root = config_file.parent

    calculator = build_calculator(config)
    package_versions = _installed_versions(config.get("package_requirements", []))
    common_metadata = {
        "schema_version": 1,
        "calculator_identifier": config["calculator"]["identifier"],
        "model": config["model"],
        "package_versions": package_versions,
    }
    started = time.perf_counter()
    results = [
        _evaluate_dataset(
            bundle_root=bundle_root,
            dataset=dataset,
            calculator=calculator,
            common_metadata=common_metadata,
        )
        for dataset in config["test_datasets"]
    ]
    _write_json(
        _bundle_path(
            bundle_root,
            config["summary_output"],
            label="summary output",
        ),
        {
            **common_metadata,
            "datasets": results,
            "wall_time_seconds": time.perf_counter() - started,
        },
    )


def main() -> None:
    """Parse the test configuration path and run all configured datasets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    run(arguments.config)


if __name__ == "__main__":
    main()
