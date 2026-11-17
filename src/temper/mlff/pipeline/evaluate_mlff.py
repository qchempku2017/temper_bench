"""Run ordered MLFF predictions through one locally selected ASE
Calculator."""

from __future__ import annotations

from temper.mlff.pipeline.calculators import build_calculator
from temper.schemas.mlff_experiment_result import MLFFExperimentManifest, MLFFTestManifest
from temper.results_loading.manifest_io import load_bundle_from_manifest
from temper.utils.file_paths import resolve_path_in_bundle
from temper._version import __version__
from temper.utils.file_hashing import file_sha256

from collections.abc import Iterable
import logging
import os
import time
from importlib import metadata
from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator
from ase.io import iread

logger = logging.getLogger(__name__)


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _installed_versions(
    package_names: Iterable[str],
) -> dict[str, str | None]:
    versions: dict[str, str | None] = {"temper-bench": __version__}
    for name in package_names:
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


def _evaluate_dataset(
    *,
    bundle_root: Path,
    calculator: Calculator,
    manifest: MLFFTestManifest,
) -> None:
    """Write predictions and fill one test's result fields after success."""
    logger.info("Evaluating dataset %s", manifest.dataset_id)
    started = time.perf_counter()
    properties = manifest.requested_properties
    if "energy" not in properties or "forces" not in properties:
        raise ValueError(
            f"Dataset {manifest.dataset_id} requires energy and forces."
        )
    source = resolve_path_in_bundle(bundle_root, manifest.execution_input_path)
    dataset_hash = file_sha256(source)
    if manifest.reference_sha256 != dataset_hash:
        raise ValueError(f"Test dataset bytes changed: {manifest.dataset_id}")

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
                    f"forces shape {frame_forces.shape} does not match"
                    f" ({len(atoms)}, 3)"
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
                    raise ValueError(
                        f"stress shape {stress.shape} does not match (3, 3)"
                    )
                stresses.append(stress)
        except Exception as error:
            raise RuntimeError(
                f"Evaluation failed for {manifest.dataset_id} frame {frame_index}:"
                f" {error}"
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
        arrays["stresses"] = np.asarray(stresses, dtype=np.float64).reshape(
            (-1, 3, 3)
        )
    if any(array.dtype == object for array in arrays.values()):
        raise RuntimeError("Prediction output must not contain object arrays.")

    prediction_path = resolve_path_in_bundle(bundle_root, manifest.prediction_path)
    wall_time = time.perf_counter() - started
    _write_npz(prediction_path, arrays)
    logger.debug("Wrote predictions to %s", prediction_path)
    manifest.prediction_sha256 = file_sha256(prediction_path)
    manifest.wall_time_seconds = wall_time
    manifest.status = "completed"


def run_test(
    manifest: MLFFExperimentManifest, *, bundle_root: str | Path,
) -> None:
    """Evaluate the manifest's datasets using its referenced bundle settings.

    Parameters
    ----------
    manifest : MLFFExperimentManifest
        Running experiment manifest created by the pipeline runner. This
        function updates its test entries and execution status in place and
        saves progress to bundle_root/results.json after each dataset. Training
        status and duration are preserved. Test descriptions, units and measured
        evaluation durations are stored here; installed package versions are
        recorded once for all tests. No manifest is loaded or created here.
        Each call evaluates all entries again, clearing a test's previous
        checksum and duration when that test starts. The referenced bundle
        supplies shared calculator settings, package requirements, and model
        selection: pretrained weights for zero-shot, or the trained model
        (including any temporary evaluation export) for fine-tuning.
    bundle_root : str or pathlib.Path
        Directory containing the manifest's referenced bundle, model files,
        and prediction outputs. Expands ~ and resolves relative paths against
        the current working directory. Copied test inputs are read relative
        to bundle_root using each test's execution_input_path. Their checksums
        are verified before prediction; original workstation paths are never used.

    Raises
    ------
    ValueError
        If a path is invalid, the bundle or reference bytes have changed,
        or test descriptions do not match the referenced bundle.
    RuntimeError
        If a Calculator fails on a requested dataset property.
    ImportError
        If the selected calculator backend is not installed.
    OSError
        If required input files cannot be read or predictions and manifest
        updates cannot be written.

    Notes
    -----
    Schema version 1 writes numerical arrays only to NPZ files and records
    their descriptions in the single root results.json.
    Exact exported filenames are preserved in test manifests;
    dataset identifiers are the original test filenames.
    Fine-tuning callers must finish training and prepare any required
    evaluation export first; run_pipeline performs both steps and cleans
    up the temporary export afterward.
    """
    bundle_root = Path(bundle_root).expanduser().resolve()
    bundle = load_bundle_from_manifest(manifest, bundle_root)
    files = bundle.files
    assert files is not None
    model_path = (
        files.evaluation_model or files.training_output_files["trained_model"]
        if bundle.unit_type == "finetune"
        else files.pretrained_model
    )
    calculator = build_calculator(
        identifier=bundle.mlff_spec.mlff_type,
        model=resolve_path_in_bundle(bundle_root, model_path),
        parameters=bundle.mlff_spec.testing_parameters,
    )
    manifest.package_versions = _installed_versions(
        item.name for item in bundle.mlff_spec.implementations
        if item.kind == "python_distribution"
    )
    manifest.status = "running"
    for test in manifest.test_manifests:
        test.status = "pending"
        test.prediction_sha256 = None
        test.wall_time_seconds = None
        try:
            _evaluate_dataset(
                bundle_root=bundle_root,
                calculator=calculator, manifest=test,
            )
            # Save each completed test in case execution stops abruptly later.
            manifest.to_json_file(bundle_root / "results.json")
        except BaseException:
            test.status = "failed"
            manifest.status = "failed"
            manifest.to_json_file(bundle_root / "results.json")
            raise
    manifest.status = "completed"
    manifest.to_json_file(bundle_root / "results.json")
