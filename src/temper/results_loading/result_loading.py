"""Load local scientific data described by a persistent result manifest."""

from __future__ import annotations

from pathlib import Path

from ase.io import read
import numpy as np
from numpy.typing import NDArray

from temper.schemas.mlff_experiment_result import (
    MLFFExperimentResult, MLFFTestResult, MLFFTestManifest, MLFFExperimentManifest,
)
from temper.schemas.mlff_train_bundle import MLFFTrainBundle
from temper.results_loading.manifest_io import load_bundle_from_manifest
from temper.utils.file_paths import resolve_path_in_bundle
from temper.utils.file_hashing import file_sha256


def load_results_from_manifest(
    manifest: MLFFExperimentManifest, *,
    base_dir: str | Path, require_finite: bool = True,
) -> MLFFExperimentResult:
    """Load all completed test_results without caching data in the manifest.

    Parameters
    ----------
    manifest : MLFFExperimentManifest
        Persistent file descriptions and recorded training measurements.
        Uses the supplied object's fields; the manifest file is not reread.
    base_dir : str or Path
        Directory containing the manifest's referenced experiment settings and
        predictions. Pass the parent of results.json. Expands ~ and resolves
        relative paths against the current working directory. Reference labels
        are read from bundle.training_unit.root_path/domain/source_filename.
        Packaging records an absolute original dataset root. Users must keep
        the split datasets at that location, unchanged, for later analysis.
    require_finite : bool, default True
        Reject NaN and infinity in predictions and labels.

    Returns
    -------
    MLFFExperimentResult
        Independently loaded settings and scientific data, with package
        versions from the experiment manifest. Each test exposes its description
        and units through its manifest; NPZ files supply only numerical arrays.
        All completed test_results are loaded in manifest order. Pending and failed
        test_results remain described in the manifest but have no loaded result.

    Raises
    ------
    ValueError
        If original datasets for completed test_results are missing,
        referenced files changed, or labels, units, shapes and frame order disagree.
    OSError
        If a required file cannot be read.
    """
    root = Path(base_dir).expanduser().resolve()
    bundle = load_bundle_from_manifest(manifest, root)
    test_results = tuple(
        _load_test_result_from_manifest(
            bundle, entry, root=root, require_finite=require_finite,
        )
        for entry in manifest.test_manifests if entry.status == "completed"
    )
    return MLFFExperimentResult(
        bundle=bundle, status=manifest.status, attempt_name=manifest.attempt_name,
        training_status=manifest.training_status,
        training_wall_time_seconds=manifest.training_wall_time_seconds,
        test_results=test_results,
        file_sha256={entry.prediction_path: entry.prediction_sha256
                     for entry in manifest.test_manifests if entry.prediction_sha256 is not None},
        package_versions=dict(manifest.package_versions),
    )


def _verify_prediction_file(manifest: MLFFTestManifest, root: Path) -> Path:
    """Check a prediction file's checksum and return its resolved path.

    Parameters
    ----------
    manifest : MLFFTestManifest
        Test metadata containing a bundle-relative prediction_path and a
        prediction_sha256 recorded after evaluation. A missing checksum
        prevents verification.
    root : pathlib.Path
        Directory containing the prediction outputs, normally the parent of
        results.json. Relative roots use the current working directory.

    Returns
    -------
    pathlib.Path
        Absolute prediction-file path, confined to root after resolving
        symbolic links, whose bytes match the recorded SHA-256 checksum.
        This reads the file for hashing but does not load or validate its
        numerical arrays. No files or manifest fields are modified.

    Raises
    ------
    ValueError
        If the checksum is absent, the path is invalid or escapes root,
        or the file bytes differ from the recorded checksum.
    OSError
        If the prediction file cannot be read or its path cannot be resolved.
    """
    if manifest.prediction_sha256 is None:
        raise ValueError("Missing prediction checksum.")
    prediction_path = resolve_path_in_bundle(root, manifest.prediction_path)
    if file_sha256(prediction_path) != manifest.prediction_sha256:
        raise ValueError("Retained prediction changed.")
    return prediction_path


def _load_test_result_from_manifest(
    bundle: MLFFTrainBundle, manifest: MLFFTestManifest,
    *, root: Path, require_finite: bool,
) -> MLFFTestResult:
    """Load one test's predictions and original labels, checking alignment.

    Parameters
    ----------
    bundle : MLFFTrainBundle
        Packaged settings already checked against the experiment manifest by
        load_bundle_from_manifest. Its files mapping must be populated.
        Original labels are read from training_unit.root_path/domain using
        manifest.source_filename; temporary execution inputs are not used.
    manifest : MLFFTestManifest
        Completed test supplied by load_results_from_manifest, with reference
        and prediction checksums, requested properties, units, and the
        prediction-file path.
        The source filename must belong to the bundle's ordered test sets.
    root : pathlib.Path
        Directory containing the prediction outputs, normally the parent of
        results.json. load_results_from_manifest supplies an absolute root;
        a relative root is interpreted against the current working directory.
    require_finite : bool
        If True, reject NaN and infinity in predictions and reference labels.
        False allows these values for inspection while retaining all checksum,
        property, unit, shape, and frame-alignment checks.

    Returns
    -------
    MLFFTestResult
        Original ASE structures and aligned prediction and reference arrays:
        energies (n_frames,) in eV, forces (total_atoms, 3) in eV/Angstrom,
        and, when requested, stresses (n_frames, 3, 3) in eV/Angstrom^3.
        Atom offsets delimit each frame's forces, and frame indices preserve
        original dataset order. The result retains the supplied test manifest.
        No files or input objects are modified, and loaded data are not cached.

    Raises
    ------
    ValueError
        If original data are missing or empty, a checksum differs, required
        labels are unavailable, properties or units disagree with the bundle,
        or NPZ array names, shapes, numeric types or frame alignment are
        invalid. Also raised for nonfinite values when require_finite is True.
    OSError
        If a dataset or prediction file cannot be read.
    AssertionError
        If the caller supplies a bundle without a populated files mapping.

    Notes
    -----
    NPZ files are opened with allow_pickle=False. Checksums establish which
    bytes were used; alignment checks establish how predictions correspond to
    reference frames and atoms, without assessing their physical accuracy.
    """
    reference = bundle.training_unit.resolve_local_dataset_path(manifest.source_filename)
    if file_sha256(reference) != manifest.reference_sha256:
        raise ValueError("Reference dataset checksum differs.")
    files = bundle.files
    assert files is not None
    index = list(bundle.training_unit.test_sets).index(manifest.source_filename)
    properties = ["energy", "forces"] + (["stress"] if files.test_stress[index] else [])
    units = {"energy": "eV", "forces": "eV/Angstrom", "stress": "eV/Angstrom^3" if "stress" in properties else None}
    if manifest.requested_properties != tuple(properties) or manifest.units != units:
        raise ValueError("Prediction properties or units do not match the bundle.")
    frames = tuple(read(reference, index=":"))
    counts = np.asarray([len(atoms) for atoms in frames], dtype=np.int64)
    if not frames or np.any(counts <= 0):
        raise ValueError("Test data must contain nonempty structures.")
    offsets = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(counts)))
    array_path = _verify_prediction_file(manifest, root)
    keys = {"energies", "forces", "atom_offsets", "frame_indices"}
    if "stress" in properties:
        keys.add("stresses")
    with np.load(array_path, allow_pickle=False) as saved:
        if set(saved.files) != keys:
            raise ValueError("Unexpected or missing prediction arrays.")
        arrays = {key: saved[key] for key in keys}
    for key, expected in (("atom_offsets", offsets), ("frame_indices", np.arange(len(frames)))):
        if arrays[key].dtype.kind not in "iu" or not np.array_equal(arrays[key], expected):
            raise ValueError(f"Invalid prediction {key} alignment.")
    shapes: dict[str, tuple[int, ...]] = {
        "energies": (len(frames),), "forces": (int(offsets[-1]), 3)
    }
    if "stress" in properties:
        shapes["stresses"] = (len(frames), 3, 3)
    predictions: dict[str, NDArray[np.float64]] = {}
    for key, shape in shapes.items():
        if arrays[key].shape != shape or arrays[key].dtype.kind not in "fi":
            raise ValueError(f"Invalid prediction {key} shape or dtype.")
        predictions[key] = np.asarray(arrays[key], dtype=np.float64)
    try:
        references = {
            "energies": np.asarray([a.get_potential_energy() for a in frames], dtype=np.float64),
            "forces": np.concatenate([a.get_forces(apply_constraint=False) for a in frames]),
        }
        if "stress" in properties:
            references["stresses"] = np.asarray([
                a.get_stress(voigt=False, apply_constraint=False, include_ideal_gas=False)
                for a in frames
            ], dtype=np.float64)
    except Exception as error:
        raise ValueError("Reference dataset lacks required physical labels.") from error
    for key, values in references.items():
        if values.shape != shapes[key]:
            raise ValueError(f"Invalid reference {key} shape.")
    if require_finite and any(not np.isfinite(a).all() for a in (*predictions.values(), *references.values())):
        raise ValueError("Predictions or labels contain NaN or infinity.")
    return MLFFTestResult(
        manifest, frames, predictions, references, offsets,
        np.asarray(arrays["frame_indices"], dtype=np.int64),
    )
