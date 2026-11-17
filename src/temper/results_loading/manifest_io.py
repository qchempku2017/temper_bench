"""Validate the files referenced by a result manifest."""

from pathlib import Path

from temper.schemas.mlff_train_bundle import MLFFTrainBundle
from temper.schemas.mlff_experiment_result import MLFFExperimentManifest
from temper.utils.file_paths import resolve_path_in_bundle
from temper.utils.file_hashing import file_sha256


def load_bundle_from_manifest(
    manifest: MLFFExperimentManifest, root: Path,
) -> MLFFTrainBundle:
    """Load the referenced bundle and check its agreement with the manifest.

    Parameters
    ----------
    manifest : MLFFExperimentManifest
        Experiment metadata specifying the bundle path and SHA-256 checksum,
        training state, and ordered test descriptions. Uses the supplied
        object's fields without rereading its JSON file.
    root : pathlib.Path
        Directory against which manifest.bundle_path is resolved, normally
        the directory containing results.json. Relative roots use the current
        working directory; user-home abbreviations are not expanded.

    Returns
    -------
    MLFFTrainBundle
        Loaded settings with verified file checksum, stored identities, and
        test filenames, paths and recorded reference checksums consistent with
        the manifest. Original datasets, model weights and predictions are
        not opened. No files or supplied objects are modified.

    Raises
    ------
    ValueError
        If the bundle path, JSON or settings are invalid, its checksum differs,
        its file mapping is absent, or its test metadata or zero-shot training
        state disagrees with the manifest.
    KeyError
        If the bundle's file mapping lacks a referenced test input or output.
    OSError
        If the referenced bundle cannot be read or its path cannot be resolved.
    """
    path = resolve_path_in_bundle(root, manifest.bundle_path)
    if file_sha256(path) != manifest.bundle_sha256:
        raise ValueError("Experiment settings checksum differs.")
    bundle = MLFFTrainBundle.from_json_file(path, context={"verify_files": False})
    files = bundle.files
    if files is None:
        raise ValueError("Result bundle has no packaged file mapping.")
    if bundle.unit_type == "zeroshot" and manifest.training_status not in ("not_applicable", "unavailable"):
        raise ValueError("Zero-shot results cannot contain training measurements.")
    if len(manifest.test_manifests) != len(bundle.training_unit.test_sets):
        raise ValueError("Result dataset count differs from the bundle.")
    for index, entry in enumerate(manifest.test_manifests):
        key = bundle.training_unit.test_sets[index]
        if (
            entry.dataset_id != key
            or entry.source_filename != bundle.training_unit.test_sets[index]
            or entry.source_domain != bundle.training_unit.domain
            or entry.execution_input_path != files.test_input_files[key]
            or entry.prediction_path != files.test_output_files[key]
        ):
            raise ValueError("Result dataset does not match its bundle.")
        expected = bundle.training_unit.dataset_sha256.get(entry.source_filename)
        if expected is not None and entry.reference_sha256 != expected:
            raise ValueError("Reference checksum differs from the bundle.")
    return bundle
