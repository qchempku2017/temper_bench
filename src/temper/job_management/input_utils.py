"""Mappings for temporary copied inputs and their recorded content digests."""
from pathlib import Path
from temper.job_management.schemas.records import JobRecord
from temper.schemas.mlff_train_bundle import MLFFTrainBundle
from temper.utils.file_hashing import file_sha256


def load_bundle(
        record: JobRecord,
        root: Path,
        *,
        verify_files: bool = False,
        require_packaged: bool = False,
) -> MLFFTrainBundle:
    """Read the shared bundle and check that it belongs to this attempt.

    Parameters
    ----------
    record : JobRecord
        Job record with a bundle path relative to the attempt directory.
    root : Path
        Attempt directory used to resolve the bundle path. The bundle must
        remain inside its parent bundle directory.
    verify_files : bool, default False
        Verify original dataset bytes when True, as required before resubmission.
        False allows inspection and collection when source datasets are absent.
    require_packaged : bool, default False
        Require the bundle's files field to contain packaged input/output
        locations. Use True for execution and result collection. These locations
        are relative to the attempt directory. False permits an unpackaged
        bundle, for example when inspecting its domain or preparing a retry.
        This check does not verify that the packaged files exist.

    Returns
    -------
    MLFFTrainBundle
        Shared bundle with its scientific identity checked against bundle_id.
        Its files field may be None unless require_packaged is True.

    Raises
    ------
    ValueError
        If the reference leaves the bundle directory, the bundle ID differs,
        the bundle or requested dataset verification is invalid, or packaged
        file locations are required but absent.
    OSError
        If the bundle or a dataset required for verification cannot be read.
    """
    context = {"verify_files": verify_files}
    relative = Path(record.relative_source_bundle_path)
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root.parent.resolve()):
        raise ValueError("Source bundle must be relative to the attempt and inside its bundle directory.")
    bundle = MLFFTrainBundle.model_validate_json(path.read_text(encoding="utf-8"), context=context)
    if str(bundle.mlff_train_bundle_id) != record.bundle_id:
        raise ValueError("Referenced bundle ID differs from the attempt's bundle ID.")
    if require_packaged and bundle.files is None:
        raise ValueError("Attempt bundle is not packaged with input and output file specifications.")
    return bundle


def get_dataset_and_model_hashes(bundle: MLFFTrainBundle) -> dict[str, str]:
    """Map packaged dataset and pretrained-model paths to recorded SHA-256 hashes.

    Parameters
    ----------
    bundle : MLFFTrainBundle
        Packaged bundle. Legacy datasets without hashes are omitted.

    Returns
    -------
    dict[str, str]
        Bundle-relative model and dataset paths mapped to SHA-256 values.
        Configuration files and scripts are excluded; no files are read.

    Raises
    ------
    ValueError
        If the bundle has not been packaged.
    """
    files = bundle.files
    if files is None:
        raise ValueError("Bundle has not been packaged.")
    unit = bundle.training_unit
    result = {files.pretrained_model: bundle.mlff_spec.pretrained_model.artifacts["model"].sha256}
    for key, name in (("train", unit.train_set), ("validation", unit.val_set)):
        if name in unit.dataset_sha256:
            result[files.training_input_files[key]] = unit.dataset_sha256[name]
    for index, name in enumerate(unit.test_sets):
        if name in unit.dataset_sha256:
            result[files.test_input_files[f"test_{index:03d}"]] = unit.dataset_sha256[name]
    return result


def verify_dataset_and_model_hashes(bundle: MLFFTrainBundle, root: Path) -> None:
    """Verify hash of copied inputs without opening source-machine paths.

    Parameters
    ----------
    bundle : MLFFTrainBundle
        Packaged bundle loaded with verify_files=False validation context.
    root : Path
        Directory containing bundle.json and the copied inputs.

    Raises
    ------
    ValueError
        If an input's bytes no longer match the scientific bundle.
    OSError
        If a copied input cannot be read.
    """
    for name, expected in get_dataset_and_model_hashes(bundle).items():
        if file_sha256(root / name) != expected:
            raise ValueError(f"Packaged input changed: {name}")
