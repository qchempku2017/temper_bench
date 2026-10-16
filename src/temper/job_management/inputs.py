"""Mappings for temporary copied inputs and their recorded content digests."""
from pathlib import Path
from temper.schemas.mlff_train_bundle import MLFFTrainBundle
from temper.utils.file_hashing import file_sha256


def input_hashes(bundle: MLFFTrainBundle) -> dict[str, str]:
    """Map packaged input filenames to their original content hashes.

    Parameters
    ----------
    bundle : MLFFTrainBundle
        Packaged recipe. Legacy datasets without hashes are omitted.

    Returns
    -------
    dict[str, str]
        Bundle-relative model and dataset paths mapped to SHA-256 values.

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


def verify_packaged_inputs(bundle: MLFFTrainBundle, root: Path) -> None:
    """Verify copied inputs without opening source-machine paths.

    Parameters
    ----------
    bundle : MLFFTrainBundle
        Packaged recipe loaded with verify_files=False validation context.
    root : Path
        Directory containing bundle.json and the copied inputs.

    Raises
    ------
    ValueError
        If an input's bytes no longer match the scientific recipe.
    OSError
        If a copied input cannot be read.
    """
    for name, expected in input_hashes(bundle).items():
        if file_sha256(root / name) != expected:
            raise ValueError(f"Packaged input changed: {name}")
