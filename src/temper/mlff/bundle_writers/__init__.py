"""Write portable MLFF bundles for installed TEMPER runtimes."""
from pathlib import Path

from temper.mlff.bundle_writers.base import MLFFBundleWriter
from temper.schemas.mlff_train_bundle import MLFFTrainBundle


def write_submit_folder(bundle: MLFFTrainBundle, target_dir: str | Path | None = None) -> Path:
    """Package an experiment for the installed TEMPER runtime.

    Parameters
    ----------
    bundle : MLFFTrainBundle
        Recipe and local input references to copy, without modifying the object.
    target_dir : str, Path, or None
        New destination directory. None creates a caller-owned temporary one.

    Returns
    -------
    Path
        Absolute destination path, with this default layout::

            bundle.json              # recipe, input mapping, evaluation settings
            run.sh                   # prepare, fine-tune, and test in one command
            datasets/train.extxyz    # fine-tuning only
            datasets/validation.extxyz  # when supplied
            datasets/test_000.extxyz # one indexed file per test dataset
            models/<model files>

        Run ``bash run.sh`` remotely with TEMPER and the selected backend
        installed. It creates training files, artifacts, and prediction outputs;
        zero-shot bundles skip fine-tuning. Directory names use the configured
        defaults. No Python code is copied into the bundle.

    Raises
    ------
    FileExistsError
        If target_dir already exists.
    ValueError
        If input labels are inconsistent or a pretrained artifact has changed.
    OSError
        If inputs cannot be read or output files cannot be written.
    """
    return MLFFBundleWriter(bundle).write_submit_folder(target_dir)


__all__ = ["MLFFBundleWriter", "write_submit_folder"]
