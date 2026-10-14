"""Write portable MLFF bundles for installed TEMPER runtimes."""
from pathlib import Path

from temper.mlff.bundle_writers.base import MLFFBundleWriter
from temper.schemas.mlff_train_bundle import MLFFTrainBundle


def write_submit_folder(bundle: MLFFTrainBundle, target_dir: str | Path | None = None) -> tuple[Path, MLFFTrainBundle]:
    """Package an experiment for the installed TEMPER runtime.

    Parameters
    ----------
    bundle : MLFFTrainBundle
        Recipe and local input references to copy, without modifying the object.
    target_dir : str, Path, or None
        New destination directory. None creates a caller-owned temporary one.

    Returns
    -------
    tuple[Path, MLFFTrainBundle]
        The first value is the absolute path to the new directory. The second
        is a separate bundle whose files field records the planned layout and
        whose contents match bundle.json. The original bundle is unchanged.

    Raises
    ------
    FileExistsError
        If target_dir already exists.
    ValueError
        If input labels are inconsistent or a pretrained artifact has changed.
    OSError
        If inputs cannot be read or output files cannot be written.

    Notes
    -----
    Packaging copies weights into pretrained_models/, training datasets into
    training_inputs/, and test datasets into testing_inputs/. It also writes
    bundle.json and run.sh at the root. Run
    ``bash run.sh`` on a host with TEMPER and the chosen backend installed.
    Runtime preparation writes missing native configs and scripts for fine-tuning
    and testing_inputs/test_config.json for every bundle, preserving existing
    files on repeat calls. Execution prepares backend-dependent inputs and writes results under
    training_outputs/ and testing_outputs/. Zero-shot runs skip training.
    Directory defaults can be configured before packaging; recorded paths are
    always relative to the bundle root. No backend Python code is copied.
    """
    return MLFFBundleWriter(bundle).write_submit_folder(target_dir)


__all__ = ["MLFFBundleWriter", "write_submit_folder"]
