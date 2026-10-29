"""Prepare and execute a portable MLFF bundle using installed backend
adapters."""

from __future__ import annotations

from temper._version import __version__
from pathlib import Path
import json
import logging
import warnings

from temper.mlff.pipeline.training_adapters import mlff_adapter_factory
from temper.mlff.pipeline.training_adapters.base import BaseMLFFAdapter
from temper.mlff.pipeline.cuda_utils import require_cuda
from temper.mlff.pipeline.evaluate_mlff import run_test
from temper.schemas.mlff_train_bundle import MLFFTrainBundle
from temper.job_management.input_utils import verify_dataset_and_model_hashes

logger = logging.getLogger(__name__)


def write_configuration_files(
    adapter: BaseMLFFAdapter,
    *,
    missing_only: bool = False,
) -> None:
    """Write configs and training scripts for an already populated
    bundle.

    Paths resolve below adapter.root using adapter.files. Copied
    datasets and weights must already exist there. No source inputs are
    copied and no backend is imported. This function never removes the
    bundle folder.

    Parameters
    ----------
    adapter : BaseMLFFAdapter
        Adapter holding the loaded bundle and its absolute root. The
        bundle's files field must be populated with the expected layout
        for the adapter to be able to perform training and evaluation.
    missing_only : bool, default False
        If true, preserves existing files, including manual edits, and
        creates only missing configs. When all expected files exist,
        their contents are not rendered again. False rewrites configs
        and scripts from the bundle, for example after changing recorded
        paths.
    """
    files = adapter.files
    assert files is not None
    contents: dict[str, str] = {}
    # Write training configs and scripts for fine-tuning bundles.
    if adapter.bundle.unit_type == "finetune":
        native_paths = [
            files.training_input_files[key]
            for key in ("config", "script")
            if key in files.training_input_files
        ]
        if not missing_only or any(
            not (adapter.root / path).exists() for path in native_paths
        ):
            contents.update(
                adapter.training_file_contents(bool(files.training_stress))
            )
    # Write test config file.
    config_path = files.test_input_files["config"]
    if not missing_only or not (adapter.root / config_path).exists():
        contents[config_path] = (
            json.dumps(adapter.bundle.test_config(), indent=2) + "\n"
        )
    for relative, content in contents.items():
        destination = adapter.root / relative
        if missing_only and destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8", newline="\n")
        logger.debug("Wrote configuration input %s", destination)


def prepare_bundle(bundle_path: str | Path) -> BaseMLFFAdapter:
    """Load an experiment and prepare its directories without executing
    it.

    Parameters
    ----------
    bundle_path : str or Path
        Path to bundle.json. Packaged paths are relative to its parent
        directory.

    Returns
    -------
    BaseMLFFAdapter
        Adapter holding the loaded bundle and its absolute root.
        Preparation writes missing native configs, training scripts, and
        test_config.json using copied inputs and paths from
        bundle.files. Existing files and manual edits are preserved on
        repeat calls. It does not recopy inputs. Output parent
        directories are created for execution. This function does not
        import a model backend or require CUDA. Backend data conversion
        and checkpoint loading occur in run_pipeline.

    Warnings
    -----
    RuntimeWarning
        If the writer's TEMPER release differs from this runtime.
        Unsupported schema versions still fail schema validation.
    """
    path = Path(bundle_path).expanduser().resolve()
    logger.debug("Reading bundle from %s", path)
    bundle = MLFFTrainBundle.model_validate_json(
        path.read_text(encoding="utf-8"), context={"verify_files": False}
    )
    verify_dataset_and_model_hashes(bundle, path.parent)

    installed = __version__
    if bundle.temper_version != installed:
        warnings.warn(
            f"Bundle was written by temper-bench {bundle.temper_version}; "
            f"running {installed}. Compatibility is not guaranteed.",
            RuntimeWarning,
            stacklevel=2,
        )
    logger.info(
        "Preparing bundle %s (%s)",
        bundle.mlff_train_bundle_id,
        bundle.mlff_spec.mlff_type,
    )
    logger.debug("Selected MLFF adapter %s", bundle.mlff_spec.mlff_type)
    adapter = mlff_adapter_factory(bundle.mlff_spec.mlff_type)(
        bundle, path.parent
    )
    files = adapter.files
    assert files is not None
    for relative in (
        *files.training_input_files.values(),
        *files.training_output_files.values(),
        *files.test_input_files.values(),
        *files.test_output_files.values(),
        *(
            [files.evaluation_model]
            if files.evaluation_model is not None
            else []
        ),
    ):
        (adapter.root / relative).parent.mkdir(parents=True, exist_ok=True)
    write_configuration_files(adapter, missing_only=True)
    return adapter


def run_pipeline(bundle_path: str | Path) -> None:
    """Prepare inputs, fine-tune when requested, and evaluate every test
    dataset.

    Parameters
    ----------
    bundle_path : str or Path
        Path to the experiment's bundle.json, usable from any working
        directory.

    Returns
    -------
    None
        Writes native configs, training logs/models, and per-dataset
        prediction arrays and JSON metadata under the bundle's
        configured directories. Zero-shot runs evaluate the pretrained
        model without a training stage.

    Raises
    ------
    RuntimeError
        If CUDA is unavailable or evaluation fails.
    subprocess.CalledProcessError
        If native training fails. Testing starts only after training
        succeeds.
    """
    adapter = prepare_bundle(bundle_path)
    logger.info("Checking CUDA availability")
    require_cuda()
    if adapter.bundle.unit_type == "finetune":
        logger.info(
            "Preparing training data for bundle %s",
            adapter.bundle.mlff_train_bundle_id,
        )
        adapter.prepare_training()
        if adapter.spec.mlff_type == "mattersim":
            logger.info(
                "MatterSim device/batching fix status cannot be inferred from"
                " its version. Use the audited dependency build and two-frame"
                " GPU validation in docs/deployment-compatibility.md."
            )
        logger.info(
            "Training bundle %s (%s)",
            adapter.bundle.mlff_train_bundle_id,
            adapter.spec.mlff_type,
        )
        try:
            adapter.train()
        except Exception:
            logger.error(
                "Training bundle %s failed; backend log: %s",
                adapter.bundle.mlff_train_bundle_id,
                adapter.root / adapter.output_path("training_log"),
            )
            raise
    logger.info(
        "Evaluating bundle %s (%s)",
        adapter.bundle.mlff_train_bundle_id,
        adapter.spec.mlff_type,
    )
    try:
        adapter.prepare_evaluation()
        assert adapter.files is not None
        run_test(
            adapter.root / adapter.files.test_input_files["config"],
            bundle_root=adapter.root,
        )
    finally:
        adapter.cleanup_evaluation()
    logger.info(
        "Completed bundle %s in %s",
        adapter.bundle.mlff_train_bundle_id,
        adapter.root,
    )
