"""Prepare and execute a portable MLFF bundle using installed backend
adapters."""

from __future__ import annotations

from temper._version import __version__
from pathlib import Path
from typing import Literal
import logging
import warnings
import time

from temper.mlff.pipeline.training_adapters import mlff_adapter_factory
from temper.mlff.pipeline.training_adapters.base import BaseMLFFAdapter
from temper.utils.cuda import require_cuda
from temper.mlff.pipeline.evaluate_mlff import run_test
from temper.schemas.mlff_train_bundle import MLFFTrainBundle
from temper.schemas.mlff_experiment_result import (
    MLFFExperimentManifest, MLFFTestManifest,
)
from temper.job_management.input_utils import verify_dataset_and_model_hashes
from temper.utils.file_hashing import file_sha256

logger = logging.getLogger(__name__)


def make_test_manifests(
    bundle: MLFFTrainBundle, *, bundle_root: str | Path | None = None,
) -> tuple[MLFFTestManifest, ...]:
    """Describe each planned test with its result fields unfilled.

    Paths remain relative to the directory containing bundle.json,
    so these records work after moving the packaged bundle. Dataset
    names retain their original exported filenames; checksums identify
    the corresponding reference labels. Shared calculator settings and
    model locations remain in this bundle.

    Parameters
    ----------
    bundle : MLFFTrainBundle
        Packaged experiment supplying test filenames, copied input and output
        paths, stress availability, and any recorded dataset checksums. Its
        files mapping must be populated. The bundle is not modified.
    bundle_root : str, pathlib.Path or None, optional
        Directory containing the packaged inputs. Only needed for older
        local bundles without recorded dataset checksums: missing hashes
        are calculated from the copied test files here. Expands ~ and
        resolves relative paths against the current working directory.
        None requires all test checksums to be recorded already.

    Returns
    -------
    tuple[MLFFTestManifest, ...]
        Independent test manifest objects pending test execution
        in TrainingUnit.test_sets order, with execution input and prediction
        paths, reference checksums, requested properties and units.
        Prediction checksums and measured durations are None until evaluation
        succeeds. No output files are created and this bundle is unchanged.

    Raises
    ------
    ValueError
        If the bundle is unpackaged, test stress flags have a different
        length from test_sets, or a checksum is missing and bundle_root
        was not supplied.
    KeyError
        If a packaged test input or prediction path is missing.
    OSError
        If a copied test file needed for hashing cannot be read.
    """
    if bundle.files is None:
        raise ValueError(
            "Bundle has not been packaged; call write_submit_folder first."
        )
    root = (
        None if bundle_root is None else Path(bundle_root).expanduser().resolve()
    )
    manifests: list[MLFFTestManifest] = []
    for filename, has_stress in zip(
        bundle.training_unit.test_sets,
        bundle.files.test_stress,
        strict=True,
    ):
        properties: list[Literal["energy", "forces", "stress"]] = [
            "energy", "forces",
        ]
        if has_stress:
            properties.append("stress")
        execution_input_path = bundle.files.test_input_files[filename]
        checksum = bundle.training_unit.dataset_sha256.get(filename)
        if checksum is None:
            if root is None:
                raise ValueError(
                    "bundle_root is required to hash a test dataset "
                    "without a recorded checksum."
                )
            checksum = file_sha256(root / execution_input_path)
        manifests.append(
            MLFFTestManifest(
                dataset_id=filename,
                source_domain=bundle.training_unit.domain,
                source_filename=filename,  # Original file name as in training_unit.test_sets.
                execution_input_path=execution_input_path,
                reference_sha256=checksum,
                requested_properties=tuple(properties),
                prediction_path=bundle.files.test_output_files[filename],
                units={
                    "energy": "eV",
                    "forces": "eV/Angstrom",
                    "stress": "eV/Angstrom^3" if has_stress else None,
                },
            )
        )
    return tuple(manifests)


def create_result_manifest(
    bundle: MLFFTrainBundle, root: Path,
) -> MLFFExperimentManifest:
    """Create an experiment manifest containing pending test instructions.

    The evaluator fills the test results in these records. The completed
    manifest later instructs the loader to assemble an MLFFExperimentResult
    from downloaded files.

    Parameters
    ----------
    bundle : MLFFTrainBundle
        Packaged experiment matching root/bundle.json.
    root : Path
        Execution directory containing bundle.json and copied test inputs.
        Expands ~ and resolves relative paths against the working directory.

    Returns
    -------
    MLFFExperimentManifest
        Pending manifest pointing to bundle.json, reference inputs and NPZ
        outputs, with original dataset names, requested properties and units.
        Evaluation later fills prediction checksums, durations and package
        versions. The caller writes it with MLFFExperimentManifest.to_json_file.

    Raises
    ------
    ValueError
        If the bundle is unpackaged or its test metadata is inconsistent.
    KeyError
        If the packaged mapping lacks a test input or prediction path.
    OSError
        If bundle.json or a reference without a recorded checksum is unreadable.
    """
    root = root.expanduser().resolve()
    tests = make_test_manifests(bundle, bundle_root=root)
    return MLFFExperimentManifest(
        bundle_sha256=file_sha256(root / "bundle.json"),
        training_status=(
            "pending" if bundle.unit_type == "finetune" else "not_applicable"
        ),
        test_manifests=tests,
    )


def write_configuration_files(
    adapter: BaseMLFFAdapter,
    *,
    missing_only: bool = False,
) -> None:
    """Write native training configs and scripts for an already populated
    bundle.

    Paths resolve below adapter.root using adapter.files. Copied
    datasets and weights must already exist there. No source inputs are
    copied and no backend is imported. This function never removes the
    bundle folder.
    Zero-shot bundles have no native training files, so nothing is written.

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

    Raises
    ------
    OSError
        If an output directory or native training file cannot be written.
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
        Preparation writes missing native training configs and scripts
        using copied inputs and paths from
        bundle.files. Existing files and manual edits are preserved on
        repeat calls. It does not recopy inputs. Output parent
        directories are created for execution. This function does not
        import a model backend or require CUDA. Backend data conversion
        and checkpoint loading occur in run_pipeline.

    Raises
    ------
    ValueError
        If the bundle is invalid or copied inputs no longer match their
        recorded checksums.
    OSError
        If bundle.json or copied inputs cannot be read, or native configs
        and output directories cannot be written.

    Warnings
    --------
    RuntimeWarning
        If the writer's TEMPER release differs from this runtime.
        Unsupported schema versions still fail schema validation.
    """
    path = Path(bundle_path).expanduser().resolve()
    logger.debug("Reading bundle from %s", path)
    bundle = MLFFTrainBundle.from_json_file(path, context={"verify_files": False})
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
        Writes native configs, training logs/models, per-dataset NPZ arrays
        and a single root results.json containing the execution manifest. The manifest
        is created before preparation and updated after each completed test.
        Zero-shot runs evaluate the pretrained model without training.
        Saves training_wall_time_seconds in the manifest immediately after
        adapter.train() succeeds. This includes loading,
        validation and saving inside that call, but excludes separate data
        preparation and evaluation. Failed or interrupted training has no
        completed duration; later evaluation failure preserves the value.
        A caught failure marks the calculation failed without removing finished
        tests. An abrupt process termination may leave the last saved running
        state; this is not proof that the process remains active.

    Raises
    ------
    RuntimeError
        If CUDA is unavailable or evaluation fails.
    subprocess.CalledProcessError
        If native training fails. Testing starts only after training
        succeeds.
    """
    path = Path(bundle_path).expanduser().resolve()
    bundle = MLFFTrainBundle.from_json_file(path, context={"verify_files": False})
    manifest = create_result_manifest(bundle, path.parent)
    manifest.status = "running"
    manifest.to_json_file(path.parent / "results.json")
    try:
        adapter = prepare_bundle(path)
        logger.info("Checking CUDA availability")
        require_cuda()
        if bundle.unit_type == "finetune":
            logger.info("Preparing training data for bundle %s", bundle.mlff_train_bundle_id)
            adapter.prepare_training()
            if adapter.spec.mlff_type == "mattersim":
                logger.info(
                    "MatterSim device/batching fix status cannot be inferred from"
                    " its version. Use the audited dependency build and two-frame"
                    " GPU validation in docs/deployment/README.md."
                )
            logger.info("Training bundle %s (%s)", bundle.mlff_train_bundle_id, adapter.spec.mlff_type)
            manifest.training_status = "running"
            manifest.to_json_file(path.parent / "results.json")
            try:
                started = time.perf_counter()
                adapter.train()
                elapsed = time.perf_counter() - started
            except BaseException:
                manifest.training_status = "failed"
                logger.error(
                    "Training bundle %s failed; backend log: %s",
                    bundle.mlff_train_bundle_id,
                    adapter.root / adapter.output_path("training_log"),
                )
                raise
            manifest.training_status = "completed"
            manifest.training_wall_time_seconds = elapsed
            manifest.to_json_file(path.parent / "results.json")
        logger.info("Evaluating bundle %s (%s)", bundle.mlff_train_bundle_id, adapter.spec.mlff_type)
        try:
            adapter.prepare_evaluation()
            run_test(
                manifest, bundle_root=adapter.root,
            )
        finally:
            adapter.cleanup_evaluation()
        if any(result.status != "completed" for result in manifest.test_manifests):
            raise RuntimeError("Evaluation returned with unfinished test datasets.")
        manifest.status = "completed"
        manifest.to_json_file(path.parent / "results.json")
    except BaseException:
        manifest.status = "failed"
        manifest.to_json_file(path.parent / "results.json")
        raise
    logger.info("Completed bundle %s in %s", bundle.mlff_train_bundle_id, path.parent)
