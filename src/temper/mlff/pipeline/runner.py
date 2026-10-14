"""Prepare and execute a portable MLFF bundle using installed backend adapters."""
from __future__ import annotations

from temper._version import __version__
import json
from pathlib import Path
import logging
import warnings

from temper.mlff.pipeline.training_adapters import mlff_adapter_factory
from temper.mlff.pipeline.training_adapters.base import BaseMLFFAdapter
from temper.mlff.pipeline.cuda_utils import require_cuda
from temper.mlff.pipeline.evaluate_mlff import run_test
from temper.schemas.mlff_train_bundle import MLFFTrainBundle


logger = logging.getLogger(__name__)


def prepare_bundle(bundle_path: str | Path) -> BaseMLFFAdapter:
    """Load an experiment and write inspectable configs without executing it.

    Parameters
    ----------
    bundle_path : str or Path
        Path to bundle.json. Packaged paths are relative to its parent directory.

    Returns
    -------
    BaseMLFFAdapter
        Adapter holding the loaded bundle and its absolute root. Preparation
        writes testing_inputs/test_config.json and, for fine-tuning, native
        configs plus training_inputs/run.sh for CLI backends. Actual paths
        come from bundle.files. TorchNEP uses a direct Python call.
        This function does not import a model backend or require CUDA.
        Backend data conversion and checkpoint loading occur in run_pipeline.

    Warns
    -----
    RuntimeWarning
        If the writer's TEMPER release differs from this runtime. Unsupported
        schema versions still fail schema validation.
    """
    path = Path(bundle_path).expanduser().resolve()
    logger.debug("Reading bundle from %s", path)
    bundle = MLFFTrainBundle.model_validate_json(path.read_text(encoding="utf-8"))

    installed = __version__
    if bundle.temper_version != installed:
        warnings.warn(
            f"Bundle was written by temper-bench {bundle.temper_version}; "
            f"running {installed}. Compatibility is not guaranteed.",
            RuntimeWarning, stacklevel=2,
        )
    logger.info("Preparing bundle %s (%s)", bundle.mlff_train_bundle_id, bundle.mlff_spec.mlff_type)
    logger.debug("Selected MLFF adapter %s", bundle.mlff_spec.mlff_type)
    adapter = mlff_adapter_factory(bundle.mlff_spec.mlff_type)(bundle, path.parent)
    files = adapter.files
    assert files is not None
    for relative in (
        *files.training_input_files.values(), *files.training_output_files.values(),
        *files.test_input_files.values(), *files.test_output_files.values(),
        *([files.evaluation_model] if files.evaluation_model is not None else []),
    ):
        (adapter.root / relative).parent.mkdir(parents=True, exist_ok=True)
    if bundle.unit_type == "finetune":
        stress = bool(files.training_stress)
        adapter.generate_training_files(stress)
        adapter.generate_training_script(stress)
    # Comment: Writing of test_config.json should be handled by BundleWriter, not here.
    (adapter.root / files.test_input_files["config"]).write_text(
        json.dumps(bundle.test_config(), indent=2) + "\n", encoding="utf-8",
    )
    return adapter


def run_pipeline(bundle_path: str | Path) -> None:
    """Prepare inputs, fine-tune when requested, and evaluate every test dataset.

    Parameters
    ----------
    bundle_path : str or Path
        Path to the experiment's bundle.json, usable from any working directory.

    Returns
    -------
    None
        Writes native configs, training logs/models, and per-dataset prediction
        arrays and JSON metadata under the bundle's configured directories.
        Zero-shot runs evaluate the pretrained model without a training stage.

    Raises
    ------
    RuntimeError
        If CUDA is unavailable or evaluation fails.
    subprocess.CalledProcessError
        If native training fails. Testing starts only after training succeeds.
    """
    adapter = prepare_bundle(bundle_path)
    logger.info("Checking CUDA availability")
    require_cuda()
    if adapter.bundle.unit_type == "finetune":
        logger.info("Preparing training data for bundle %s", adapter.bundle.mlff_train_bundle_id)
        adapter.prepare_training()
        if adapter.spec.mlff_type == "mattersim":
            warnings.warn(
                "MatterSim 1.2.5 has unresolved CUDA fine-tuning and batch-index "
                "issues (microsoft/mattersim#163). See docs/mlff-bundles.md.",
                RuntimeWarning,
            )
        logger.info("Training bundle %s (%s)", adapter.bundle.mlff_train_bundle_id, adapter.spec.mlff_type)
        try:
            adapter.train()
        except Exception:
            logger.error("Training bundle %s failed; backend log: %s",
                         adapter.bundle.mlff_train_bundle_id,
                         adapter.root / adapter.output_path("training_log"))
            raise
    logger.info("Evaluating bundle %s (%s)", adapter.bundle.mlff_train_bundle_id, adapter.spec.mlff_type)
    try:
        adapter.prepare_evaluation()
        assert adapter.files is not None
        run_test(adapter.root / adapter.files.test_input_files["config"], bundle_root=adapter.root)
    finally:
        adapter.cleanup_evaluation()
    logger.info("Completed bundle %s in %s", adapter.bundle.mlff_train_bundle_id, adapter.root)
