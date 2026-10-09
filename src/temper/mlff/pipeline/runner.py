"""Prepare and execute a portable MLFF bundle using installed backend adapters."""
from __future__ import annotations

from temper._version import __version__
import json
from pathlib import Path
import shlex
import warnings

from temper.mlff.pipeline.training_adapters import mlff_adapter_factory
from temper.mlff.pipeline.training_adapters.base import BaseMLFFAdapter
from temper.mlff.pipeline.cuda_utils import require_cuda
from temper.mlff.pipeline import run_test
from temper.schemas.mlff_train_bundle import MLFFTrainBundle


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
        writes test_config.json and, for fine-tuning, native configs plus
        training/run.sh for CLI backends. TorchNEP uses a direct Python call.
        This function does not import a model backend or require CUDA.
        Backend data conversion and checkpoint loading occur in run_pipeline.

    Warns
    -----
    RuntimeWarning
        If the writer's TEMPER release differs from this runtime. Unsupported
        schema versions still fail schema validation.
    """
    path = Path(bundle_path).expanduser().resolve()
    bundle = MLFFTrainBundle.model_validate_json(path.read_text(encoding="utf-8"))

    installed = __version__
    if bundle.temper_version != installed:
        warnings.warn(
            f"Bundle was written by temper-bench {bundle.temper_version}; "
            f"running {installed}. Compatibility is not guaranteed.",
            RuntimeWarning, stacklevel=2,
        )
    adapter = mlff_adapter_factory(bundle.mlff_spec.mlff_type)(bundle, path.parent)
    files = adapter.files
    for directory in (files.artifacts_dir, files.outputs_dir):
        (adapter.root / directory).mkdir(parents=True, exist_ok=True)
    if bundle.unit_type == "finetune":
        stress = bool(files.training_stress)
        generated = adapter.generated_training_files(stress)
        # Keep the existing native commands, with Bash's fail-fast semantics.
        # Now training scripts are generated at remote.
        lines = adapter.training_lines(stress)
        if lines:
            generated[f"{files.training_dir}/run.sh"] = (
                "#!/usr/bin/env bash\nset -euo pipefail\n{\n"
                + "\n".join(lines)
                + "\n} 2>&1 | tee " + shlex.quote(f"{files.outputs_dir}/training.log") + "\n"
            )
        for relative, content in generated.items():
            destination = adapter.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8", newline="\n")
    (adapter.root / "test_config.json").write_text(
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
    require_cuda()
    if adapter.bundle.unit_type == "finetune":
        adapter.prepare_training()
        if adapter.mlff_type == "mattersim":
            warnings.warn(
                "MatterSim 1.2.5 has unresolved CUDA fine-tuning and batch-index "
                "issues (microsoft/mattersim#163). See docs/mlff-bundles.md.",
                RuntimeWarning,
            )
        adapter.train()
    run_test.run(adapter.root / "test_config.json")
