"""Package inputs and a portable experiment record for an installed runtime."""
from __future__ import annotations

import hashlib
import logging
from temper._version import __version__
from pathlib import Path
import shutil
import tempfile

from temper.mlff.pipeline.training_adapters import mlff_adapter_factory
from temper.schemas.mlff_train_bundle import MLFFBundleFiles, MLFFTrainBundle
from temper.utils.defaults import (
    DEFAULT_MLFF_MODELS_DIR, DEFAULT_MLFF_TRAINING_INPUTS_DIR,
    DEFAULT_MLFF_TRAINING_OUTPUTS_DIR, DEFAULT_MLFF_TESTING_INPUTS_DIR,
    DEFAULT_MLFF_TESTING_OUTPUTS_DIR,
)

logger = logging.getLogger(__name__)


# Comment: maybe bundle writer can take in bundle and adapter as input. Adapter should only generate
#   file contents rather than being responsible for writing and copying files. These should be handled
#   by the bundle writer.
class MLFFBundleWriter:
    """Copy one bundle's inputs without changing its local source records.

    Parameters
    ----------
    bundle : MLFFTrainBundle
        Local experiment containing dataset and pretrained-model paths. Can be read and uploaded to
        remote executor.
    """

    def __init__(self, bundle: MLFFTrainBundle) -> None:
        self.bundle = bundle

    def write_submit_folder(self, target_dir: str | Path | None = None) -> tuple[Path, MLFFTrainBundle]:
        """Write copied inputs, bundle.json, and an end-to-end run.sh launcher.

        Parameters
        ----------
        target_dir : str, Path, or None
            New directory to create. None creates a caller-owned temporary one.

        Returns
        -------
        tuple[Path, MLFFTrainBundle]
            Absolute submit-directory path and independent packaged bundle.
            The returned record matches bundle.json; the source is unchanged.
        """
        unit = self.bundle.training_unit
        spec = self.bundle.mlff_spec
        adapter = mlff_adapter_factory(spec.mlff_type)
        logger.info(
            "Packaging bundle %s (%s, %s)",
            self.bundle.mlff_train_bundle_id, spec.mlff_type, self.bundle.unit_type,
        )
        finetune = self.bundle.unit_type == "finetune"
        training_inputs = {
            key: f"{DEFAULT_MLFF_TRAINING_INPUTS_DIR}/{name}"
            for key, name in adapter.training_input_filenames.items()
        } if finetune else {}
        if finetune:
            training_inputs["train"] = f"{DEFAULT_MLFF_TRAINING_INPUTS_DIR}/train.extxyz"
            if unit.val_set is not None:
                training_inputs["validation"] = f"{DEFAULT_MLFF_TRAINING_INPUTS_DIR}/validation.extxyz"
            else:
                training_inputs.pop("converted_validation", None)
        files = MLFFBundleFiles(
            pretrained_model=f"{DEFAULT_MLFF_MODELS_DIR}/{adapter.pretrained_model_filename}",
            training_input_files=training_inputs,
            training_output_files={
                key: f"{DEFAULT_MLFF_TRAINING_OUTPUTS_DIR}/{name}"
                for key, name in adapter.training_output_filenames.items()
            } if finetune else {},
            test_input_files={
                "config": f"{DEFAULT_MLFF_TESTING_INPUTS_DIR}/test_config.json",
                **{f"test_{i:03d}": f"{DEFAULT_MLFF_TESTING_INPUTS_DIR}/test_{i:03d}.extxyz"
                   for i in range(len(unit.test_sets))},
            },
            test_output_files={
                "summary": f"{DEFAULT_MLFF_TESTING_OUTPUTS_DIR}/summary.json",
                **{
                    f"test_{i:03d}_{kind}": f"{DEFAULT_MLFF_TESTING_OUTPUTS_DIR}/test_{i:03d}.{suffix}"
                    for i in range(len(unit.test_sets))
                    for kind, suffix in (("predictions", "npz"), ("metadata", "json"))
                },
            },
            evaluation_model=(
                f"{DEFAULT_MLFF_TRAINING_OUTPUTS_DIR}/{adapter.evaluation_model_filename}"
                if finetune and adapter.evaluation_model_filename is not None else None
            ),
            training_stress=unit.training_has_stress,
            test_stress=unit.test_has_stress,
        )
        packaged = self.bundle.model_copy(deep=True)
        packaged.files = files
        packaged.temper_version = __version__
        if target_dir is None:
            target = Path(tempfile.mkdtemp(prefix="temper-submit-"))
        else:
            target = Path(target_dir).expanduser().absolute()
            target.mkdir(parents=True, exist_ok=False)
        try:
            for i, filename in enumerate(unit.test_sets):
                self._copy(unit.dataset_source(filename), target / files.test_input_files[f"test_{i:03d}"])
            for key, filename in (("train", unit.train_set), ("validation", unit.val_set)):
                if filename is not None:
                    self._copy(unit.dataset_source(filename), target / files.training_input_files[key])
            artifact = spec.pretrained_model.artifacts["model"]
            source = artifact.path.expanduser().resolve()
            with source.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != artifact.sha256:
                raise ValueError(f"Local MLFF artifact changed after the specification was built: {source}.")
            self._copy(source, target / files.pretrained_model)
            (target / "bundle.json").write_text(
                packaged.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n",
            )
            # One command calls pipeline.runner.run_pipeline: prepare, fine-tune when requested,
            # then evaluate_mlff.run_test for every test dataset, including zero-shot.
            (target / "run.sh").write_text(
                '#!/usr/bin/env bash\nset -euo pipefail\n'
                'cd -- "$(dirname -- "$0")"\n'
                'exec temper_bench run_pipeline bundle.json\n',
                encoding="utf-8", newline="\n",
            )
            (target / "run.sh").chmod(0o755)
        except Exception:
            shutil.rmtree(target)
            raise

        logger.info("Wrote bundle %s to %s", packaged.mlff_train_bundle_id, target)
        return target, packaged

    @staticmethod
    def _copy(source: Path, destination: Path) -> None:
        logger.debug("Copying %s to %s", source, destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
