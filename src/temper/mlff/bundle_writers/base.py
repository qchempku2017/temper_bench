"""Package inputs and a portable experiment record for an installed runtime."""
from __future__ import annotations

import hashlib
from temper._version import __version__
from pathlib import Path
import shutil
import tempfile

from temper.mlff.pipeline.training_adapters import mlff_adapter_factory
from temper.schemas.mlff_train_bundle import MLFFBundleFiles, MLFFTrainBundle
from temper.utils.defaults import DEFAULT_MLFF_DATASETS_DIR, DEFAULT_MLFF_MODELS_DIR


class MLFFBundleWriter:
    """Copy one bundle's inputs without changing its local source records.

    Parameters
    ----------
    bundle : MLFFTrainBundle
        Local experiment whose dataset and pretrained-model paths are readable.
    """

    def __init__(self, bundle: MLFFTrainBundle) -> None:
        self.bundle = bundle

    def write_submit_folder(self, target_dir: str | Path | None = None) -> Path:
        """Write copied inputs, bundle.json, and an end-to-end run.sh launcher.

        Parameters
        ----------
        target_dir : str, Path, or None
            New directory to create. None creates a caller-owned temporary one.

        Returns
        -------
        Path
            Absolute path to the written folder. See the module-level
            write_submit_folder function for its layout and execution command.
        """
        unit = self.bundle.training_unit
        spec = self.bundle.mlff_spec
        adapter = mlff_adapter_factory(spec.mlff_type)
        datasets = {
            filename: f"{DEFAULT_MLFF_DATASETS_DIR}/test_{index:03d}.extxyz"
            for index, filename in enumerate(unit.test_sets)
        }
        if unit.train_set is not None:
            datasets[unit.train_set] = f"{DEFAULT_MLFF_DATASETS_DIR}/train.extxyz"
        if unit.val_set is not None:
            datasets[unit.val_set] = f"{DEFAULT_MLFF_DATASETS_DIR}/validation.extxyz"
        files = MLFFBundleFiles(
            datasets=datasets,
            models={key: f"{DEFAULT_MLFF_MODELS_DIR}/{name}"
                    for key, name in adapter.model_filenames.items()},
            training_stress=unit.training_has_stress,
            test_stress=unit.test_has_stress,
            trained_model_filename=adapter.trained_model_filename,
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
            for filename, relative in files.datasets.items():
                self._copy(unit.dataset_source(filename), target / relative)
            for key, relative in files.models.items():
                artifact = spec.pretrained_model.artifacts[key]
                source = artifact.path.expanduser().resolve()
                with source.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest != artifact.sha256:
                    raise ValueError(f"Local MLFF artifact changed after the specification was built: {source}.")
                self._copy(source, target / relative)
            (target / "bundle.json").write_text(
                packaged.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n",
            )
            # One command calls pipeline.runner.run_pipeline: prepare, fine-tune when requested,
            # then run_test.run for every test dataset, including zero-shot.
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
        return target

    @staticmethod
    def _copy(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
