"""MACE submit-folder writer."""

from __future__ import annotations

import shlex

from temper.mlff.bundle_writers.base import (
    BaseMLFFBundleWriter,
    command,
    yaml_text,
)
from temper.utils.defaults import (
    DEFAULT_MLFF_DATASETS_DIR,
    DEFAULT_MLFF_RUNTIME_DIR,
    DEFAULT_MLFF_TRAINING_DIR,
)


class MACEBundleWriter(BaseMLFFBundleWriter):
    """Write fixed-layout mace-torch 0.3.16 train-and-test bundles."""

    mlff_type = "mace"
    calculator_resource = "calculators/mace.py"
    model_filenames = {"model": "mace.model"}
    trained_model_filename = "mace.model"

    def extra_runtime_resources(self) -> dict[str, str]:
        return {"device.py": "device.py"}

    def generated_training_files(self, training_stress: bool) -> dict[str, str]:
        config = dict(self.spec.training or {})
        epochs = config.get("max_num_epochs")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise ValueError(
                "MACE training parameter 'max_num_epochs' must be a positive "
                "integer."
            )
        config["patience"] = epochs + 1
        work = f"{DEFAULT_MLFF_TRAINING_DIR}/mace"
        config.update(
            {
                "name": "temper_model",
                "foundation_model": self.artifact_path("model"),
                "train_file": f"{DEFAULT_MLFF_DATASETS_DIR}/train.extxyz",
                "model_dir": f"{work}/models",
                "checkpoints_dir": f"{work}/checkpoints",
                "results_dir": f"{work}/results",
                "log_dir": f"{work}/logs",
                "energy_weight": 1.0,
                "forces_weight": 1.0,
                "stress_weight": 1.0 if training_stress else 0.0,
            }
        )
        if self.unit.val_set is not None:
            config["valid_file"] = (
                f"{DEFAULT_MLFF_DATASETS_DIR}/validation.extxyz"
            )
            config.pop("valid_fraction", None)
        return {f"{DEFAULT_MLFF_TRAINING_DIR}/mace.yaml": yaml_text(config)}

    def training_lines(self, training_stress: bool) -> tuple[str, ...]:
        del training_stress
        device_script = f"{DEFAULT_MLFF_RUNTIME_DIR}/device.py"
        resolve = (
            'MLFF_DEVICE="$("$PYTHON_BIN" '
            + shlex.quote(device_script)
            + ' torch --mps)"'
        )
        work = f"{DEFAULT_MLFF_TRAINING_DIR}/mace"
        return (
            resolve,
            command("mkdir", "-p", f"{work}/models"),
            command(
                "mace_run_train",
                "--config",
                f"{DEFAULT_MLFF_TRAINING_DIR}/mace.yaml",
                "--device",
                "$MLFF_DEVICE",
            ),
            command(
                "cp",
                f"{work}/models/temper_model.model",
                self.trained_model_path,
            ),
        )


__all__ = ["MACEBundleWriter"]
