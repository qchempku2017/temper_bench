"""MatterSim submit-folder writer."""

from __future__ import annotations

import shlex
from typing import Any

from temper.mlff.bundle_writers.base import BaseMLFFBundleWriter, command
from temper.utils.defaults import (
    DEFAULT_MLFF_DATASETS_DIR,
    DEFAULT_MLFF_RUNTIME_DIR,
    DEFAULT_MLFF_TRAINING_DIR,
)


_BOOLEAN_OPTIONS = {
    "include_forces",
    "include_stresses",
    "re_normalize",
    "trainable_scale",
    "trainable_shift",
}


class MatterSimBundleWriter(BaseMLFFBundleWriter):
    """Write fixed-layout MatterSim 1.2.5 train-and-test bundles."""

    mlff_type = "mattersim"
    calculator_resource = "calculators/mattersim.py"
    model_filenames = {"model": "mattersim.pth"}
    trained_model_filename = "mattersim.pth"

    def extra_runtime_resources(self) -> dict[str, str]:
        return {"device.py": "device.py"}

    @staticmethod
    def _options(parameters: dict[str, Any]) -> list[str]:
        result: list[str] = []
        for key, value in parameters.items():
            option = f"--{key}"
            if key in _BOOLEAN_OPTIONS:
                result.append(option if value else f"--no-{key}")
            elif value is not None:
                result.extend((option, str(value)))
        return result

    def training_lines(self, training_stress: bool) -> tuple[str, ...]:
        parameters = dict(self.spec.training or {})
        epochs = parameters.get("epochs")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise ValueError(
                "MatterSim training parameter 'epochs' must be a positive integer."
            )
        parameters["early_stop_patience"] = epochs + 1
        parameters["batch_size"] = 1
        parameters["include_forces"] = True
        parameters["include_stresses"] = training_stress
        parameters["force_loss_ratio"] = 1.0
        parameters["stress_loss_ratio"] = 0.1 if training_stress else 0.0
        device_script = f"{DEFAULT_MLFF_RUNTIME_DIR}/device.py"
        resolve = (
            'MLFF_DEVICE="$("$PYTHON_BIN" '
            + shlex.quote(device_script)
            + ' torch --warn-mattersim)"'
        )
        work = f"{DEFAULT_MLFF_TRAINING_DIR}/mattersim"
        arguments = [
            "$PYTHON_BIN",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=1",
            "--module",
            "mattersim.training.finetune_mattersim",
            "--train_data_path",
            f"{DEFAULT_MLFF_DATASETS_DIR}/train.extxyz",
            "--load_model_path",
            self.artifact_path("model"),
            "--save_path",
            work,
            "--save_checkpoint",
            "--device",
            "$MLFF_DEVICE",
            *self._options(parameters),
        ]
        if self.unit.val_set is not None:
            arguments.extend(
                (
                    "--valid_data_path",
                    f"{DEFAULT_MLFF_DATASETS_DIR}/validation.extxyz",
                )
            )
        return (
            resolve,
            command("mkdir", "-p", work),
            command(*arguments),
            command(
                "cp", f"{work}/last_model.pth", self.trained_model_path
            ),
        )


__all__ = ["MatterSimBundleWriter"]
