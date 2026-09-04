"""SevenNet submit-folder writer."""

from __future__ import annotations

from temper.mlff.bundle_writers.base import (
    BaseMLFFBundleWriter,
    command,
    yaml_text,
)
from temper.utils.defaults import (
    DEFAULT_MLFF_DATASETS_DIR,
    DEFAULT_MLFF_TRAINING_DIR,
)


_MODEL = {
    "chemical_species": "Auto",
    "cutoff": 5.0,
    "channel": 128,
    "is_parity": False,
    "lmax": 2,
    "num_convolution_layer": 5,
    "irreps_manual": [
        "128x0e",
        "128x0e+64x1e+32x2e",
        "128x0e+64x1e+32x2e",
        "128x0e+64x1e+32x2e",
        "128x0e+64x1e+32x2e",
        "128x0e",
    ],
    "weight_nn_hidden_neurons": [64, 64],
    "radial_basis": {
        "radial_basis_name": "bessel",
        "bessel_basis_num": 8,
    },
    "cutoff_function": {
        "cutoff_function_name": "XPLOR",
        "cutoff_on": 4.5,
    },
    "self_connection_type": "linear",
}


class SevenNetBundleWriter(BaseMLFFBundleWriter):
    """Write fixed-layout SevenNet 0.13.0 train-and-test bundles."""

    mlff_type = "sevennet"
    calculator_resource = "calculators/sevennet.py"
    model_filenames = {"model": "sevennet.pth"}
    trained_model_filename = "sevennet.pth"

    def generated_training_files(self, training_stress: bool) -> dict[str, str]:
        parameters = dict(self.spec.training or {})
        epoch = parameters.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
            raise ValueError(
                "SevenNet training parameter 'epoch' must be a positive integer."
            )
        model = dict(_MODEL)
        model["train_shift_scale"] = parameters.pop("train_shift_scale")
        model["train_denominator"] = parameters.pop("train_denominator")
        batch_size = parameters.pop("batch_size")
        data_divide_ratio = parameters.pop("data_divide_ratio")

        train = parameters
        train["is_train_stress"] = training_stress
        train["force_loss_weight"] = 1.0
        train["stress_loss_weight"] = 0.01 if training_stress else 0.0
        train["error_record"] = [
            ["Energy", "RMSE"],
            ["Force", "RMSE"],
        ]
        if training_stress:
            train["error_record"].append(["Stress", "RMSE"])
        train["error_record"].append(["TotalLoss", "None"])
        train["continue"] = {
            "reset_optimizer": True,
            "reset_scheduler": True,
            "reset_epoch": True,
            "checkpoint": self.artifact_path("model"),
        }
        data = {
            "batch_size": batch_size,
            "data_divide_ratio": data_divide_ratio,
            "data_format_args": {"index": ":"},
            "load_trainset_path": [
                f"{DEFAULT_MLFF_DATASETS_DIR}/train.extxyz"
            ],
        }
        if self.unit.val_set is not None:
            data["load_validset_path"] = [
                f"{DEFAULT_MLFF_DATASETS_DIR}/validation.extxyz"
            ]
        return {
            f"{DEFAULT_MLFF_TRAINING_DIR}/sevennet.yaml": yaml_text(
                {"model": model, "train": train, "data": data}
            )
        }

    def training_lines(self, training_stress: bool) -> tuple[str, ...]:
        del training_stress
        epoch = (self.spec.training or {})["epoch"]
        work = f"{DEFAULT_MLFF_TRAINING_DIR}/sevennet"
        return (
            command("mkdir", "-p", work),
            command(
                "sevenn",
                "train",
                f"{DEFAULT_MLFF_TRAINING_DIR}/sevennet.yaml",
                "-s",
                "-w",
                work,
            ),
            command(
                "cp",
                f"{work}/checkpoint_{epoch}.pth",
                self.trained_model_path,
            ),
        )


__all__ = ["SevenNetBundleWriter"]
