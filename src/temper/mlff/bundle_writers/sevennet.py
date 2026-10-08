"""SevenNet submit-folder writer."""

from __future__ import annotations

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


@BaseMLFFBundleWriter.register(name="sevennet")
class SevenNetBundleWriter(BaseMLFFBundleWriter):
    """Write fixed-layout SevenNet 0.13.0 train-and-test bundles."""

    mlff_type = "sevennet"
    calculator_resource = "calculators/sevennet.py"
    model_filenames = {"model": "sevennet.pth"}
    trained_model_filename = "finetuned_sevennet.pth"

    def extra_runtime_resources(self) -> dict[str, str]:
        """Return the checkpoint-based native YAML preparation script."""
        return {"prepare_sevennet.py": "data_preparation/sevennet.py"}

    def generated_training_files(self, training_stress: bool) -> dict[str, str]:
        """Return YAML that fine-tunes SevenNet-0 via train.continue.

        The checkpoint supplies pretrained weights and species statistics;
        optimizer, scheduler and epoch counters restart for this dataset.
        The runner reads architecture from the checkpoint before training.
        """
        parameters = dict(self.spec.training_parameters or {})
        model = {}
        model["train_shift_scale"] = parameters.pop("train_shift_scale")
        model["train_denominator"] = parameters.pop("train_denominator")
        batch_size = parameters.pop("batch_size")
        data_divide_ratio = parameters.pop("data_divide_ratio")

        train = parameters
        train["device"] = "cuda"
        train["is_train_stress"] = training_stress
        if not training_stress:
            train["stress_loss_weight"] = 0.0
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
        if self.training_unit.val_set is not None:
            data["load_validset_path"] = [
                f"{DEFAULT_MLFF_DATASETS_DIR}/validation.extxyz"
            ]
        return {
            f"{DEFAULT_MLFF_TRAINING_DIR}/sevennet.yaml": yaml_text(
                {"model": model, "train": train, "data": data}
            )
        }

    def training_lines(self, training_stress: bool) -> tuple[str, ...]:
        """Return the native continuation command and final checkpoint copy."""
        del training_stress
        epoch = (self.spec.training_parameters or {})["epoch"]
        work = f"{DEFAULT_MLFF_TRAINING_DIR}/sevennet"
        return (
            command("mkdir", "-p", work),
            command(
                "$PYTHON_BIN",
                f"{DEFAULT_MLFF_RUNTIME_DIR}/prepare_sevennet.py",
                "--config", f"{DEFAULT_MLFF_TRAINING_DIR}/sevennet.yaml",
                "--output", f"{DEFAULT_MLFF_TRAINING_DIR}/sevennet_resolved.yaml",
            ),
            command(
                "sevenn",
                "train",
                f"{DEFAULT_MLFF_TRAINING_DIR}/sevennet_resolved.yaml",
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
