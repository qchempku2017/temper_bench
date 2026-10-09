"""SevenNet runtime adapter."""

from __future__ import annotations

from temper.mlff.pipeline.training_adapters.base import (
    BaseMLFFAdapter,
    command,
    yaml_text,
)


@BaseMLFFAdapter.register(name="sevennet")
class SevenNetAdapter(BaseMLFFAdapter):
    """Prepare SevenNet 0.13.0 train-and-test bundles."""

    mlff_type = "sevennet"
    calculator_module = "temper.mlff.pipeline.calculators.sevennet"
    model_filenames = {"model": "sevennet.pth"}
    trained_model_filename = "finetuned_sevennet.pth"

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
                self.dataset_path(self.training_unit.train_set)
            ],
        }
        if self.training_unit.val_set is not None:
            data["load_validset_path"] = [
                self.dataset_path(self.training_unit.val_set)
            ]
        return {
            f"{self.files.training_dir}/sevennet.yaml": yaml_text(
                {"model": model, "train": train, "data": data}
            )
        }

    def prepare_training(self) -> None:
        """Resolve the checkpoint architecture before native SevenNet training."""
        from temper.mlff.pipeline.data_preparation.sevennet import prepare_config

        prepare_config(
            self.root / self.files.training_dir / "sevennet.yaml",
            self.root / self.files.training_dir / "sevennet_resolved.yaml",
            bundle_root=self.root,
        )

    def training_lines(self, training_stress: bool) -> tuple[str, ...]:
        """Return the native continuation command and final checkpoint copy."""
        del training_stress
        epoch = (self.spec.training_parameters or {})["epoch"]
        work = f"{self.files.training_dir}/sevennet"
        return (
            command("mkdir", "-p", work),
            command(
                "sevenn",
                "train",
                f"{self.files.training_dir}/sevennet_resolved.yaml",
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


__all__ = ["SevenNetAdapter"]
