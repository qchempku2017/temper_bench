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

    pretrained_model_filename = "sevennet.pth"
    training_input_filenames = {"script": "run.sh", "config": "sevennet.yaml"}
    training_output_filenames = {
        "trained_model": "finetuned_sevennet.pth", "training_log": "training.log",
        "resolved_config": "sevennet_resolved.yaml", "native_outputs": "sevennet",
    }

    def _training_file_contents(self, training_stress: bool) -> dict[str, str]:
        """Return YAML that fine-tunes SevenNet-0 via train.continue.

        The checkpoint supplies pretrained weights and species statistics;
        optimizer, scheduler and epoch counters restart for this dataset.
        The runner reads architecture from the checkpoint before training.
        """
        parameters = dict(self.spec.training_parameters or {})
        model = {
            "train_shift_scale": parameters.pop("train_shift_scale"),
            "train_denominator": parameters.pop("train_denominator"),
        }
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
        assert self.files is not None
        train["continue"] = {
            "reset_optimizer": True,
            "reset_scheduler": True,
            "reset_epoch": True,
            "checkpoint": self.files.pretrained_model,
        }
        data = {
            "batch_size": batch_size,
            "data_divide_ratio": data_divide_ratio,
            "data_format_args": {"index": ":"},
            "load_trainset_path": [
                self.input_path("train")
            ],
        }
        if self.training_unit.val_set is not None:
            data["load_validset_path"] = [
                self.input_path("validation")
            ]
        return {
            self.input_path("config"): yaml_text(
                {"model": model, "train": train, "data": data}
            )
        }

    def prepare_training(self) -> None:
        """Resolve the checkpoint architecture before native SevenNet training."""
        from temper.mlff.pipeline.data_preparation.sevennet import prepare_config

        prepare_config(
            self.root / self.input_path("config"),
            self.root / self.output_path("resolved_config"),
            bundle_root=self.root,
        )

    def _training_commands(self, training_stress: bool) -> tuple[str, ...]:
        """Return the native continuation command and final checkpoint copy."""
        del training_stress
        epoch = (self.spec.training_parameters or {})["epoch"]
        work = self.output_path("native_outputs")
        return (
            command("mkdir", "-p", work),
            command(
                "sevenn",
                "train",
                self.output_path("resolved_config"),
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
