"""MatterSim runtime adapter."""

from __future__ import annotations

from typing import Any

from temper.mlff.pipeline.training_adapters.base import BaseMLFFAdapter, command


_BOOLEAN_OPTIONS = {
    "include_forces",
    "include_stresses",
    "re_normalize",
    "trainable_scale",
    "trainable_shift",
}


@BaseMLFFAdapter.register(name="mattersim")
class MatterSimAdapter(BaseMLFFAdapter):
    """Prepare MatterSim 1.2.5 train-and-test bundles."""

    pretrained_model_filename = "mattersim.pth"
    training_output_filenames = {
        "trained_model": "finetuned_mattersim.pth", "training_log": "training.log",
        "native_outputs": "mattersim",
    }

    def _training_file_contents(self, training_stress: bool) -> dict[str, str]:
        """Return no config files: MatterSim accepts all controls on its CLI."""
        return {}

    @staticmethod
    def _options(parameters: dict[str, Any]) -> list[str]:
        """Return native CLI arguments, preserving false boolean switches."""
        result: list[str] = []
        for key, value in parameters.items():
            option = f"--{key}"
            if key in _BOOLEAN_OPTIONS:
                result.append(option if value else f"--no-{key}")
            elif value is not None:
                result.extend((option, str(value)))
        return result

    def _training_commands(self, training_stress: bool) -> tuple[str, ...]:
        """Return the CUDA launcher command with recipe options and data paths."""
        parameters = dict(self.spec.training_parameters or {})
        parameters["include_forces"] = True
        parameters["include_stresses"] = training_stress
        if not training_stress:
            parameters["stress_loss_ratio"] = 0.0
        work = self.output_path("native_outputs")
        arguments = [
            "$PYTHON_BIN",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=1",
            "--module",
            "mattersim.training.finetune_mattersim",
            "--train_data_path",
            self.input_path("train"),
            "--load_model_path",
            self.files.pretrained_model,
            "--save_path",
            work,
            "--save_checkpoint",
            "--device",
            "cuda",
            *self._options(parameters),
        ]
        if self.training_unit.val_set is not None:
            arguments.extend(
                (
                    "--valid_data_path",
                    self.input_path("validation"),
                )
            )
        return (
            command("mkdir", "-p", work),
            command(*arguments),
            command(
                "cp", f"{work}/last_model.pth", self.trained_model_path
            ),
        )
