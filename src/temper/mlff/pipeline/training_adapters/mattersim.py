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
        """Return the CUDA launcher command with bundle options and data paths."""
        parameters = dict(self.spec.training_parameters or {})
        parameters["early_stop_patience"] = int(parameters["epochs"]) + 1
        parameters["include_forces"] = True
        parameters["include_stresses"] = training_stress
        if not training_stress:
            parameters["stress_loss_ratio"] = 0.0
        work = self.output_path("native_outputs")
        assert self.files is not None
        arguments = [
            "$PYTHON_BIN",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=1",
            "--module",
            "temper.mlff.pipeline.mattersim_final",
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

    def train(self) -> None:
        """Train without weight averaging and verify the final epoch was reached.

        Notes
        -----
        The launcher runs the native CLI with EMA decay zero. The last_model
        checkpoint therefore contains final-step weights. Native epoch indices
        start at zero; validation-best checkpoints are never used for evaluation.

        Raises
        ------
        RuntimeError
            If the exported checkpoint does not record the configured last epoch.
        OSError
            If the exported checkpoint cannot be read.
        subprocess.CalledProcessError
            If native training fails.
        """
        import torch

        super().train()
        checkpoint = torch.load(self.root / self.trained_model_path,
                                map_location="cpu", weights_only=False)
        epochs = int((self.spec.training_parameters or {})["epochs"])
        if checkpoint.get("last_epoch") != epochs - 1:
            raise RuntimeError("MatterSim did not complete the configured final epoch.")
