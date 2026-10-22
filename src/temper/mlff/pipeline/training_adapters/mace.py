"""MACE runtime adapter."""

from __future__ import annotations


from temper.mlff.pipeline.training_adapters.base import (
    BaseMLFFAdapter,
    command,
    yaml_text,
)


@BaseMLFFAdapter.register(name="mace")
class MACEAdapter(BaseMLFFAdapter):
    """Prepare mace-torch 0.3.16 train-and-test bundles."""

    pretrained_model_filename = "mace.model"
    training_input_filenames = {"script": "run.sh", "config": "mace.yaml"}
    training_output_filenames = {
        "trained_model": "finetuned_mace.model", "training_log": "training.log",
        "native_models": "mace/models",
        "checkpoints": "mace/checkpoints", "logs": "mace/logs",
        "metrics": "mace/results",
    }

    def _training_file_contents(self, training_stress: bool) -> dict[str, str]:
        """Return native YAML with bundle paths and dataset-dependent stress."""
        config = dict(self.spec.training_parameters or {})
        # MACE otherwise reloads the latest *best* checkpoint before export.
        # Save every evaluated epoch and disable early termination/averaging.
        config.update(save_all_checkpoints=True, eval_interval=1,
                      patience=int(config["max_num_epochs"]) + 1,
                      swa=False, ema=False, restart_latest=False)
        assert self.files is not None
        config.update(
            {
                "name": "temper_model",
                "foundation_model": self.files.pretrained_model,
                "train_file": self.input_path("train"),
                "model_dir": self.output_path("native_models"),
                "checkpoints_dir": self.output_path("checkpoints"),
                "results_dir": self.output_path("metrics"),
                "log_dir": self.output_path("logs"),
            }
        )
        if not training_stress:
            config["stress_weight"] = 0.0
            config["loss"] = "weighted"
        config["compute_stress"] = training_stress

        if self.training_unit.val_set is not None:
            config["valid_file"] = (
                self.input_path("validation")
            )
            config.pop("valid_fraction", None)
        return {self.input_path("config"): yaml_text(config)}

    def _training_commands(self, training_stress: bool) -> tuple[str, ...]:
        """Return CUDA fine-tuning and trained-model copy commands."""
        del training_stress
        return (
            command("mkdir", "-p", self.output_path("native_models")),
            command(
                "mace_run_train",
                "--config",
                self.input_path("config"),
                "--device",
                "cuda",
            ),
            command(
                "cp",
                f"{self.output_path('native_models')}/temper_model.model",
                self.trained_model_path,
            ),
        )

    def train(self) -> None:
        """Train and require a checkpoint from the configured final epoch.

        Notes
        -----
        Native MACE epochs are zero-based. Export uses the latest checkpoint;
        averaging and early stopping are disabled in the generated config.

        Raises
        ------
        RuntimeError
            If training returns without writing its final epoch checkpoint.
        subprocess.CalledProcessError
            If the native training command fails.
        """
        super().train()
        parameters = self.spec.training_parameters or {}
        epoch = int(parameters["max_num_epochs"]) - 1
        seed = parameters.get("seed", 123)
        checkpoint = self.root / self.output_path("checkpoints") / f"temper_model_run-{seed}_epoch-{epoch}.pt"
        if not checkpoint.is_file():
            raise RuntimeError("MACE did not produce the configured final epoch checkpoint.")
