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

    mlff_type = "mace"
    calculator_module = "temper.mlff.pipeline.calculators.mace"
    model_filenames = {"model": "mace.model"}
    trained_model_filename = "finetuned_mace.model"

    def generated_training_files(self, training_stress: bool) -> dict[str, str]:
        """Return native YAML with bundle paths and dataset-dependent stress."""
        config = dict(self.spec.training_parameters or {})
        work = f"{self.files.training_dir}/mace"
        config.update(
            {
                "name": "temper_model",
                "foundation_model": self.artifact_path("model"),
                "train_file": self.dataset_path(self.training_unit.train_set),
                "model_dir": f"{work}/models",
                "checkpoints_dir": f"{work}/checkpoints",
                "results_dir": f"{work}/results",
                "log_dir": f"{work}/logs",
            }
        )
        if not training_stress:
            config["stress_weight"] = 0.0
            config["loss"] = "weighted"
        config["compute_stress"] = training_stress

        if self.training_unit.val_set is not None:
            config["valid_file"] = (
                self.dataset_path(self.training_unit.val_set)
            )
            config.pop("valid_fraction", None)
        return {f"{self.files.training_dir}/mace.yaml": yaml_text(config)}

    def training_lines(self, training_stress: bool) -> tuple[str, ...]:
        """Return CUDA fine-tuning and trained-model copy commands."""
        del training_stress
        work = f"{self.files.training_dir}/mace"
        return (
            command("mkdir", "-p", f"{work}/models"),
            command(
                "mace_run_train",
                "--config",
                f"{self.files.training_dir}/mace.yaml",
                "--device",
                "cuda",
            ),
            command(
                "cp",
                f"{work}/models/temper_model.model",
                self.trained_model_path,
            ),
        )


__all__ = ["MACEAdapter"]
