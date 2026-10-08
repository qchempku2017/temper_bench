"""MACE submit-folder writer."""

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


@BaseMLFFBundleWriter.register(name="mace")
class MACEBundleWriter(BaseMLFFBundleWriter):
    """Write fixed-layout mace-torch 0.3.16 train-and-test bundles."""

    mlff_type = "mace"
    calculator_resource = "calculators/mace.py"
    model_filenames = {"model": "mace.model"}
    trained_model_filename = "finetuned_mace.model"

    def extra_runtime_resources(self) -> dict[str, str]:
        """Return no extra resources beyond the shared runtime."""
        return {}

    def generated_training_files(self, training_stress: bool) -> dict[str, str]:
        """Return native YAML with bundle paths and dataset-dependent stress."""
        config = dict(self.spec.training_parameters or {})
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
            }
        )
        if not training_stress:
            config["stress_weight"] = 0.0
            config["loss"] = "weighted"
        config["compute_stress"] = training_stress

        if self.training_unit.val_set is not None:
            config["valid_file"] = (
                f"{DEFAULT_MLFF_DATASETS_DIR}/validation.extxyz"
            )
            config.pop("valid_fraction", None)
        return {f"{DEFAULT_MLFF_TRAINING_DIR}/mace.yaml": yaml_text(config)}

    def training_lines(self, training_stress: bool) -> tuple[str, ...]:
        """Return CUDA fine-tuning and trained-model copy commands."""
        del training_stress
        work = f"{DEFAULT_MLFF_TRAINING_DIR}/mace"
        return (
            command("mkdir", "-p", f"{work}/models"),
            command(
                "mace_run_train",
                "--config",
                f"{DEFAULT_MLFF_TRAINING_DIR}/mace.yaml",
                "--device",
                "cuda",
            ),
            command(
                "cp",
                f"{work}/models/temper_model.model",
                self.trained_model_path,
            ),
        )


__all__ = ["MACEBundleWriter"]
