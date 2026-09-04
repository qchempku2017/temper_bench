"""MatterSim specification builder."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from temper.mlff.spec_builders._training import validate_epoch
from temper.schemas.mlff_spec import (
    LocalArtifactRef,
    MLFFImplementation,
    MLFFSpec,
    PretrainedMLFFSpec,
)
from temper.utils.defaults import DEFAULT_MLFF_PRETRAINED_MODELS_DIR


_TRAINING_DEFAULTS: dict[str, Any] = {
    "run_name": "temper",
    "epochs": 100,
    "batch_size": 1,
    "lr": 2e-4,
    "step_size": 10,
    "force_loss_ratio": 1.0,
    "stress_loss_ratio": 0.1,
    "early_stop_patience": 101,
    "seed": 42,
    "re_normalize": False,
    "scale_key": "per_species_forces_rms",
    "shift_key": "per_species_energy_mean_linear_reg",
    "init_scale": None,
    "init_shift": None,
    "trainable_scale": False,
    "trainable_shift": False,
    "ckpt_interval": 10,
    "cutoff": 5.0,
    "threebody_cutoff": 4.0,
}


class MatterSimSpecBuilder:
    """Build a MatterSim 1.2.5 specification using a local checkpoint.

    Parameters
    ----------
    pretrained_model_path : str, pathlib.Path, or None, optional
        Checkpoint to copy. Defaults to mattersim.pth below
        DEFAULT_MLFF_PRETRAINED_MODELS_DIR.
    training_parameters : dict[str, Any] or None, optional
        MatterSim launcher options. None means zero-shot; an empty dictionary
        enables defaults. Batch size is fixed to one and early stopping is
        placed beyond the configured ``epochs`` by the writer.
    testing_parameters : dict[str, Any] or None, optional
        Keyword arguments forwarded to MatterSimCalculator, excluding device.
    """

    def __init__(
        self,
        *,
        pretrained_model_path: str | Path | None = None,
        training_parameters: dict[str, Any] | None = None,
        testing_parameters: dict[str, Any] | None = None,
    ) -> None:
        self.pretrained_model_path = pretrained_model_path
        self.training_parameters = training_parameters
        self.testing_parameters = testing_parameters

    def build(self) -> MLFFSpec:
        """Build the content-addressed MatterSim specification.

        Returns
        -------
        MLFFSpec
            Persistable zero-shot or fine-tuning recipe.

        Raises
        ------
        ValueError
            If the checkpoint is missing.
        """
        path = (
            Path(DEFAULT_MLFF_PRETRAINED_MODELS_DIR) / "mattersim.pth"
            if self.pretrained_model_path is None
            else self.pretrained_model_path
        )
        training = None
        if self.training_parameters is not None:
            if "early_stop_patience" in self.training_parameters:
                raise ValueError(
                    "MatterSim 'early_stop_patience' is managed by TEMPER so "
                    "training runs the configured number of epochs."
                )
            training = deepcopy(_TRAINING_DEFAULTS)
            training.update(deepcopy(self.training_parameters))
            epochs = validate_epoch(training, "epochs", "MatterSim")
            training["early_stop_patience"] = epochs + 1
        return MLFFSpec(
            mlff_type="mattersim",
            implementations=(
                MLFFImplementation(name="mattersim", version="1.2.5"),
            ),
            pretrained_model=PretrainedMLFFSpec(
                name="MatterSim-v1-5M",
                version="1.0.0",
                artifacts={"model": LocalArtifactRef.from_path(path)},
            ),
            training=training,
            testing=(
                deepcopy(self.testing_parameters)
                if self.testing_parameters is not None
                else {}
            ),
        )


__all__ = ["MatterSimSpecBuilder"]
