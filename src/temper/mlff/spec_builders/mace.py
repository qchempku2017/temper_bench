"""MACE specification builder."""

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
    "multiheads_finetuning": False,
    "valid_fraction": 0.05,
    "energy_weight": 1.0,
    "forces_weight": 1.0,
    "E0s": "average",
    "lr": 0.01,
    "scaling": "rms_forces_scaling",
    "batch_size": 2,
    "max_num_epochs": 100,
    "patience": 101,
    "ema": True,
    "ema_decay": 0.99,
    "amsgrad": True,
    "default_dtype": "float64",
    "seed": 3,
}


class MACESpecBuilder:
    """Build a mace-torch 0.3.16 specification using a local model.

    Parameters
    ----------
    pretrained_model_path : str, pathlib.Path, or None, optional
        Foundation model to copy. Defaults to mace.model below
        DEFAULT_MLFF_PRETRAINED_MODELS_DIR.
    training_parameters : dict[str, Any] or None, optional
        Native mace_run_train configuration values. None means zero-shot; an
        empty dictionary enables defaults. ``max_num_epochs`` controls length;
        TEMPER manages patience so all epochs run.
    testing_parameters : dict[str, Any] or None, optional
        Keyword arguments forwarded to MACECalculator, excluding device.
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
        """Build the content-addressed MACE specification.

        Returns
        -------
        MLFFSpec
            Persistable zero-shot or fine-tuning recipe.

        Raises
        ------
        ValueError
            If the model is missing.
        """
        path = (
            Path(DEFAULT_MLFF_PRETRAINED_MODELS_DIR) / "mace.model"
            if self.pretrained_model_path is None
            else self.pretrained_model_path
        )
        training = None
        if self.training_parameters is not None:
            if "patience" in self.training_parameters:
                raise ValueError(
                    "MACE 'patience' is managed by TEMPER so training runs "
                    "the configured number of epochs."
                )
            training = deepcopy(_TRAINING_DEFAULTS)
            training.update(deepcopy(self.training_parameters))
            epochs = validate_epoch(training, "max_num_epochs", "MACE")
            training["patience"] = epochs + 1
        return MLFFSpec(
            mlff_type="mace",
            implementations=(
                MLFFImplementation(name="mace-torch", version="0.3.16"),
            ),
            pretrained_model=PretrainedMLFFSpec(
                name="MACE",
                version="2024.0",
                artifacts={"model": LocalArtifactRef.from_path(path)},
            ),
            training=training,
            testing=(
                deepcopy(self.testing_parameters)
                if self.testing_parameters is not None
                else {}
            ),
        )


__all__ = ["MACESpecBuilder"]
