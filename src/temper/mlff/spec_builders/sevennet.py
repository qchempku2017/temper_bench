"""SevenNet specification builder."""

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
    "random_seed": 1,
    "epoch": 100,
    "loss": "Huber",
    "loss_param": {"delta": 0.01},
    "optimizer": "adam",
    "optim_param": {"lr": 0.004},
    "scheduler": "exponentiallr",
    "scheduler_param": {"gamma": 0.99},
    "force_loss_weight": 1.0,
    "stress_loss_weight": 0.01,
    "per_epoch": 10,
    "batch_size": 4,
    "data_divide_ratio": 0.1,
    "train_shift_scale": False,
    "train_denominator": False,
}


class SevenNetSpecBuilder:
    """Build a SevenNet 0.13.0 specification using a local checkpoint.

    Parameters
    ----------
    pretrained_model_path : str, pathlib.Path, or None, optional
        SevenNet-0 checkpoint to copy. Defaults to sevennet.pth below
        DEFAULT_MLFF_PRETRAINED_MODELS_DIR.
    training_parameters : dict[str, Any] or None, optional
        Values used in the native SevenNet training configuration. None means
        zero-shot; an empty dictionary enables fine-tuning defaults.
    testing_parameters : dict[str, Any] or None, optional
        Keyword arguments forwarded to SevenNetCalculator, excluding device.
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
        """Build the content-addressed SevenNet specification.

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
            Path(DEFAULT_MLFF_PRETRAINED_MODELS_DIR) / "sevennet.pth"
            if self.pretrained_model_path is None
            else self.pretrained_model_path
        )
        training = None
        if self.training_parameters is not None:
            training = deepcopy(_TRAINING_DEFAULTS)
            training.update(deepcopy(self.training_parameters))
            validate_epoch(training, "epoch", "SevenNet")
        return MLFFSpec(
            mlff_type="sevennet",
            implementations=(
                MLFFImplementation(name="sevenn", version="0.13.0"),
            ),
            pretrained_model=PretrainedMLFFSpec(
                name="SevenNet-0",
                version="11July2024",
                artifacts={"model": LocalArtifactRef.from_path(path)},
            ),
            training=training,
            testing=(
                deepcopy(self.testing_parameters)
                if self.testing_parameters is not None
                else {}
            ),
        )


__all__ = ["SevenNetSpecBuilder"]
