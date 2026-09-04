"""NEP-89 specification builder."""

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
    "epoch": 100,
    "batch": 32,
    "lr": 0.01,
    "stop_lr": 1e-6,
    "lambda_e": 0.01,
    "lambda_f": 1.0,
    "lambda_v": 0.01,
    "max_grad_norm": 10.0,
    "lr_scheduler": "plateau",
    "scheduler_patience": 15,
    "early_stop": 0,
    "scheduler_factor": 0.7,
    "stage2": 0,
    "stage2_lr": 1e-3,
    "stage2_lambda_e": 1.0,
    "stage2_lambda_f": 0.05,
    "stage2_lambda_v": 0.1,
    "weight_decay": 1e-4,
}
_OPTIONAL_TRAINING_KEYS = {
    "start_stage2",
    "stage2_scheduler_patience",
    "stage2_scheduler_factor",
}
_ARCHITECTURE_KEYS = {
    "type",
    "version",
    "zbl",
    "use_typewise_cutoff_zbl",
    "cutoff",
    "n_max",
    "basis_size",
    "l_max",
    "neuron",
}
_LEGACY_TRAINING_KEYS = {
    "generation",
    "population",
    "save_potential",
    "lambda_1",
    "lambda_2",
    "pos_noise",
}


class NEP89SpecBuilder:
    """Build a TorchNEP 1.0.2 and calorine 3.5 NEP-89 specification.

    Parameters
    ----------
    pretrained_model_path : str, pathlib.Path, or None, optional
        NEP-89 potential. Defaults to nep89.txt below
        DEFAULT_MLFF_PRETRAINED_MODELS_DIR.
    training_parameters : dict[str, Any] or None, optional
        Native TorchNEP hyperparameters. Architecture is derived from the
        pretrained potential. None means zero-shot; an empty dictionary enables
        fine-tuning defaults.
    testing_parameters : dict[str, Any] or None, optional
        Keyword arguments forwarded to calorine's selected Calculator,
        excluding device.
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
        """Build the content-addressed NEP-89 specification.

        Returns
        -------
        MLFFSpec
            Persistable zero-shot or fine-tuning recipe.

        Raises
        ------
        ValueError
            If the model is missing or training controls are incompatible.
        """
        root = Path(DEFAULT_MLFF_PRETRAINED_MODELS_DIR)
        training = None
        if self.training_parameters is not None:
            keys = self.training_parameters.keys()
            architecture = sorted(_ARCHITECTURE_KEYS & keys)
            if architecture:
                raise ValueError(
                    "TorchNEP architecture is derived from the pretrained "
                    f"nep.txt; remove overrides {architecture!r}."
                )
            legacy = sorted(_LEGACY_TRAINING_KEYS & keys)
            if legacy:
                raise ValueError(
                    "GPUMD/SNES training parameters are not supported by "
                    f"TorchNEP: {legacy!r}."
                )
            unknown = sorted(
                set(keys) - set(_TRAINING_DEFAULTS) - _OPTIONAL_TRAINING_KEYS
            )
            if unknown:
                raise ValueError(
                    f"Unsupported TorchNEP training parameters: {unknown!r}."
                )
            if self.training_parameters.get("early_stop", 0) != 0:
                raise ValueError(
                    "TorchNEP 'early_stop' must be 0 so every configured "
                    "epoch is completed."
                )
            training = deepcopy(_TRAINING_DEFAULTS)
            training.update(deepcopy(self.training_parameters))
            validate_epoch(training, "epoch", "TorchNEP")
            if training["stage2"] not in (0, 1, False, True):
                raise ValueError("TorchNEP 'stage2' must be 0 or 1.")
            training["early_stop"] = 0
        return MLFFSpec(
            mlff_type="nep89",
            implementations=(
                *(
                    (MLFFImplementation(name="torchnep", version="1.0.2"),)
                    if training is not None
                    else ()
                ),
                MLFFImplementation(
                    name="gpumd", version="5.7", kind="executable"
                ),
                MLFFImplementation(name="calorine", version="3.5"),
            ),
            pretrained_model=PretrainedMLFFSpec(
                name="NEP-89",
                version="2025.1",
                artifacts={
                    "model": LocalArtifactRef.from_path(
                        root / "nep89.txt"
                        if self.pretrained_model_path is None
                        else self.pretrained_model_path
                    ),
                },
            ),
            training=training,
            testing=(
                deepcopy(self.testing_parameters)
                if self.testing_parameters is not None
                else {}
            ),
        )


__all__ = ["NEP89SpecBuilder"]
