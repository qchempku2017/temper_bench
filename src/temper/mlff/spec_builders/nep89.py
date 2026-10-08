"""NEP-89 TorchNEP fine-tuning defaults."""
from __future__ import annotations
from typing import Any
from temper.mlff.spec_builders.base import BaseSpecBuilder
from temper.schemas.mlff_spec import MLFFImplementation

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



@BaseSpecBuilder.register(name="nep89")
class NEP89SpecBuilder(BaseSpecBuilder):
    """Build NEP-89 recipes with architecture owned by the pretrained file."""

    mlff_type = "nep89"
    model_name = "NEP-89"
    model_version = "2025.1"
    model_filename = "nep89.txt"
    implementations = (
        MLFFImplementation(name="gpumd", version="5.7", kind="executable"),
        MLFFImplementation(name="calorine", version="3.5"),
    )
    training_implementations = (MLFFImplementation(name="torchnep", version="1.0.2"),)
    training_defaults = _TRAINING_DEFAULTS
    epoch_key = "epoch"

    def prepare_training(self) -> dict[str, Any]:
        """Return native controls, rejecting incompatible architecture overrides."""
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
        training = super().prepare_training()
        if training["stage2"] not in (0, 1, False, True):
            raise ValueError("TorchNEP 'stage2' must be 0 or 1.")
        return training
