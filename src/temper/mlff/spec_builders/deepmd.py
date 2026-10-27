"""DeepMD fine-tuning policy, independent of submitted datasets."""
from __future__ import annotations

from abc import ABC
from typing import Any

from temper.mlff.spec_builders.base import BaseSpecBuilder
from temper.schemas.mlff_spec import MLFFImplementation


class _DeepMDSpecBuilder(BaseSpecBuilder, ABC):
    """Share DPA training policy; instances select the pretrained release."""

    implementations = (MLFFImplementation(name="deepmd-kit", version="3.2.0"),)
    epoch_key = "numb_epoch"
    training_defaults = {
        "numb_epoch": 60,
        "save_freq": 400,
        "disp_freq": 100,
        "seed": 42,
        "gradient_max_norm": 1.0,
        "training_data": {"batch_size": "auto:128"},
        "validation_data": {"batch_size": "auto:128"},
        "optimizer": {"type": "HybridMuon", "weight_decay": 0.001},
        "learning_rate": {"type": "exp", "start_lr": 1e-4, "stop_lr": 1e-6},
        "loss": {
            "type": "ener",
            "loss_func": "mae",
            "f_use_norm": True,
            "start_pref_e": 20.0,
            "limit_pref_e": 20.0,
            "start_pref_f": 20.0,
            "limit_pref_f": 20.0,
            "start_pref_v": 5.0,
            "limit_pref_v": 5.0,
        },
    }

    def prepare_training(self) -> dict[str, Any]:
        """Return training controls plus native loss and learning_rate sections."""
        aliases = {
            "numb_steps", "stop_batch", "num_step", "num_steps",
            "numb_step", "num_epochs", "num_epoch", "numb_epochs",
        }
        # Intercept illegal aliases for DeepMD training length, which must be specified as 'numb_epoch'.
        if aliases & (self.training_parameters or {}).keys():
            raise ValueError("DeepMD training length must use canonical 'numb_epoch'.")
        return super().prepare_training()


@BaseSpecBuilder.register(name="dpa4")
class DPA4SpecBuilder(_DeepMDSpecBuilder):
    """Build a selected DPA-4 checkpoint specification with 60 training epochs."""

    mlff_type = "dpa4"


@BaseSpecBuilder.register(name="dpa4c")
class DPA4CSpecBuilder(_DeepMDSpecBuilder):
    """Build a selected DPA-4C checkpoint specification with 100 training epochs."""

    mlff_type = "dpa4c"
    training_defaults = {**_DeepMDSpecBuilder.training_defaults, "numb_epoch": 100}
