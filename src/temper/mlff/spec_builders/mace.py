"""MACE naive fine-tuning defaults."""
from typing import Any
from temper.mlff.spec_builders.base import BaseSpecBuilder
from temper.schemas.mlff_spec import MLFFImplementation


@BaseSpecBuilder.register(name="mace")
class MACESpecBuilder(BaseSpecBuilder):
    """Build a MACE recipe; see BaseSpecBuilder for source and override options."""

    mlff_type = "mace"
    implementations = (MLFFImplementation(name="mace-torch", version="0.3.16"),)
    epoch_key = "max_num_epochs"
    patience_key = "patience"
    training_defaults = {
        "multiheads_finetuning": False,
        "energy_weight": 10.0,
        "forces_weight": 10.0,
        "stress_weight": 1.0,
        "loss": "stress",
        "E0s": "estimated",
        "lr": 0.001,
        "weight_decay": 0.0,
        "scaling": "rms_forces_scaling",
        "batch_size": 4,
        "max_num_epochs": 100,
        "ema": False,
        "swa": False,
        "save_all_checkpoints": True,
        "eval_interval": 1,
        "restart_latest": False,
        "ema_decay": 0.999,
        "amsgrad": True,
        "clip_grad": 1.0,
        "default_dtype": "float32",
        "seed": 42,
    }

    def prepare_training(self) -> dict[str, Any]:
        """Build parameters that export the configured final epoch weights.

        Returns
        -------
        dict[str, Any]
            Native parameters with averaging disabled and every epoch saved.

        Raises
        ------
        ValueError
            If overrides enable averaging, checkpoint restart, sparse checkpoint
            evaluation, or best-only checkpoint storage, or epochs are invalid.
        """
        parameters = super().prepare_training()
        for key, expected in {"ema": False, "swa": False, "restart_latest": False,
                              "save_all_checkpoints": True, "eval_interval": 1}.items():
            if parameters.get(key) != expected:
                raise ValueError(f"{key} must be {expected!r} for final-epoch evaluation.")
        return parameters
