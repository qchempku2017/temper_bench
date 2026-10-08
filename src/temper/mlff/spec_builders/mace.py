"""MACE naive fine-tuning defaults."""
from temper.mlff.spec_builders.base import BaseSpecBuilder
from temper.schemas.mlff_spec import MLFFImplementation


@BaseSpecBuilder.register(name="mace")
class MACESpecBuilder(BaseSpecBuilder):
    """Build a MACE recipe; see BaseSpecBuilder for source and override options."""

    mlff_type = "mace"
    model_name = "MACE"
    model_version = "2024.0"
    model_filename = "mace.model"
    implementations = (MLFFImplementation(name="mace-torch", version="0.3.16"),)
    epoch_key = "max_num_epochs"
    patience_key = "patience"
    training_defaults = {
        "multiheads_finetuning": False,
        "valid_fraction": 0.05,
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
        "ema": True,
        "ema_decay": 0.999,
        "amsgrad": True,
        "clip_grad": 1.0,
        "default_dtype": "float32",
        "seed": 42,
    }
